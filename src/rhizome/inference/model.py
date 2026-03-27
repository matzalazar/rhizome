"""
ONNX model management and inference — no framework dependencies.

Files are downloaded once via HTTP from the Xenova pre-exported model on
HuggingFace.  At inference time the only runtime dependencies are
onnxruntime and the Rust-backed `tokenizers` library.

Why this approach over optimum/transformers:
  - No PyTorch, no Transformers, no ONNX export step.
  - The Xenova project publishes ready-to-use ONNX exports for hundreds of
    sentence-transformer models; we just download the two files we need.
  - Cold-start is faster (HTTP fetch vs. framework import + conversion).
  - Dependency footprint is a fraction of the transformers stack.

Why mean pooling with attention mask?
  The model was fine-tuned with a mean-pooling objective over the full
  sequence.  Averaging only the real (non-padding) token hidden states
  produces a better sentence representation than using the CLS token alone,
  which was designed for classification pre-training, not sentence similarity.
"""

import threading
from pathlib import Path

import numpy as np
import numpy.typing as npt
import onnxruntime as ort
from loguru import logger

_DEFAULT_MODEL_NAME = "Xenova/paraphrase-multilingual-MiniLM-L12-v2"
_HF_BASE = "https://huggingface.co"

# Default chunk size and overlap — mirror the defaults in config.py so that
# callers outside the pipeline (tests, scripts) get sensible behaviour without
# needing a Settings object.  The pipeline always passes explicit values from
# settings.chunk_size / settings.chunk_overlap.
# chunk_size=0 disables chunking: the full token sequence is passed to the model
# and truncated at the hardware limit (512 for MiniLM).
_MAX_TOKENS = 512
_CHUNK_OVERLAP = 32


def _model_urls(model_name: str) -> tuple[str, str]:
    """Derive ONNX model and tokenizer URLs from a HuggingFace model name."""
    base = f"{_HF_BASE}/{model_name}/resolve/main"
    return f"{base}/onnx/model.onnx", f"{base}/tokenizer.json"

# ---------------------------------------------------------------------------
# Chunking helper (public for unit testing)
# ---------------------------------------------------------------------------

def chunk_token_ids(
    ids: list[int],
    max_tokens: int = _MAX_TOKENS,
    overlap: int = _CHUNK_OVERLAP,
) -> list[list[int]]:
    """
    Split a flat token-ID sequence into overlapping windows.

    When len(ids) <= max_tokens, a single-element list containing all ids
    is returned (no chunking needed).  Otherwise, consecutive windows of
    *max_tokens* tokens are produced with a stride of (max_tokens - overlap),
    so adjacent chunks share *overlap* tokens to preserve context across
    boundaries.  The final chunk always extends to the last token, even when
    it is shorter than max_tokens.
    """
    if len(ids) <= max_tokens:
        return [ids]
    stride = max_tokens - overlap
    chunks: list[list[int]] = []
    start = 0
    while start < len(ids):
        end = min(start + max_tokens, len(ids))
        chunks.append(ids[start:end])
        if end == len(ids):
            break
        start += stride
    return chunks


# ---------------------------------------------------------------------------
# Module-level singleton state
# ---------------------------------------------------------------------------

_model_lock = threading.Lock()
_model: "PureONNXEmbeddingModel | None" = None
_model_error: Exception | None = None


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------

class PureONNXEmbeddingModel:
    """
    Sentence embedding model backed by ONNX Runtime and the Rust tokenizer.

    Lifecycle:
      1. Construct with the model directory path — lightweight, no I/O.
      2. Call _load_model() to download files (idempotent) and initialise
         the ORT session and tokenizer.  This is the expensive step.
      3. Call encode() for inference.  Session is reused across calls.

    The separation of construction from initialisation lets the CLI pre-cache
    model files without holding a live ORT session in memory:
        PureONNXEmbeddingModel(model_dir)._load_model()
    """

    def __init__(self, model_dir: Path, model_name: str = _DEFAULT_MODEL_NAME) -> None:
        self.model_dir = model_dir
        self.model_name = model_name
        self.model_path = model_dir / "model.onnx"
        self.tokenizer_path = model_dir / "tokenizer.json"
        self.MODEL_URL, self.TOKENIZER_URL = _model_urls(model_name)
        # session and tokenizer are initialised by _load_model(), not here.
        self.session: ort.InferenceSession | None = None
        self.tokenizer = None
        self._input_names: set[str] = set()

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def _download_file(self, url: str, dest: Path) -> None:
        """
        Download *url* to *dest* using chunked streaming.

        Idempotent: if *dest* already exists the download is skipped.
        On any failure the partial file is deleted so the next call
        can start clean — never leave a corrupted cache.
        """
        if dest.exists():
            logger.info(f"Using cached {dest.name}")
            return

        logger.info(f"Downloading {dest.name} ...")
        dest.parent.mkdir(parents=True, exist_ok=True)

        try:
            import requests

            with requests.get(url, stream=True, timeout=60) as response:
                response.raise_for_status()
                total_bytes = int(response.headers.get("content-length", 0))
                downloaded = 0

                with dest.open("wb") as fh:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        fh.write(chunk)
                        downloaded += len(chunk)
                        if total_bytes:
                            pct = downloaded / total_bytes * 100
                            logger.debug(
                                f"  {dest.name}: {pct:.0f}%"
                                f" ({downloaded / 1e6:.1f}/{total_bytes / 1e6:.1f} MB)"
                            )

        except Exception as exc:
            dest.unlink(missing_ok=True)
            raise RuntimeError(
                f"Failed to download {dest.name}: {exc}\n"
                "Check your network connection and try again."
            ) from exc

        size_mb = dest.stat().st_size / (1024 * 1024)
        logger.success(f"Downloaded {dest.name} ({size_mb:.1f} MB)")

    def _load_model(self) -> None:
        """
        Download model files if not cached, then initialise the ORT session
        and Rust tokenizer.

        This is the single method that moves the object from the lightweight
        constructed state to the ready-to-infer state.  Calling it on an
        already-initialised instance is safe (downloads are skipped, but the
        session and tokenizer are re-created — prefer the singleton via
        get_model() to avoid that overhead).
        """
        self._download_file(self.MODEL_URL, self.model_path)
        self._download_file(self.TOKENIZER_URL, self.tokenizer_path)

        sess_options = ort.SessionOptions()
        # Two intra-op threads: enough parallelism for matmuls without
        # saturating the CPU when running alongside other processes.
        sess_options.intra_op_num_threads = 2
        sess_options.inter_op_num_threads = 1

        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        self._input_names = {i.name for i in self.session.get_inputs()}

        # Rust-backed tokenizer — fast BPE/WordPiece with no Python overhead.
        # Truncation is disabled: long documents are handled by chunk_token_ids()
        # in encode(), which splits the full token sequence into overlapping
        # windows and averages their embeddings instead of silently dropping tail content.
        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
        self.tokenizer.enable_padding()

        logger.debug("ONNX inference session initialised")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _embed_chunk(self, ids: list[int], mask: list[int]) -> npt.NDArray[np.float32]:
        """
        Run one forward pass for a token sequence that fits within _MAX_TOKENS.

        *ids* and *mask* are already-tokenized sequences (no padding needed for
        single-sequence inference — attention_mask handles real vs. padding tokens).
        Returns an un-normalised (hidden_dim,) embedding.
        """
        input_ids = np.array([ids], dtype=np.int64)
        attention_mask = np.array([mask], dtype=np.int64)
        inputs: dict[str, npt.NDArray] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        # Some Xenova ONNX exports include token_type_ids; supply zeros when present.
        if "token_type_ids" in self._input_names:
            inputs["token_type_ids"] = np.zeros_like(input_ids)

        outputs = self.session.run(None, inputs)  # type: ignore[union-attr]
        token_embeddings = outputs[0][0]          # (seq_len, hidden_dim)
        return mean_pool(token_embeddings, np.array(mask, dtype=float))

    def _embed_chunked(
        self,
        ids: list[int],
        chunk_size: int,
        chunk_overlap: int,
    ) -> npt.NDArray[np.float32]:
        """
        Embed a long token sequence by splitting it into overlapping chunks
        and averaging the resulting embeddings.

        Each chunk is at most *chunk_size* tokens long with *chunk_overlap*
        tokens shared with the next chunk so that sentences split across a
        boundary are still represented in both windows.  The final average gives
        a single document vector with uniform weight across all sections of the note.
        """
        chunks = chunk_token_ids(ids, max_tokens=chunk_size, overlap=chunk_overlap)
        chunk_embeddings = [
            self._embed_chunk(chunk, [1] * len(chunk)) for chunk in chunks
        ]
        logger.debug(f"Long document ({len(ids)} tokens) embedded via {len(chunks)} chunks")
        return np.mean(chunk_embeddings, axis=0).astype(np.float32)

    def encode(
        self,
        texts: list[str],
        normalize: bool = True,
        chunk_size: int = _MAX_TOKENS,
        chunk_overlap: int = _CHUNK_OVERLAP,
    ) -> npt.NDArray[np.float32]:
        """
        Encode *texts* into sentence embeddings.

        Short texts (≤ *chunk_size* tokens) are encoded in a single forward pass.
        Long texts are split into overlapping chunks of *chunk_size* tokens with
        *chunk_overlap* tokens shared between adjacent windows; chunk embeddings
        are averaged into one document vector so the output shape is always (N, 384).

        *chunk_size* and *chunk_overlap* default to the module-level constants but
        should be set from settings.chunk_size / settings.chunk_overlap in the
        pipeline so users can tune them via environment variables.

        When *normalize* is True (default) each vector is L2-normalised so that
        dot product == cosine similarity, which is what similarity.py expects.
        """
        if self.session is None or self.tokenizer is None:
            raise RuntimeError(
                "Model not initialised — call _load_model() before encode()."
            )

        embeddings: list[npt.NDArray[np.float32]] = []

        for text in texts:
            encoding = self.tokenizer.encode(text)
            ids = encoding.ids

            if chunk_size == 0 or len(ids) <= chunk_size:
                embedding = self._embed_chunk(ids, encoding.attention_mask)
            else:
                embedding = self._embed_chunked(ids, chunk_size, chunk_overlap)

            embeddings.append(embedding)

        result = np.stack(embeddings).astype(np.float32)  # (N, 384)

        if normalize:
            result = _l2_normalise(result)

        return result


# ---------------------------------------------------------------------------
# Pooling / normalisation (public for unit testing)
# ---------------------------------------------------------------------------

def mean_pool(
    token_embeddings: npt.NDArray[np.float32],
    attention_mask: npt.NDArray,
) -> npt.NDArray[np.float32]:
    """
    Weighted mean over the token dimension for a single sequence.

    Args:
      token_embeddings: shape (seq_len, hidden_dim)
      attention_mask:   shape (seq_len,) — 1 for real tokens, 0 for padding

    Returns:
      shape (hidden_dim,), dtype float32

    Padding tokens contribute zero to both the numerator (masked out) and
    denominator (not counted) so they cannot dilute the representation.
    """
    mask_expanded = np.expand_dims(attention_mask, axis=-1)        # (seq_len, 1)
    sum_embeddings = np.sum(token_embeddings * mask_expanded, axis=0)  # (hidden_dim,)
    sum_mask = np.sum(mask_expanded, axis=0)                        # (1,) or (hidden_dim,)
    return (sum_embeddings / np.maximum(sum_mask, 1e-9)).astype(np.float32)


def _l2_normalise(embeddings: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """
    Project each row onto the unit hypersphere.

    After normalisation, dot product == cosine similarity — the similarity
    module can use a plain matrix multiply without a per-query division.
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-9, a_max=None)
    return (embeddings / norms).astype(np.float32)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

def get_model(model_dir: Path, model_name: str = _DEFAULT_MODEL_NAME) -> PureONNXEmbeddingModel:
    """
    Return the process-wide model instance, initialising it on first call.

    Uses double-checked locking so concurrent callers (e.g. in a future async
    context) do not race to initialise the session.

    Error caching: if initialisation fails the exception is stored and every
    subsequent call re-raises it immediately, without retrying the expensive
    download and session-load.  To recover, restore the model files and
    restart the process.
    """
    global _model, _model_error

    # Fast path — no lock needed once the model is ready.
    if _model is not None:
        return _model
    if _model_error is not None:
        raise _model_error

    with _model_lock:
        # Re-check inside the lock; another thread may have initialised by now.
        if _model is not None:
            return _model
        if _model_error is not None:
            raise _model_error

        try:
            instance = PureONNXEmbeddingModel(model_dir, model_name)
            instance._load_model()
            _model = instance
        except Exception as exc:
            _model_error = exc
            raise

    return _model
