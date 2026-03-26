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


def _model_urls(model_name: str) -> tuple[str, str]:
    """Derive ONNX model and tokenizer URLs from a HuggingFace model name."""
    base = f"{_HF_BASE}/{model_name}/resolve/main"
    return f"{base}/onnx/model.onnx", f"{base}/tokenizer.json"

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

        # Rust-backed tokenizer — fast BPE/WordPiece with no Python overhead.
        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
        self.tokenizer.enable_truncation(max_length=128)
        self.tokenizer.enable_padding()

        logger.debug("ONNX inference session initialised")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def encode(self, texts: list[str], normalize: bool = True) -> npt.NDArray[np.float32]:
        """
        Encode *texts* into sentence embeddings.

        Returns shape (N, 384) float32.  When *normalize* is True (default)
        each vector is L2-normalised so that dot product == cosine similarity,
        which is what similarity.py expects.
        """
        if self.session is None or self.tokenizer is None:
            raise RuntimeError(
                "Model not initialised — call _load_model() before encode()."
            )

        # Determine once which inputs this export expects.
        input_names = {i.name for i in self.session.get_inputs()}

        embeddings: list[npt.NDArray[np.float32]] = []

        for text in texts:
            encoding = self.tokenizer.encode(text)

            input_ids = np.array([encoding.ids], dtype=np.int64)
            attention_mask = np.array([encoding.attention_mask], dtype=np.int64)

            inputs_onnx: dict[str, npt.NDArray] = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            # Some Xenova ONNX exports include token_type_ids; supply zeros
            # if the graph expects it so the session does not raise.
            if "token_type_ids" in input_names:
                inputs_onnx["token_type_ids"] = np.zeros_like(input_ids)

            outputs = self.session.run(None, inputs_onnx)

            # outputs[0] shape: (1, seq_len, hidden_size)
            token_embeddings = outputs[0][0]  # (seq_len, hidden_size)
            mask = np.array(encoding.attention_mask, dtype=float)

            embedding = mean_pool(token_embeddings, mask)
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
