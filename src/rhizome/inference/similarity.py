"""
Similarity strategies for nearest-neighbour retrieval.

Design: SimilarityStrategy is a runtime_checkable Protocol.  Concrete
implementations (NumpyStrategy for small vaults, HNSWStrategy for large ones)
are selected by the pipeline based on vault size.  The pipeline calls only
.build() and .query(), so strategies are interchangeable without any changes
to pipeline logic.

Why a Protocol instead of an ABC?
  Structural subtyping lets third-party strategies satisfy the interface without
  importing from this module.  A FAISS wrapper, for example, could live in a
  separate package and still be accepted by the pipeline.
"""

from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
from loguru import logger

# Notes beyond this count make exact O(N²) search noticeably slow.
LARGE_VAULT_THRESHOLD = 500


@runtime_checkable
class SimilarityStrategy(Protocol):
    """
    Contract for similarity backends.

    build()  -- index the full embedding matrix once
    query()  -- for each note return the top-K most similar indices + scores
    """

    def build(self, embeddings: npt.NDArray[np.float32]) -> None:
        """Index *embeddings* so that query() can be called."""
        ...

    def query(
        self,
        embeddings: npt.NDArray[np.float32],
        top_k: int,
        threshold: float,
    ) -> list[list[tuple[int, float]]]:
        """
        For every row in *embeddings*, return up to *top_k* (index, score) pairs
        whose cosine similarity exceeds *threshold*, excluding self-matches.

        Returns a list of length N, each entry sorted by score descending.
        """
        ...


class NumpyStrategy:
    """
    Exact pairwise cosine similarity using numpy.

    O(N²) time and memory, but branch-free and fast for N <= 500.
    Embeddings are assumed L2-normalised, so cosine similarity == dot product.
    """

    def __init__(self) -> None:
        self._embeddings: npt.NDArray[np.float32] | None = None

    def build(self, embeddings: npt.NDArray[np.float32]) -> None:
        self._embeddings = embeddings
        logger.debug(f"NumpyStrategy: indexed {len(embeddings)} embeddings (exact search)")

    def query(
        self,
        embeddings: npt.NDArray[np.float32],
        top_k: int,
        threshold: float,
    ) -> list[list[tuple[int, float]]]:
        assert self._embeddings is not None, "call build() before query()"

        # Full (N, N) similarity matrix via a single matrix multiply.
        sim_matrix: npt.NDArray[np.float32] = embeddings @ self._embeddings.T

        results: list[list[tuple[int, float]]] = []
        for i in range(len(embeddings)):
            row = sim_matrix[i].copy()
            row[i] = -1.0  # mask self-similarity

            ranked = np.argsort(row)[::-1]  # descending

            neighbours: list[tuple[int, float]] = []
            for idx in ranked:
                score = float(row[idx])
                if score < threshold:
                    break
                neighbours.append((int(idx), score))
                if len(neighbours) >= top_k:
                    break

            results.append(neighbours)

        return results


class HNSWStrategy:
    """
    Approximate nearest-neighbour search using hnswlib.

    Builds a Hierarchical Navigable Small World graph: O(log N) per query
    instead of O(N).  Suitable for vaults with thousands of notes.

    Trade-off: approximate — a small fraction of true neighbours may be missed.
    For semantic linking this is acceptable; we prefer speed over exhaustive recall.
    """

    _EF_CONSTRUCTION = 200
    _M = 16
    _EF_SEARCH = 50

    def __init__(self) -> None:
        self._index: object | None = None
        self._n_items: int = 0

    def build(self, embeddings: npt.NDArray[np.float32]) -> None:
        try:
            import hnswlib
        except ImportError as exc:
            raise ImportError("hnswlib is required for large vaults: pip install hnswlib") from exc

        dim = embeddings.shape[1]
        self._n_items = len(embeddings)

        index = hnswlib.Index(space="cosine", dim=dim)
        index.init_index(
            max_elements=self._n_items,
            ef_construction=self._EF_CONSTRUCTION,
            M=self._M,
        )
        index.set_ef(self._EF_SEARCH)
        index.add_items(embeddings, ids=list(range(self._n_items)))

        self._index = index
        logger.debug(
            f"HNSWStrategy: built index for {self._n_items} embeddings (approximate search)"
        )

    def query(
        self,
        embeddings: npt.NDArray[np.float32],
        top_k: int,
        threshold: float,
    ) -> list[list[tuple[int, float]]]:
        assert self._index is not None, "call build() before query()"

        k = min(top_k + 1, self._n_items)  # +1 to discard the self-match
        # hnswlib cosine space: distance = 1 - similarity
        indices_batch, distances_batch = self._index.knn_query(embeddings, k=k)

        results: list[list[tuple[int, float]]] = []
        for i, (idxs, dists) in enumerate(zip(indices_batch, distances_batch, strict=False)):
            neighbours: list[tuple[int, float]] = []
            for idx, dist in zip(idxs, dists, strict=False):
                if int(idx) == i:
                    continue
                score = float(1.0 - dist)
                if score < threshold:
                    continue
                neighbours.append((int(idx), score))
                if len(neighbours) >= top_k:
                    break
            results.append(neighbours)

        return results


def select_strategy(n_notes: int) -> SimilarityStrategy:
    """
    Factory: choose the right backend based on vault size.

    This is the only place that branches on vault size; the pipeline
    calls .build()/.query() and never inspects the concrete type.
    """
    if n_notes > LARGE_VAULT_THRESHOLD:
        logger.info(
            f"Vault has {n_notes} notes (>{LARGE_VAULT_THRESHOLD}): "
            "using approximate HNSW search"
        )
        return HNSWStrategy()
    logger.info(
        f"Vault has {n_notes} notes (<={LARGE_VAULT_THRESHOLD}): "
        "using exact numpy search"
    )
    return NumpyStrategy()
