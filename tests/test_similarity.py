"""
Tests for inference/similarity.py — strategy selection and retrieval correctness.

We use synthetic embeddings (hand-crafted unit vectors) so tests are
deterministic and do not require a real model or a GPU.
"""

import numpy as np
import pytest

from rhizome.inference.similarity import (
    LARGE_VAULT_THRESHOLD,
    HNSWStrategy,
    NumpyStrategy,
    SimilarityStrategy,
    select_strategy,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unit_vec(direction: list[float]) -> np.ndarray:
    """Return a 1-D unit vector from an un-normalised direction."""
    v = np.array(direction, dtype=np.float32)
    return v / np.linalg.norm(v)


def _make_embeddings(*directions: list[float]) -> np.ndarray:
    """Stack multiple unit vectors into an (N, D) embedding matrix."""
    return np.stack([_unit_vec(d) for d in directions])


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------

def test_numpy_strategy_satisfies_protocol() -> None:
    assert isinstance(NumpyStrategy(), SimilarityStrategy)


def test_hnsw_strategy_satisfies_protocol() -> None:
    assert isinstance(HNSWStrategy(), SimilarityStrategy)


# ---------------------------------------------------------------------------
# select_strategy
# ---------------------------------------------------------------------------

def test_select_strategy_returns_numpy_for_small_vaults() -> None:
    strategy = select_strategy(LARGE_VAULT_THRESHOLD)
    assert isinstance(strategy, NumpyStrategy)


def test_select_strategy_returns_hnsw_for_large_vaults() -> None:
    strategy = select_strategy(LARGE_VAULT_THRESHOLD + 1)
    assert isinstance(strategy, HNSWStrategy)


# ---------------------------------------------------------------------------
# NumpyStrategy correctness
# ---------------------------------------------------------------------------

class TestNumpyStrategy:
    def _strategy(self, embeddings: np.ndarray) -> NumpyStrategy:
        s = NumpyStrategy()
        s.build(embeddings)
        return s

    def test_returns_most_similar_neighbour(self) -> None:
        embeddings = _make_embeddings(
            [1.0, 0.0, 0.0],   # note 0
            [0.99, 0.14, 0.0], # note 1 — very close to note 0
            [0.0, 0.0, 1.0],   # note 2 — orthogonal
        )
        s = self._strategy(embeddings)
        results = s.query(embeddings, top_k=1, threshold=0.0)
        assert results[0][0][0] == 1

    def test_self_similarity_excluded(self) -> None:
        embeddings = _make_embeddings([1.0, 0.0], [0.0, 1.0])
        s = self._strategy(embeddings)
        results = s.query(embeddings, top_k=2, threshold=0.0)
        for i, neighbours in enumerate(results):
            assert all(idx != i for idx, _ in neighbours)

    def test_threshold_filters_low_similarity(self) -> None:
        embeddings = _make_embeddings([1.0, 0.0], [0.0, 1.0])
        s = self._strategy(embeddings)
        results = s.query(embeddings, top_k=5, threshold=0.5)
        assert results[0] == []
        assert results[1] == []

    def test_top_k_limits_results(self) -> None:
        embeddings = _make_embeddings(
            [1.0, 0.1, 0.0],
            [0.9, 0.2, 0.0],
            [0.8, 0.3, 0.0],
            [0.7, 0.4, 0.0],
        )
        s = self._strategy(embeddings)
        results = s.query(embeddings, top_k=2, threshold=0.0)
        assert all(len(r) <= 2 for r in results)

    def test_scores_are_descending(self) -> None:
        embeddings = _make_embeddings(
            [1.0, 0.0, 0.0],
            [0.9, 0.4, 0.0],
            [0.5, 0.5, 0.7],
        )
        s = self._strategy(embeddings)
        results = s.query(embeddings, top_k=5, threshold=0.0)
        for neighbours in results:
            scores = [score for _, score in neighbours]
            assert scores == sorted(scores, reverse=True)

    def test_empty_results_when_no_notes_above_threshold(self) -> None:
        embeddings = _make_embeddings([1.0, 0.0], [0.0, 1.0])
        s = self._strategy(embeddings)
        results = s.query(embeddings, top_k=5, threshold=0.99)
        assert all(r == [] for r in results)


# ---------------------------------------------------------------------------
# HNSWStrategy correctness
# ---------------------------------------------------------------------------

class TestHNSWStrategy:
    """
    HNSW is approximate, so we only test structural properties, not exact recall.
    """

    @pytest.fixture()
    def strategy_10(self) -> HNSWStrategy:
        rng = np.random.default_rng(42)
        raw = rng.standard_normal((10, 16)).astype(np.float32)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        embeddings = raw / norms
        s = HNSWStrategy()
        s.build(embeddings)
        s._test_embeddings = embeddings  # type: ignore[attr-defined]
        return s

    def test_hnsw_no_self_matches(self, strategy_10: HNSWStrategy) -> None:
        emb = strategy_10._test_embeddings  # type: ignore[attr-defined]
        results = strategy_10.query(emb, top_k=3, threshold=0.0)
        for i, neighbours in enumerate(results):
            assert all(idx != i for idx, _ in neighbours)

    def test_hnsw_top_k_respected(self, strategy_10: HNSWStrategy) -> None:
        emb = strategy_10._test_embeddings  # type: ignore[attr-defined]
        results = strategy_10.query(emb, top_k=2, threshold=0.0)
        assert all(len(r) <= 2 for r in results)

    def test_hnsw_scores_in_range(self, strategy_10: HNSWStrategy) -> None:
        emb = strategy_10._test_embeddings  # type: ignore[attr-defined]
        results = strategy_10.query(emb, top_k=5, threshold=0.0)
        for neighbours in results:
            for _, score in neighbours:
                assert -1.0 <= score <= 1.0
