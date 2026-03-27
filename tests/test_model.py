"""
Tests for inference/model.py — mean_pool and chunk_token_ids correctness.

These tests use synthetic numpy arrays so no ONNX session or network
access is required.
"""

import numpy as np
import pytest

from rhizome.inference.model import chunk_token_ids, mean_pool


class TestMeanPool:
    """
    mean_pool(token_embeddings, attention_mask) -> np.ndarray

    Input shapes:
      token_embeddings: (seq_len, hidden_dim)
      attention_mask:   (seq_len,)
    Output shape: (hidden_dim,)
    """

    def test_known_input_output(self) -> None:
        """
        Manual calculation:
          seq_len=3, hidden_dim=4
          tokens: [[1,0,0,0], [0,1,0,0], [0,0,1,0]]
          mask:   [1, 1, 0]   — third token is padding

          pooled = (token_0 * 1 + token_1 * 1 + token_2 * 0) / 2
                 = ([1,0,0,0] + [0,1,0,0]) / 2
                 = [0.5, 0.5, 0.0, 0.0]
        """
        token_embeddings = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
            dtype=np.float32,
        )
        attention_mask = np.array([1, 1, 0], dtype=np.int64)

        result = mean_pool(token_embeddings, attention_mask)

        expected = np.array([0.5, 0.5, 0.0, 0.0], dtype=np.float32)
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_all_tokens_unmasked(self) -> None:
        """When all mask values are 1, mean_pool == plain mean."""
        token_embeddings = np.array(
            [[2, 0], [0, 2], [1, 1]],
            dtype=np.float32,
        )
        attention_mask = np.ones(3, dtype=np.int64)

        result = mean_pool(token_embeddings, attention_mask)
        expected = np.mean(token_embeddings, axis=0)

        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_single_real_token(self) -> None:
        """With only one unmasked token, pooled == that token."""
        token_embeddings = np.array(
            [[3, 1, 4, 1], [0, 0, 0, 0], [0, 0, 0, 0]],
            dtype=np.float32,
        )
        attention_mask = np.array([1, 0, 0], dtype=np.int64)

        result = mean_pool(token_embeddings, attention_mask)
        expected = np.array([3, 1, 4, 1], dtype=np.float32)

        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_output_shape(self) -> None:
        """Output shape is (hidden_dim,) — no batch dimension."""
        seq_len = 4
        hidden_dim = 8

        rng = np.random.default_rng(0)
        token_embeddings = rng.standard_normal((seq_len, hidden_dim)).astype(np.float32)
        attention_mask = np.ones(seq_len, dtype=np.int64)

        result = mean_pool(token_embeddings, attention_mask)

        assert result.shape == (hidden_dim,)

    def test_output_dtype_is_float32(self) -> None:
        token_embeddings = np.ones((3, 4), dtype=np.float32)
        attention_mask = np.ones(3, dtype=np.int64)

        result = mean_pool(token_embeddings, attention_mask)

        assert result.dtype == np.float32

    def test_zero_mask_does_not_divide_by_zero(self) -> None:
        """All-zero mask must not raise — clamp ensures safe division."""
        token_embeddings = np.ones((3, 4), dtype=np.float32)
        attention_mask = np.zeros(3, dtype=np.int64)
        result = mean_pool(token_embeddings, attention_mask)
        assert result.shape == (4,)
        assert np.all(np.isfinite(result))

    def test_padding_does_not_affect_result(self) -> None:
        """
        Adding a fully-masked (padding) token with arbitrary values must not
        change the pooled output.
        """
        token_embeddings_no_pad = np.array(
            [[1, 0], [0, 1]],
            dtype=np.float32,
        )
        mask_no_pad = np.array([1, 1], dtype=np.int64)

        # Same tokens plus a padding token with large values
        token_embeddings_with_pad = np.array(
            [[1, 0], [0, 1], [99, 99]],
            dtype=np.float32,
        )
        mask_with_pad = np.array([1, 1, 0], dtype=np.int64)

        result_no_pad = mean_pool(token_embeddings_no_pad, mask_no_pad)
        result_with_pad = mean_pool(token_embeddings_with_pad, mask_with_pad)

        np.testing.assert_allclose(result_no_pad, result_with_pad, atol=1e-6)


class TestChunkTokenIds:
    """
    chunk_token_ids(ids, max_tokens, overlap) -> list[list[int]]

    All assertions use explicit max_tokens / overlap values so the tests
    are independent of the module-level defaults (_MAX_TOKENS, _CHUNK_OVERLAP).
    """

    def test_short_sequence_returned_as_single_chunk(self) -> None:
        ids = list(range(10))
        result = chunk_token_ids(ids, max_tokens=512, overlap=64)
        assert result == [ids]

    def test_exactly_max_tokens_is_single_chunk(self) -> None:
        ids = list(range(512))
        result = chunk_token_ids(ids, max_tokens=512, overlap=64)
        assert len(result) == 1
        assert result[0] == ids

    def test_long_sequence_produces_multiple_chunks(self) -> None:
        # 1000 tokens, max=512, overlap=64 → stride=448
        # chunk 0: [0, 512)
        # chunk 1: [448, 960)
        # chunk 2: [896, 1000)
        ids = list(range(1000))
        result = chunk_token_ids(ids, max_tokens=512, overlap=64)
        assert len(result) == 3

    def test_all_tokens_covered(self) -> None:
        """Every token ID must appear in at least one chunk."""
        ids = list(range(700))
        result = chunk_token_ids(ids, max_tokens=512, overlap=64)
        covered = set()
        for chunk in result:
            covered.update(chunk)
        assert covered == set(ids)

    def test_overlap_tokens_appear_in_consecutive_chunks(self) -> None:
        ids = list(range(600))
        overlap = 64
        result = chunk_token_ids(ids, max_tokens=512, overlap=overlap)
        assert len(result) >= 2
        # The tail of chunk 0 and the head of chunk 1 must share 'overlap' tokens.
        tail = result[0][-overlap:]
        head = result[1][:overlap]
        assert tail == head

    def test_no_chunk_exceeds_max_tokens(self) -> None:
        ids = list(range(2000))
        result = chunk_token_ids(ids, max_tokens=512, overlap=64)
        assert all(len(chunk) <= 512 for chunk in result)

    def test_final_chunk_ends_at_last_token(self) -> None:
        ids = list(range(550))
        result = chunk_token_ids(ids, max_tokens=512, overlap=64)
        assert result[-1][-1] == ids[-1]

    def test_empty_sequence_returns_single_empty_chunk(self) -> None:
        result = chunk_token_ids([], max_tokens=512, overlap=64)
        assert result == [[]]

    @pytest.mark.parametrize("length", [513, 600, 1024, 2000])
    def test_various_lengths_always_cover_all_tokens(self, length: int) -> None:
        ids = list(range(length))
        result = chunk_token_ids(ids, max_tokens=512, overlap=64)
        covered = {tok for chunk in result for tok in chunk}
        assert covered == set(ids)
