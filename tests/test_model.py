"""
Tests for inference/model.py — mean_pool function correctness.

These tests use synthetic numpy arrays so no ONNX session or network
access is required.
"""

import numpy as np

from rhizome.inference.model import mean_pool


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
