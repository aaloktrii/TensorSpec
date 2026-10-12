import numpy as np
import pytest
from tensorspec.core.ml.ssl.defl_pool import aggregate_defl_embeddings


def test_mean_over_defl():
    emb = np.array([[1.0, 0.0], [3.0, 2.0], [5.0, 4.0]], dtype=np.float64)
    out = aggregate_defl_embeddings(emb, mode="mean")
    np.testing.assert_allclose(out, [3.0, 2.0])


def test_mid_default_index():
    emb = np.arange(17 * 4, dtype=np.float64).reshape(17, 4)
    out = aggregate_defl_embeddings(emb, mode="mid")  # mid_index = 8
    np.testing.assert_array_equal(out, emb[8])


def test_mid_explicit_index():
    emb = np.eye(3, 2)
    out = aggregate_defl_embeddings(emb, mode="mid", mid_index=1)
    np.testing.assert_array_equal(out, emb[1])


def test_rejects_bad_ndim():
    with pytest.raises(ValueError):
        aggregate_defl_embeddings(np.zeros(3), mode="mean")


def test_rejects_bad_mid_index():
    with pytest.raises(ValueError):
        aggregate_defl_embeddings(np.zeros((3, 2)), mode="mid", mid_index=9)
