"""
checking jax imports for new CI build
"""
import deepchem as dc
import pytest

try:
    import jax.numpy as jnp
    from jax import random
    import numpy as np
    has_jax = True
except:
    has_jax = False


@pytest.mark.jax
def test_jax_import():
    """Used to check if Jax is imported correctly. Will be useful in Mac and Windows build"""
    key = random.PRNGKey(0)
    x = random.normal(key, (10, 10), dtype=jnp.float32)
    y = random.normal(key, (10, 10), dtype=jnp.float32)
    assert jnp.all(x == y)

    n_data_points = 10
    n_features = 2
    np.random.seed(1234)
    X = np.random.rand(n_data_points, n_features)
    y = (X[:, 0] > X[:, 1]).astype(np.float32)
    dataset = dc.data.NumpyDataset(X, y)
    assert dataset.X.shape == (10, 2)
