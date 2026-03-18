import pytest

try:
    import jax
    import jax.numpy as jnp
    from jax import random  # noqa: F401
    import haiku as hk
except:
    has_haiku_and_optax = False


@pytest.mark.jax
def test_linear():
    from deepchem.models.jax_models import layers as jax_models_layers

    def forward(x):
        layer = jax_models_layers.Linear(2)
        return layer(x)

    forward = hk.transform(forward)
    rng = jax.random.PRNGKey(42)
    x = jnp.ones([8, 28 * 28])
    params = forward.init(rng, x)
    output = forward.apply(params, rng, x)
    assert output.shape == (8, 2)
