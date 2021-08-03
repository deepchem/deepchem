import pytest
import deepchem as dc
import numpy as np

try:
  import jax
  import jax.numpy as jnp
  from jax import random
  import haiku as hk
except:
  has_haiku_and_optax = False


@pytest.mark.jax
def test_linear():
  import deepchem as dc
  import haiku as hk
  import deepchem.models.jax_models.layers

  def forward(x):
    layer = dc.models.jax_models.layers.Linear(2)
    return layer(x)

  forward = hk.transform(forward)
  rng = jax.random.PRNGKey(42)
  x = jnp.ones([8, 28 * 28])
  params = forward.init(rng, x)
  output = forward.apply(params, rng, x)
  assert output.shape == (8, 2)
