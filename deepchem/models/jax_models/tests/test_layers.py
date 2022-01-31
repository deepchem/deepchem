import pytest
import deepchem as dc

try:
  import jax
  import jax.numpy as jnp
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


@pytest.mark.jax
def test_selfattention_single():
  seq_len = 5
  embed_size = 2

  def model(x):
    q = k = v = jnp.zeros((seq_len, embed_size))
    causal_mask = jnp.tril(jnp.ones((seq_len, seq_len)))
    causal_mask = causal_mask[None, :, :]

    sa = dc.models.jax_models.layers.SelfAttention(
        key_size=7, num_heads=11, value_size=13, model_size=2,
        w_init_scale=1.0)(q, k, v, causal_mask)
    print(sa.shape)
    return sa

  x = jnp.ones([1, 5])
  print(x)
  f = hk.transform(model)
  rng = jax.random.PRNGKey(42)
  params = f.init(rng, x)
  out = f.apply(params, rng, x)

  assert out.shape == (seq_len, embed_size)
