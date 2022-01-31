import deepchem as dc
import jax
import haiku as hk
import deepchem.models.jax_models.layers
import jax.numpy as jnp

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
