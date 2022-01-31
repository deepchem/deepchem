import logging
try:
  import jax.numpy as jnp
  import haiku as hk
  import numpy as np

  from typing import Optional

except:
  raise ImportError('These classes require Jax and Haiku to be installed.')

logger = logging.getLogger(__name__)


class Linear(hk.Module):
  """Protein folding specific Linear Module.

  This differs from the standard Haiku Linear in a few ways:
    * It supports inputs of arbitrary rank
    * Initializers are specified by strings

  This code is adapted from DeepMind's AlphaFold code release
  (https://github.com/deepmind/alphafold).

  Examples
  --------
  >>> import deepchem as dc
  >>> import haiku as hk
  >>> import jax
  >>> import deepchem.models.jax_models.layers
  >>> def forward_model(x):
  ...   layer = dc.models.jax_models.layers.Linear(2)
  ...   return layer(x)
  >>> f = hk.transform(forward_model)
  >>> rng = jax.random.PRNGKey(42)
  >>> x = jnp.ones([8, 28 * 28])
  >>> params = f.init(rng, x)
  >>> output = f.apply(params, rng, x)
  """

  def __init__(self,
               num_output: int,
               initializer: str = 'linear',
               use_bias: bool = True,
               bias_init: float = 0.,
               name: str = 'linear'):
    """Constructs Linear Module.

    Parameters
    ----------
    num_output: int
      number of output channels.
    initializer: str (default 'linear')
      What initializer to use, should be one of {'linear', 'relu', 'zeros'}
    use_bias: bool (default True)
      Whether to include trainable bias
    bias_init: float (default 0)
      Value used to initialize bias.
    name: str (default 'linear')
      name of module, used for name scopes.
    """

    super().__init__(name=name)
    self.num_output = num_output
    self.initializer = initializer
    self.use_bias = use_bias
    self.bias_init = bias_init

  def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
    """Connects Module.

    Parameters
    ----------
    inputs: jnp.ndarray
      Tensor of shape [..., num_channel]

    Returns
    -------
    output of shape [..., num_output]
    """
    n_channels = int(inputs.shape[-1])

    weight_shape = [n_channels, self.num_output]
    if self.initializer == 'linear':
      weight_init = hk.initializers.VarianceScaling(mode='fan_in', scale=1.)
    elif self.initializer == 'relu':
      weight_init = hk.initializers.VarianceScaling(mode='fan_in', scale=2.)
    elif self.initializer == 'zeros':
      weight_init = hk.initializers.Constant(0.0)

    weights = hk.get_parameter('weights', weight_shape, inputs.dtype,
                               weight_init)

    # this is equivalent to einsum('...c,cd->...d', inputs, weights)
    # but turns out to be slightly faster
    inputs = jnp.swapaxes(inputs, -1, -2)
    output = jnp.einsum('...cb,cd->...db', inputs, weights)
    output = jnp.swapaxes(output, -1, -2)

    if self.use_bias:
      bias = hk.get_parameter('bias', [self.num_output], inputs.dtype,
                              hk.initializers.Constant(self.bias_init))
      output += bias

    return output


def layer_norm(arr: jnp.ndarray, name: Optional[str] = None) -> jnp.ndarray:
  """
  Implementation of LayerNorm
  """
  return hk.LayerNorm(
      axis=-1, create_scale=True, create_offset=True, name=name)(arr)


class SelfAttention(hk.MultiHeadAttention):
  '''
  Implementation of Self Attention as per Attention is All you need Paper : https://arxiv.org/abs/1706.03762
  '''

  def __call__(self,
               q: jnp.ndarray,
               k: Optional[jnp.ndarray] = None,
               v: Optional[jnp.ndarray] = None,
               mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    '''
    Parameters
    ----------
    inputs:
      q : jnp.ndarray
        Standard query parameter for attention
      k : jnp.ndarray [Optional Parameter]
        Standard key parameter for attention
      v : jnp.ndarray [Optional Parameter]
        Standard value parameter for attention
      mask : jnp.ndarray [Optional Parameter]
        Attention Mask

    Returns
    -------
    output of shape [..., num_output]
    '''

    if k is None:
      k = q

    if v is None:
      v = q

    seq_length = q.shape[0]
    causal_mask = np.tril(np.ones((seq_length, seq_length)))

    if mask is None:
      mask = causal_mask
    else:
      mask = causal_mask * mask

    return super().__call__(q, k, v, mask)
