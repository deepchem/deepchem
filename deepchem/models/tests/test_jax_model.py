import unittest

from deepchem.models.tests.test_graph_models import get_dataset
import numpy as np

try:
  import jax
  import jax.numpy as jnp
  import haiku as hk
  import optax
  from deepchem.models import JaxModel
  has_haiku_and_optax = True
except:
  has_haiku_and_optax = False


@unittest.skipIf(not has_haiku_and_optax,
                 'Jax, Haiku, or Optax are not installed')
def test_jax_model_for_regression():
  tasks, dataset, transformers, metric = get_dataset(
      'regression', featurizer='ECFP')

  # sample network
  def f(x):
    net = hk.nets.MLP([512, 256, 128, 1])
    return net(x)

  def rms_loss(pred, tar, w):
    return jnp.mean(optax.l2_loss(pred, tar))

  # Model Initilisation
  model = hk.without_apply_rng(hk.transform(f))
  rng = jax.random.PRNGKey(500)
  inputs, _, _, _ = next(iter(dataset.iterbatches(batch_size=256)))
  modified_inputs = jnp.array(
      [x.astype(np.float32) if x.dtype == np.float64 else x for x in inputs])
  params = model.init(rng, modified_inputs)

  # Loss Function
  criterion = rms_loss

  # JaxModel Working
  j_m = JaxModel(
      model,
      params,
      criterion,
      batch_size=256,
      learning_rate=0.001,
      log_frequency=2)
  results = j_m.fit(dataset, deterministic=True)
  assert results < 0.5


@unittest.skipIf(not has_haiku_and_optax,
                 'Jax, Haiku, or Optax are not installed')
def test_jax_model_for_classification():
  tasks, dataset, transformers, metric = get_dataset(
      'classification', featurizer='ECFP')

  # sample network
  class Encoder(hk.Module):

    def __init__(self, output_size: int = 1):
      super().__init__()
      self._network = hk.nets.MLP([512, 256, 128, output_size])

    def __call__(self, x: jnp.ndarray):
      x = self._network(x)
      return x, jax.nn.softmax(x)

  def f(x):
    net = Encoder(2)
    return net(x)

  def bce_loss(pred, tar, w):
    tar = jnp.array(
        [x.astype(np.float32) if x.dtype != np.float32 else x for x in tar])
    return jnp.mean(optax.softmax_cross_entropy(pred[0], tar))

  # Model Initilisation
  model = hk.without_apply_rng(hk.transform(f))
  rng = jax.random.PRNGKey(500)
  inputs, _, _, _ = next(iter(dataset.iterbatches(batch_size=256)))
  modified_inputs = jnp.array(
      [x.astype(np.float32) if x.dtype == np.float64 else x for x in inputs])
  params = model.init(rng, modified_inputs)

  # Loss Function
  criterion = bce_loss

  # JaxModel Working
  j_m = JaxModel(
      model,
      params,
      criterion,
      output_types=['loss', 'prediction'],
      batch_size=256,
      learning_rate=0.001,
      log_frequency=2)
  results = j_m.fit(dataset, nb_epochs=50, deterministic=True)
  assert results < 1.0
