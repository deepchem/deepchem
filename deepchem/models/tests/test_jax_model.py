import pytest
from deepchem.models.tests.test_graph_models import get_dataset
import deepchem as dc
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


@pytest.mark.jax
def test_jax_model_for_regression():
  tasks, dataset, transformers, metric = get_dataset(
      'regression', featurizer='ECFP')

  # sample network
  def f(x):
    net = hk.nets.MLP([512, 256, 128, 2])
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
  n_tasks = len(tasks)
  j_m = JaxModel(
      model,
      params,
      criterion,
      batch_size=256,
      learning_rate=0.001,
      log_frequency=2)
  results = j_m.fit(dataset, nb_epochs=25, deterministic=True)
  scores = j_m.evaluate(dataset, [metric])
  assert scores[metric.name] < 0.5


@pytest.mark.jax
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
  _ = j_m.fit(dataset, nb_epochs=25, deterministic=True)
  scores = j_m.evaluate(dataset, [metric])
  assert scores[metric.name] > 0.8


@pytest.mark.jax
def test_overfit_subclass_model():
  """Test fitting a JaxModel defined by subclassing Module."""
  n_data_points = 10
  n_features = 2
  np.random.seed(1234)
  X = np.random.rand(n_data_points, n_features)
  y = (X[:, 0] > X[:, 1]).astype(np.float32)
  dataset = dc.data.NumpyDataset(X, np.expand_dims(y, axis=1))

  class Encoder(hk.Module):

    def __init__(self, output_size: int = 1):
      super().__init__()
      self._network = hk.nets.MLP([512, 256, 128, output_size])

    def __call__(self, x: jnp.ndarray):
      x = self._network(x)
      return x, jax.nn.sigmoid(x)

  def f(x):
    net = Encoder(1)
    return net(x)

  # Model Initilisation
  model = hk.without_apply_rng(hk.transform(f))
  rng = jax.random.PRNGKey(500)
  inputs, _, _, _ = next(iter(dataset.iterbatches(batch_size=100)))
  #inputs = np.random.rand(100, n_features)
  #print(inputs.shape)

  modified_inputs = jnp.array(
      [x.astype(np.float32) if x.dtype == np.float64 else x for x in inputs])
  params = model.init(rng, modified_inputs)

  # Loss Function
  criterion = lambda pred, tar, w: jnp.mean(optax.sigmoid_binary_cross_entropy(pred[0], tar))

  # JaxModel Working
  j_m = JaxModel(
      model,
      params,
      criterion,
      output_types=['loss', 'prediction'],
      batch_size=100,
      learning_rate=0.001,
      log_frequency=2)
  j_m.fit(dataset, nb_epochs=1000)
  prediction = np.squeeze(j_m.predict_on_batch(X))
  assert np.array_equal(y, np.round(prediction))
  metric = dc.metrics.Metric(dc.metrics.roc_auc_score)
  scores = j_m.evaluate(dataset, [metric])
  assert scores[metric.name] > 0.9


def test_fit_use_all_losses():
  """Test fitting a TorchModel defined by subclassing Module."""
  n_data_points = 10
  n_features = 2
  np.random.seed(1234)
  X = np.random.rand(n_data_points, n_features)
  y = (X[:, 0] > X[:, 1]).astype(np.float32)
  dataset = dc.data.NumpyDataset(X, np.expand_dims(y, axis=1))

  class Encoder(hk.Module):

    def __init__(self, output_size: int = 1):
      super().__init__()
      self._network = hk.nets.MLP([512, 256, 128, output_size])

    def __call__(self, x: jnp.ndarray):
      x = self._network(x)
      return x, jax.nn.sigmoid(x)

  def f(x):
    net = Encoder(1)
    return net(x)

  # Model Initilisation
  model = hk.without_apply_rng(hk.transform(f))
  rng = jax.random.PRNGKey(500)
  inputs, _, _, _ = next(iter(dataset.iterbatches(batch_size=100)))

  modified_inputs = jnp.array(
      [x.astype(np.float32) if x.dtype == np.float64 else x for x in inputs])
  params = model.init(rng, modified_inputs)

  # Loss Function
  criterion = lambda pred, tar, w: jnp.mean(optax.sigmoid_binary_cross_entropy(pred[0], tar))

  # JaxModel Working
  j_m = JaxModel(
      model,
      params,
      criterion,
      output_types=['loss', 'prediction'],
      learning_rate=0.005,
      log_frequency=10)

  losses = []
  j_m.fit(dataset, nb_epochs=1000, all_losses=losses)
  # Each epoch is a single step for this model
  assert len(losses) == 100
  assert np.count_nonzero(np.array(losses)) == 100
