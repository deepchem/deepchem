from haiku import Linear
import deepchem as dc
import deepchem.models.optimizers as optimizers
import unittest
import pytest

try:
  import tensorflow as tf
  has_tensorflow = True
except:
  has_tensorflow = False

try:
  import tensorflow_addons as tfa
  has_tensorflow_addons = True
except:
  has_tensorflow_addons = False

try:
  import torch
  has_pytorch = True
except:
  has_pytorch = False

try:
  import jax
  import optax
  has_jax = True
except:
  has_jax = False


class TestOptimizers(unittest.TestCase):
  """Test optimizers and related classes."""

  @pytest.mark.tensorflow
  def test_adam_tf(self):
    """Test creating an Adam optimizer."""
    opt = optimizers.Adam(learning_rate=0.01)
    global_step = tf.Variable(0)
    tfopt = opt._create_tf_optimizer(global_step)
    assert isinstance(tfopt, tf.keras.optimizers.Adam)

  @pytest.mark.torch
  def test_adam_pytorch(self):
    """Test creating an Adam optimizer."""
    opt = optimizers.Adam(learning_rate=0.01)
    params = [torch.nn.Parameter(torch.Tensor([1.0]))]
    torchopt = opt._create_pytorch_optimizer(params)
    assert isinstance(torchopt, torch.optim.Adam)

  @pytest.mark.jax
  def test_adam_jax(self):
    """Test creating an Adam optimizer."""
    import optax
    opt = optimizers.Adam(learning_rate=0.01)
    jaxopt = opt._create_jax_optimizer()
    assert isinstance(jaxopt, optax.GradientTransformation)

  @pytest.mark.tensorflow
  def test_adamw_tf(self):
    """Test creating an AdamW optimizer."""
    opt = optimizers.AdamW(learning_rate=0.01)
    global_step = tf.Variable(0)
    tfopt = opt._create_tf_optimizer(global_step)
    assert isinstance(tfopt, tfa.optimizers.AdamW)

  @pytest.mark.torch
  def test_adamw_pytorch(self):
    """Test creating an AdamW optimizer."""
    opt = optimizers.AdamW(learning_rate=0.01)
    params = [torch.nn.Parameter(torch.Tensor([1.0]))]
    torchopt = opt._create_pytorch_optimizer(params)
    assert isinstance(torchopt, torch.optim.AdamW)

  @pytest.mark.jax
  def test_adamw_jax(self):
    """Test creating an AdamW optimizer."""
    import optax
    opt = optimizers.AdamW(learning_rate=0.01)
    jaxopt = opt._create_jax_optimizer()
    assert isinstance(jaxopt, optax.GradientTransformation)

  @pytest.mark.tensorflow
  def test_sparseadam_tf(self):
    """Test creating a SparseAdam optimizer."""
    opt = optimizers.SparseAdam(learning_rate=0.01)
    global_step = tf.Variable(0)
    tfopt = opt._create_tf_optimizer(global_step)
    assert isinstance(tfopt, tfa.optimizers.LazyAdam)

  @pytest.mark.torch
  def test_sparseadam_pytorch(self):
    """Test creating a SparseAdam optimizer."""
    opt = optimizers.SparseAdam(learning_rate=0.01)
    params = [torch.nn.Parameter(torch.Tensor([1.0]))]
    torchopt = opt._create_pytorch_optimizer(params)
    assert isinstance(torchopt, torch.optim.SparseAdam)

  @pytest.mark.tensorflow
  def test_adagrad_tf(self):
    """Test creating an AdaGrad optimizer."""
    opt = optimizers.AdaGrad(learning_rate=0.01)
    global_step = tf.Variable(0)
    tfopt = opt._create_tf_optimizer(global_step)
    assert isinstance(tfopt, tf.keras.optimizers.Adagrad)

  @pytest.mark.torch
  def test_adagrad_pytorch(self):
    """Test creating an AdaGrad optimizer."""
    opt = optimizers.AdaGrad(learning_rate=0.01)
    params = [torch.nn.Parameter(torch.Tensor([1.0]))]
    torchopt = opt._create_pytorch_optimizer(params)
    assert isinstance(torchopt, torch.optim.Adagrad)

  @pytest.mark.jax
  def test_adagrad_jax(self):
    """Test creating an AdaGrad optimizer."""
    import optax
    opt = optimizers.AdaGrad(learning_rate=0.01)
    jaxopt = opt._create_jax_optimizer()
    assert isinstance(jaxopt, optax.GradientTransformation)

  @pytest.mark.tensorflow
  def test_rmsprop_tf(self):
    """Test creating an RMSProp Optimizer."""
    opt = optimizers.RMSProp(learning_rate=0.01)
    global_step = tf.Variable(0)
    tfopt = opt._create_tf_optimizer(global_step)
    assert isinstance(tfopt, tf.keras.optimizers.RMSprop)

  @pytest.mark.torch
  def test_rmsprop_pytorch(self):
    """Test creating an RMSProp Optimizer."""
    opt = optimizers.RMSProp(learning_rate=0.01)
    params = [torch.nn.Parameter(torch.Tensor([1.0]))]
    torchopt = opt._create_pytorch_optimizer(params)
    assert isinstance(torchopt, torch.optim.RMSprop)

  @pytest.mark.jax
  def test_rmsprop_jax(self):
    """Test creating an RMSProp Optimizer."""
    import optax
    opt = optimizers.RMSProp(learning_rate=0.01)
    jaxopt = opt._create_jax_optimizer()
    assert isinstance(jaxopt, optax.GradientTransformation)

  @pytest.mark.tensorflow
  def test_gradient_descent_tf(self):
    """Test creating a Gradient Descent optimizer."""
    opt = optimizers.GradientDescent(learning_rate=0.01)
    global_step = tf.Variable(0)
    tfopt = opt._create_tf_optimizer(global_step)
    assert isinstance(tfopt, tf.keras.optimizers.SGD)

  @pytest.mark.torch
  def test_gradient_descent_pytorch(self):
    """Test creating a Gradient Descent optimizer."""
    opt = optimizers.GradientDescent(learning_rate=0.01)
    params = [torch.nn.Parameter(torch.Tensor([1.0]))]
    torchopt = opt._create_pytorch_optimizer(params)
    assert isinstance(torchopt, torch.optim.SGD)

  @pytest.mark.jax
  def test_gradient_descent_jax(self):
    """Test creating an Gradient Descent Optimizer."""
    import optax
    opt = optimizers.GradientDescent(learning_rate=0.01)
    jaxopt = opt._create_jax_optimizer()
    assert isinstance(jaxopt, optax.GradientTransformation)

  @pytest.mark.tensorflow
  def test_exponential_decay_tf(self):
    """Test creating an optimizer with an exponentially decaying learning rate."""
    rate = optimizers.ExponentialDecay(initial_rate=0.001,
                                       decay_rate=0.99,
                                       decay_steps=10000)
    opt = optimizers.Adam(learning_rate=rate)
    global_step = tf.Variable(0)
    tfopt = opt._create_tf_optimizer(global_step)

  @pytest.mark.torch
  def test_exponential_decay_pytorch(self):
    """Test creating an optimizer with an exponentially decaying learning rate."""
    rate = optimizers.ExponentialDecay(initial_rate=0.001,
                                       decay_rate=0.99,
                                       decay_steps=10000)
    opt = optimizers.Adam(learning_rate=rate)
    params = [torch.nn.Parameter(torch.Tensor([1.0]))]
    torchopt = opt._create_pytorch_optimizer(params)
    schedule = rate._create_pytorch_schedule(torchopt)

  @pytest.mark.jax
  def test_exponential_decay_jax(self):
    """Test creating an optimizer with an exponentially decaying learning rate."""
    import optax
    rate = optimizers.ExponentialDecay(initial_rate=0.001,
                                       decay_rate=0.99,
                                       decay_steps=10000)
    opt = optimizers.Adam(learning_rate=rate)
    jaxopt = opt._create_jax_optimizer()
    assert isinstance(jaxopt, optax.GradientTransformation)

  @pytest.mark.tensorflow
  def test_polynomial_decay_tf(self):
    """Test creating an optimizer with a polynomially decaying learning rate."""
    rate = optimizers.PolynomialDecay(initial_rate=0.001,
                                      final_rate=0.0001,
                                      decay_steps=10000)
    opt = optimizers.Adam(learning_rate=rate)
    global_step = tf.Variable(0)
    tfopt = opt._create_tf_optimizer(global_step)

  @pytest.mark.torch
  def test_polynomial_decay_pytorch(self):
    """Test creating an optimizer with a polynomially decaying learning rate."""
    rate = optimizers.PolynomialDecay(initial_rate=0.001,
                                      final_rate=0.0001,
                                      decay_steps=10000)
    opt = optimizers.Adam(learning_rate=rate)
    params = [torch.nn.Parameter(torch.Tensor([1.0]))]
    torchopt = opt._create_pytorch_optimizer(params)
    schedule = rate._create_pytorch_schedule(torchopt)

  @pytest.mark.jax
  def test_polynomial_decay_jax(self):
    """Test creating an optimizer with a polynomially decaying learning rate."""
    import optax
    rate = optimizers.PolynomialDecay(initial_rate=0.001,
                                      final_rate=0.0001,
                                      decay_steps=10000)
    opt = optimizers.Adam(learning_rate=rate)
    jaxopt = opt._create_jax_optimizer()
    assert isinstance(jaxopt, optax.GradientTransformation)

  @pytest.mark.tensorflow
  def test_linearCosine_decay_tf(self):
    """test creating an optimizer with a linear cosine decay to the learning rate"""
    rate = optimizers.LinearCosineDecay(initial_rate=0.1, decay_steps=10000)
    opt = optimizers.Adam(learning_rate=rate)
    global_step = tf.Variable(0)
    tfopt = opt._create_tf_optimizer(global_step)

  @pytest.mark.torch
  def test_linearCosine_decay_pytorch(self):
    """test creating an optimizer with a linear cosine decay to the learning rate"""
    rate = optimizers.LinearCosineDecay(initial_rate=0.1, decay_steps=10000)
    opt = optimizers.Adam(learning_rate=rate)
    params = [torch.nn.Parameter(torch.Tensor([1.0]))]
    torchopt = opt._create_pytorch_optimizer(params)
    schedule = rate._create_pytorch_schedule(torchopt)

  @pytest.mark.jax
  def test_linearCosine_decay_jax(self):
    """test creating an optimizer with a linear cosine decay to the learning rate"""
    import optax
    rate = optimizers.LinearCosineDecay(initial_rate=0.1, decay_steps=10000)
    opt = optimizers.Adam(learning_rate=rate)
    jaxopt = opt._create_jax_optimizer()
    assert isinstance(jaxopt, optax.GradientTransformation)

  @pytest.mark.jax
  def test_PieceWise_decay_jax(self):
    """test creating an optimizer with a PeiceWise constant decay to the learning rate"""
    import optax
    rate = optimizers.PiecewiseConstantSchedule(initial_rate=0.1,
                                                boundaries_and_scales={
                                                    5000: 0.1,
                                                    10000: 0.1,
                                                    15000: 0.1
                                                })
    opt = optimizers.Adam(learning_rate=rate)
    jaxopt = opt._create_jax_optimizer()
    assert isinstance(jaxopt, optax.GradientTransformation)

  @pytest.mark.torch
  def test_KFAC(self):
    """test creating a KFAC optimizer"""
    import torch
    import numpy as np

    np.random.seed(123)
    """linear layers test"""

    n_samples = 10
    n_features = 100
    n_tasks = 1

    X = np.random.rand(n_samples, n_features)
    y = np.random.randint(2, size=(n_samples, n_tasks)).astype(np.float32)
    dataset = dc.data.NumpyDataset(X, y)
    metric = dc.metrics.Metric(dc.metrics.pearson_r2_score)
    pytorch_model = torch.nn.Sequential(torch.nn.Linear(n_features, 50),
                                        torch.nn.ReLU(), torch.nn.Linear(50, 1))
    model = dc.models.TorchModel(pytorch_model,
                                 dc.models.losses.L2Loss(),
                                 optimizers=optimizers.KFAC(model=pytorch_model,
                                                            learning_rate=0.1,
                                                            Tinv=50))
    model.fit(dataset, nb_epoch=100)

    scores = model.evaluate(dataset, [metric])
    assert scores[metric.name] > 0.9
    """Conv2d layers test"""
    n_samples = 10
    n_features = 3
    n_tasks = 1

    X = np.random.rand(n_samples, 10, 10, n_features)
    y = np.random.randint(2, size=(n_samples, n_tasks)).astype(np.float32)
    dataset = dc.data.NumpyDataset(X, y)

    metric = dc.metrics.Metric(dc.metrics.roc_auc_score)
    model = torch.nn.Sequential(
        torch.nn.Conv2d(n_features, 32, kernel_size=3, padding=1),
        torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
        torch.nn.Linear(64 * 10 * 10, 20), torch.nn.ReLU(),
        torch.nn.Linear(20, n_tasks))
    model = dc.models.TorchModel(pytorch_model,
                                 dc.models.losses.L2Loss(),
                                 optimizers=optimizers.KFAC(model=pytorch_model,
                                                            learning_rate=0.003,
                                                            Tinv=50))
    # Fit trained model
    model.fit(
        dataset,
        nb_epoch=100,
    )

    # Eval model on train
    scores = model.evaluate(dataset, [metric])
    assert scores[metric.name] > 0.9
