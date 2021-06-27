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

  @unittest.skipIf(not has_tensorflow_addons,
                   'TensorFlow Addons is not installed')
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

  @unittest.skipIf(not has_tensorflow_addons,
                   'TensorFlow Addons is not installed')
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

  @pytest.mark.tensorflow
  def test_exponential_decay_tf(self):
    """Test creating an optimizer with an exponentially decaying learning rate."""
    rate = optimizers.ExponentialDecay(
        initial_rate=0.001, decay_rate=0.99, decay_steps=10000)
    opt = optimizers.Adam(learning_rate=rate)
    global_step = tf.Variable(0)
    tfopt = opt._create_tf_optimizer(global_step)

  @pytest.mark.torch
  def test_exponential_decay_pytorch(self):
    """Test creating an optimizer with an exponentially decaying learning rate."""
    rate = optimizers.ExponentialDecay(
        initial_rate=0.001, decay_rate=0.99, decay_steps=10000)
    opt = optimizers.Adam(learning_rate=rate)
    params = [torch.nn.Parameter(torch.Tensor([1.0]))]
    torchopt = opt._create_pytorch_optimizer(params)
    schedule = rate._create_pytorch_schedule(torchopt)

  @pytest.mark.tensorflow
  def test_polynomial_decay_tf(self):
    """Test creating an optimizer with a polynomially decaying learning rate."""
    rate = optimizers.PolynomialDecay(
        initial_rate=0.001, final_rate=0.0001, decay_steps=10000)
    opt = optimizers.Adam(learning_rate=rate)
    global_step = tf.Variable(0)
    tfopt = opt._create_tf_optimizer(global_step)

  @pytest.mark.torch
  def test_polynomial_decay_pytorch(self):
    """Test creating an optimizer with a polynomially decaying learning rate."""
    rate = optimizers.PolynomialDecay(
        initial_rate=0.001, final_rate=0.0001, decay_steps=10000)
    opt = optimizers.Adam(learning_rate=rate)
    params = [torch.nn.Parameter(torch.Tensor([1.0]))]
    torchopt = opt._create_pytorch_optimizer(params)
    schedule = rate._create_pytorch_schedule(torchopt)

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
