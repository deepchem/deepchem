import deepchem.models.optimizers as optimizers
import unittest

try:
  import tensorflow as tf
  has_tensorflow = True
except:
  has_tensorflow = False

try:
  import torch
  has_pytorch = True
except:
  has_pytorch = False


class TestOptimizers(unittest.TestCase):
  """Test optimizers and related classes."""

  @unittest.skipIf(not has_tensorflow, 'TensorFlow is not installed')
  def test_adam_tf(self):
    """Test creating an Adam optimizer."""
    opt = optimizers.Adam(learning_rate=0.01)
    global_step = tf.Variable(0)
    tfopt = opt._create_tf_optimizer(global_step)
    assert isinstance(tfopt, tf.keras.optimizers.Adam)

  @unittest.skipIf(not has_pytorch, 'PyTorch is not installed')
  def test_adam_pytorch(self):
    """Test creating an Adam optimizer."""
    opt = optimizers.Adam(learning_rate=0.01)
    params = [torch.nn.Parameter(torch.Tensor([1.0]))]
    torchopt = opt._create_pytorch_optimizer(params)
    assert isinstance(torchopt, torch.optim.Adam)

  @unittest.skipIf(not has_tensorflow, 'TensorFlow is not installed')
  def test_adagrad_tf(self):
    """Test creating an AdaGrad optimizer."""
    opt = optimizers.AdaGrad(learning_rate=0.01)
    global_step = tf.Variable(0)
    tfopt = opt._create_tf_optimizer(global_step)
    assert isinstance(tfopt, tf.keras.optimizers.Adagrad)

  @unittest.skipIf(not has_pytorch, 'PyTorch is not installed')
  def test_adagrad_pytorch(self):
    """Test creating an AdaGrad optimizer."""
    opt = optimizers.AdaGrad(learning_rate=0.01)
    params = [torch.nn.Parameter(torch.Tensor([1.0]))]
    torchopt = opt._create_pytorch_optimizer(params)
    assert isinstance(torchopt, torch.optim.Adagrad)

  @unittest.skipIf(not has_tensorflow, 'TensorFlow is not installed')
  def test_rmsprop_tf(self):
    """Test creating an RMSProp Optimizer."""
    opt = optimizers.RMSProp(learning_rate=0.01)
    global_step = tf.Variable(0)
    tfopt = opt._create_tf_optimizer(global_step)
    assert isinstance(tfopt, tf.keras.optimizers.RMSprop)

  @unittest.skipIf(not has_pytorch, 'PyTorch is not installed')
  def test_rmsprop_pytorch(self):
    """Test creating an RMSProp Optimizer."""
    opt = optimizers.RMSProp(learning_rate=0.01)
    params = [torch.nn.Parameter(torch.Tensor([1.0]))]
    torchopt = opt._create_pytorch_optimizer(params)
    assert isinstance(torchopt, torch.optim.RMSprop)

  @unittest.skipIf(not has_tensorflow, 'TensorFlow is not installed')
  def test_gradient_descent_tf(self):
    """Test creating a Gradient Descent optimizer."""
    opt = optimizers.GradientDescent(learning_rate=0.01)
    global_step = tf.Variable(0)
    tfopt = opt._create_tf_optimizer(global_step)
    assert isinstance(tfopt, tf.keras.optimizers.SGD)

  @unittest.skipIf(not has_pytorch, 'PyTorch is not installed')
  def test_gradient_descent_pytorch(self):
    """Test creating a Gradient Descent optimizer."""
    opt = optimizers.GradientDescent(learning_rate=0.01)
    params = [torch.nn.Parameter(torch.Tensor([1.0]))]
    torchopt = opt._create_pytorch_optimizer(params)
    assert isinstance(torchopt, torch.optim.SGD)

  @unittest.skipIf(not has_tensorflow, 'TensorFlow is not installed')
  def test_exponential_decay_tf(self):
    """Test creating an optimizer with an exponentially decaying learning rate."""
    rate = optimizers.ExponentialDecay(
        initial_rate=0.001, decay_rate=0.99, decay_steps=10000)
    opt = optimizers.Adam(learning_rate=rate)
    global_step = tf.Variable(0)
    tfopt = opt._create_tf_optimizer(global_step)

  @unittest.skipIf(not has_pytorch, 'PyTorch is not installed')
  def test_exponential_decay_pytorch(self):
    """Test creating an optimizer with an exponentially decaying learning rate."""
    rate = optimizers.ExponentialDecay(
        initial_rate=0.001, decay_rate=0.99, decay_steps=10000)
    opt = optimizers.Adam(learning_rate=rate)
    params = [torch.nn.Parameter(torch.Tensor([1.0]))]
    torchopt = opt._create_pytorch_optimizer(params)
    schedule = rate._create_pytorch_schedule(torchopt)

  @unittest.skipIf(not has_tensorflow, 'TensorFlow is not installed')
  def test_polynomial_decay_tf(self):
    """Test creating an optimizer with a polynomially decaying learning rate."""
    rate = optimizers.PolynomialDecay(
        initial_rate=0.001, final_rate=0.0001, decay_steps=10000)
    opt = optimizers.Adam(learning_rate=rate)
    global_step = tf.Variable(0)
    tfopt = opt._create_tf_optimizer(global_step)

  @unittest.skipIf(not has_pytorch, 'PyTorch is not installed')
  def test_polynomial_decay_pytorch(self):
    """Test creating an optimizer with a polynomially decaying learning rate."""
    rate = optimizers.PolynomialDecay(
        initial_rate=0.001, final_rate=0.0001, decay_steps=10000)
    opt = optimizers.Adam(learning_rate=rate)
    params = [torch.nn.Parameter(torch.Tensor([1.0]))]
    torchopt = opt._create_pytorch_optimizer(params)
    schedule = rate._create_pytorch_schedule(torchopt)

  @unittest.skipIf(not has_tensorflow, 'TensorFlow is not installed')
  def test_linearCosine_decay_tf(self):
    """test creating an optimizer with a linear cosine decay to the learning rate"""
    rate = optimizers.LinearCosineDecay(initial_rate=0.1, decay_steps=10000)
    opt = optimizers.Adam(learning_rate=rate)
    global_step = tf.Variable(0)
    tfopt = opt._create_tf_optimizer(global_step)

  @unittest.skipIf(not has_pytorch, 'PyTorch is not installed')
  def test_linearCosine_decay_pytorch(self):
    """test creating an optimizer with a linear cosine decay to the learning rate"""
    rate = optimizers.LinearCosineDecay(initial_rate=0.1, decay_steps=10000)
    opt = optimizers.Adam(learning_rate=rate)
    params = [torch.nn.Parameter(torch.Tensor([1.0]))]
    torchopt = opt._create_pytorch_optimizer(params)
    schedule = rate._create_pytorch_schedule(torchopt)
