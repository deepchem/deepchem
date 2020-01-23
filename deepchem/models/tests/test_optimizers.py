import deepchem.models.optimizers as optimizers
import tensorflow as tf
from tensorflow.python.framework import test_util


class TestLayers(test_util.TensorFlowTestCase):
  """Test optimizers and related classes."""

  def test_adam(self):
    """Test creating an Adam optimizer."""
    opt = optimizers.Adam(learning_rate=0.01)
    with self.session() as sess:
      global_step = tf.Variable(0)
      tfopt = opt._create_optimizer(global_step)
      assert isinstance(tfopt, tf.keras.optimizers.Adam)

  def test_rmsprop(self):
    """Test creating an RMSProp Optimizer."""
    opt = optimizers.RMSProp(learning_rate=0.01)
    with self.session() as sess:
      global_step = tf.Variable(0)
      tfopt = opt._create_optimizer(global_step)
      assert isinstance(tfopt, tf.keras.optimizers.RMSprop)

  def test_gradient_descent(self):
    """Test creating a Gradient Descent optimizer."""
    opt = optimizers.GradientDescent(learning_rate=0.01)
    with self.session() as sess:
      global_step = tf.Variable(0)
      tfopt = opt._create_optimizer(global_step)
      assert isinstance(tfopt, tf.keras.optimizers.SGD)

  def test_exponential_decay(self):
    """Test creating an optimizer with an exponentially decaying learning rate."""
    rate = optimizers.ExponentialDecay(
        initial_rate=0.001, decay_rate=0.99, decay_steps=10000)
    opt = optimizers.Adam(learning_rate=rate)
    with self.session() as sess:
      global_step = tf.Variable(0)
      tfopt = opt._create_optimizer(global_step)

  def test_polynomial_decay(self):
    """Test creating an optimizer with a polynomially decaying learning rate."""
    rate = optimizers.PolynomialDecay(
        initial_rate=0.001, final_rate=0.0001, decay_steps=10000)
    opt = optimizers.Adam(learning_rate=rate)
    with self.session() as sess:
      global_step = tf.Variable(0)
      tfopt = opt._create_optimizer(global_step)

  def test_linearCosine_decay(self):
    """test creating an optimizer with a linear cosine decay to the learning rate"""
    rate = optimizers.LinearCosineDecay(initial_rate=0.1, decay_steps=10000)
    opt = optimizers.Adam(learning_rate=rate)
    with self.session() as sess:
      global_step = tf.Variable(0)
      tfopt = opt._create_optimizer(global_step)
