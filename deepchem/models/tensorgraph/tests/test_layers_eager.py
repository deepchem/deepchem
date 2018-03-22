import numpy as np
import tensorflow as tf
import deepchem.models.tensorgraph.layers as layers
import tensorflow.contrib.eager as tfe
from tensorflow.python.eager import context
from tensorflow.python.framework import test_util

class TestLayersEager(test_util.TensorFlowTestCase):
  """
  Test that layers function in eager mode.
  """

  def test_conv_1d(self):
    """Test invoking Conv1D in eager mode."""
    with context.eager_mode():
      with tfe.IsolateTest():
        width = 5
        in_channels = 2
        filters = 3
        kernel_size = 2
        batch_size = 10
        input = np.random.rand(batch_size, width, in_channels).astype(np.float32)
        layer = layers.Conv1D(filters, kernel_size)
        result = layer(input)
        self.assertEqual(result.shape[0], batch_size)
        self.assertEqual(result.shape[2], filters)

        # Creating a second layer should produce different results, since it has
        # different random weights.

        layer2 = layers.Conv1D(filters, kernel_size)
        result2 = layer2(input)
        assert not np.allclose(result, result2)

        # But evaluating the first layer again should produce the same result as before.

        result3 = layer(input)
        assert np.allclose(result, result3)


  def test_add(self):
    """Test invoking Add in eager mode."""
    with context.eager_mode():
      with tfe.IsolateTest():
        result = layers.Add()([1,2], [3,4])
        assert np.array_equal(result, [4,6])

  def test_multiply(self):
    """Test invoking Multiply in eager mode."""
    with context.eager_mode():
      with tfe.IsolateTest():
        result = layers.Multiply()([1,2], [3,4])
        assert np.array_equal(result, [3,8])

  def test_divide(self):
    """Test invoking Divide in eager mode."""
    with context.eager_mode():
      with tfe.IsolateTest():
        result = layers.Divide()([1,2], [2,5])
        assert np.array_equal(result, [0.5,0.4])

  def test_log(self):
    """Test invoking Log in eager mode."""
    with context.eager_mode():
      with tfe.IsolateTest():
        result = layers.Log()(2.5)
        assert np.allclose(result, np.log(2.5))

  def test_exp(self):
    """Test invoking Exp in eager mode."""
    with context.eager_mode():
      with tfe.IsolateTest():
        result = layers.Exp()(2.5)
        assert np.allclose(result, np.exp(2.5))
