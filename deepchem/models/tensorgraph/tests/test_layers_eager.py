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

  def test_dense(self):
    """Test invoking Dense in eager mode."""
    with context.eager_mode():
      with tfe.IsolateTest():
        in_dim = 2
        out_dim = 3
        batch_size = 10
        input = np.random.rand(batch_size, in_dim).astype(np.float32)
        layer = layers.Dense(out_dim)
        result = layer(input)
        assert result.shape == (batch_size, out_dim)

        # Creating a second layer should produce different results, since it has
        # different random weights.

        layer2 = layers.Dense(out_dim)
        result2 = layer2(input)
        assert not np.allclose(result, result2)

        # But evaluating the first layer again should produce the same result as before.

        result3 = layer(input)
        assert np.allclose(result, result3)

  def test_highway(self):
    """Test invoking Highway in eager mode."""
    with context.eager_mode():
      with tfe.IsolateTest():
        width = 5
        batch_size = 10
        input = np.random.rand(batch_size, width).astype(np.float32)
        layer = layers.Highway(width)
        result = layer(input)
        assert result.shape == (batch_size, width)

        # Creating a second layer should produce different results, since it has
        # different random weights.

        layer2 = layers.Highway(width)
        result2 = layer2(input)
        assert not np.allclose(result, result2)

        # But evaluating the first layer again should produce the same result as before.

        result3 = layer(input)
        assert np.allclose(result, result3)

  def test_flatten(self):
    """Test invoking Flatten in eager mode."""
    with context.eager_mode():
      with tfe.IsolateTest():
        input = np.random.rand(5, 10, 4).astype(np.float32)
        result = layers.Flatten()(input)
        assert result.shape == (5, 40)

  def test_reshape(self):
    """Test invoking Reshape in eager mode."""
    with context.eager_mode():
      with tfe.IsolateTest():
        input = np.random.rand(5, 10, 4).astype(np.float32)
        result = layers.Reshape((100, 2))(input)
        assert result.shape == (100, 2)

  def test_cast(self):
    """Test invoking Cast in eager mode."""
    with context.eager_mode():
      with tfe.IsolateTest():
        input = np.random.rand(5, 3)
        result = layers.Cast(dtype=tf.float32)(input)
        assert result.dtype == tf.float32

  def test_squeeze(self):
    """Test invoking Squeeze in eager mode."""
    with context.eager_mode():
      with tfe.IsolateTest():
        input = np.random.rand(5, 1, 4).astype(np.float32)
        result = layers.Squeeze()(input)
        assert result.shape == (5, 4)

  def test_transpose(self):
    """Test invoking Transpose in eager mode."""
    with context.eager_mode():
      with tfe.IsolateTest():
        input = np.random.rand(5, 10, 4).astype(np.float32)
        result = layers.Transpose((1, 2, 0))(input)
        assert result.shape == (10, 4, 5)

  def test_combine_mean_std(self):
    """Test invoking CombineMeanStd in eager mode."""
    with context.eager_mode():
      with tfe.IsolateTest():
        mean = np.random.rand(5, 3).astype(np.float32)
        std = np.random.rand(5, 3).astype(np.float32)
        layer = layers.CombineMeanStd(training_only=True, noise_epsilon=0.01)
        result1 = layer(mean, std)
        assert np.array_equal(result1, mean) # No noise in test mode
        result2 = layer(mean, std, training=True)
        assert not np.array_equal(result2, mean)
        assert np.allclose(result2, mean, atol=0.1)

  def test_repeat(self):
    """Test invoking Repeat in eager mode."""
    with context.eager_mode():
      with tfe.IsolateTest():
        input = np.random.rand(5, 4).astype(np.float32)
        result = layers.Repeat(3)(input)
        assert result.shape == (5, 3, 4)
        assert np.array_equal(result[:,0,:], result[:,1,:])

  def test_gather(self):
    """Test invoking Gather in eager mode."""
    with context.eager_mode():
      with tfe.IsolateTest():
        input = np.random.rand(5).astype(np.float32)
        indices = [[1], [3]]
        result = layers.Gather()(input, indices)
        assert np.array_equal(result, [input[1], input[3]])

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
