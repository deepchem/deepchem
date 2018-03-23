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
        input = np.random.rand(batch_size, width, in_channels).astype(
            np.float32)
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
        assert np.array_equal(result1, mean)  # No noise in test mode
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
        assert np.array_equal(result[:, 0, :], result[:, 1, :])

  def test_gather(self):
    """Test invoking Gather in eager mode."""
    with context.eager_mode():
      with tfe.IsolateTest():
        input = np.random.rand(5).astype(np.float32)
        indices = [[1], [3]]
        result = layers.Gather()(input, indices)
        assert np.array_equal(result, [input[1], input[3]])

  def test_gru(self):
    """Test invoking GRU in eager mode."""
    with context.eager_mode():
      with tfe.IsolateTest():
        batch_size = 10
        n_hidden = 7
        in_channels = 4
        n_steps = 6
        input = np.random.rand(batch_size, n_steps, in_channels).astype(
            np.float32)
        layer = layers.GRU(n_hidden, batch_size)
        result, state = layer(input)
        assert result.shape == (batch_size, n_steps, n_hidden)

        # Creating a second layer should produce different results, since it has
        # different random weights.

        layer2 = layers.GRU(n_hidden, batch_size)
        result2, state2 = layer2(input)
        assert not np.allclose(result, result2)

        # But evaluating the first layer again should produce the same result as before.

        result3, state3 = layer(input)
        assert np.allclose(result, result3)

        # But if we specify a different starting state, that should produce a
        # different result.

        result4, state4 = layer(input, initial_state=state3)
        assert not np.allclose(result, result4)

  def test_time_series_dense(self):
    """Test invoking TimeSeriesDense in eager mode."""
    with context.eager_mode():
      with tfe.IsolateTest():
        in_dim = 2
        out_dim = 3
        n_steps = 6
        batch_size = 10
        input = np.random.rand(batch_size, n_steps, in_dim).astype(np.float32)
        layer = layers.TimeSeriesDense(out_dim)
        result = layer(input)
        assert result.shape == (batch_size, n_steps, out_dim)

        # Creating a second layer should produce different results, since it has
        # different random weights.

        layer2 = layers.TimeSeriesDense(out_dim)
        result2 = layer2(input)
        assert not np.allclose(result, result2)

        # But evaluating the first layer again should produce the same result as before.

        result3 = layer(input)
        assert np.allclose(result, result3)

  def test_l1_loss(self):
    """Test invoking L1Loss in eager mode."""
    with context.eager_mode():
      with tfe.IsolateTest():
        input1 = np.random.rand(5, 10).astype(np.float32)
        input2 = np.random.rand(5, 10).astype(np.float32)
        result = layers.L1Loss()(input1, input2)
        expected = np.mean(np.abs(input1 - input2), axis=1)
        assert np.allclose(result, expected)

  def test_l2_loss(self):
    """Test invoking L2Loss in eager mode."""
    with context.eager_mode():
      with tfe.IsolateTest():
        input1 = np.random.rand(5, 10).astype(np.float32)
        input2 = np.random.rand(5, 10).astype(np.float32)
        result = layers.L2Loss()(input1, input2)
        expected = np.mean((input1 - input2)**2, axis=1)
        assert np.allclose(result, expected)

  def test_softmax(self):
    """Test invoking SoftMax in eager mode."""
    with context.eager_mode():
      with tfe.IsolateTest():
        input = np.random.rand(5, 10).astype(np.float32)
        result = layers.SoftMax()(input)
        expected = tf.nn.softmax(input)
        assert np.allclose(result, expected)

  def test_sigmoid(self):
    """Test invoking Sigmoid in eager mode."""
    with context.eager_mode():
      with tfe.IsolateTest():
        input = np.random.rand(5, 10).astype(np.float32)
        result = layers.Sigmoid()(input)
        expected = tf.nn.sigmoid(input)
        assert np.allclose(result, expected)

  def test_relu(self):
    """Test invoking ReLU in eager mode."""
    with context.eager_mode():
      with tfe.IsolateTest():
        input = np.random.normal(size=(5, 10)).astype(np.float32)
        result = layers.ReLU()(input)
        expected = tf.nn.relu(input)
        assert np.allclose(result, expected)

  def test_concat(self):
    """Test invoking Concat in eager mode."""
    with context.eager_mode():
      with tfe.IsolateTest():
        input1 = np.random.rand(5, 10).astype(np.float32)
        input2 = np.random.rand(5, 4).astype(np.float32)
        result = layers.Concat()(input1, input2)
        assert result.shape == (5, 14)
        assert np.array_equal(input1, result[:, :10])
        assert np.array_equal(input2, result[:, 10:])

  def test_stack(self):
    """Test invoking Stack in eager mode."""
    with context.eager_mode():
      with tfe.IsolateTest():
        input1 = np.random.rand(5, 4).astype(np.float32)
        input2 = np.random.rand(5, 4).astype(np.float32)
        result = layers.Stack()(input1, input2)
        assert result.shape == (5, 2, 4)
        assert np.array_equal(input1, result[:, 0, :])
        assert np.array_equal(input2, result[:, 1, :])

  def test_constant(self):
    """Test invoking Constant in eager mode."""
    with context.eager_mode():
      with tfe.IsolateTest():
        value = np.random.rand(5, 4).astype(np.float32)
        result = layers.Constant(value)()
        assert np.array_equal(result, value)

  def test_variable(self):
    """Test invoking Variable in eager mode."""
    with context.eager_mode():
      with tfe.IsolateTest():
        value = np.random.rand(5, 4).astype(np.float32)
        result = layers.Variable(value)()
        assert np.array_equal(result.numpy(), value)

  def test_add(self):
    """Test invoking Add in eager mode."""
    with context.eager_mode():
      with tfe.IsolateTest():
        result = layers.Add()([1, 2], [3, 4])
        assert np.array_equal(result, [4, 6])

  def test_multiply(self):
    """Test invoking Multiply in eager mode."""
    with context.eager_mode():
      with tfe.IsolateTest():
        result = layers.Multiply()([1, 2], [3, 4])
        assert np.array_equal(result, [3, 8])

  def test_divide(self):
    """Test invoking Divide in eager mode."""
    with context.eager_mode():
      with tfe.IsolateTest():
        result = layers.Divide()([1, 2], [2, 5])
        assert np.array_equal(result, [0.5, 0.4])

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

  def test_interatomic_l2_distances(self):
    """Test invoking InteratomicL2Distances in eager mode."""
    with context.eager_mode():
      with tfe.IsolateTest():
        atoms = 5
        neighbors = 2
        coords = np.random.rand(atoms, 3)
        neighbor_list = np.random.randint(0, atoms, size=(atoms, neighbors))
        layer = layers.InteratomicL2Distances(atoms, neighbors, 3)
        result = layer(coords, neighbor_list)
        assert result.shape == (atoms, neighbors)
        for atom in range(atoms):
          for neighbor in range(neighbors):
            delta = coords[atom] - coords[neighbor_list[atom, neighbor]]
            dist2 = np.dot(delta, delta)
            assert np.allclose(dist2, result[atom, neighbor])

  def test_sparse_softmax_cross_entropy(self):
    """Test invoking SparseSoftMaxCrossEntropy in eager mode."""
    with context.eager_mode():
      with tfe.IsolateTest():
        batch_size = 10
        n_features = 5
        logits = np.random.rand(batch_size, n_features).astype(np.float32)
        labels = np.random.rand(batch_size).astype(np.int32)
        result = layers.SparseSoftMaxCrossEntropy()(labels, logits)
        expected = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits)
        assert np.allclose(result, expected)

  def test_softmax_cross_entropy(self):
    """Test invoking SoftMaxCrossEntropy in eager mode."""
    with context.eager_mode():
      with tfe.IsolateTest():
        batch_size = 10
        n_features = 5
        logits = np.random.rand(batch_size, n_features).astype(np.float32)
        labels = np.random.rand(batch_size, n_features).astype(np.float32)
        result = layers.SoftMaxCrossEntropy()(labels, logits)
        expected = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=labels, logits=logits)
        assert np.allclose(result, expected)

  def test_sigmoid_cross_entropy(self):
    """Test invoking SigmoidCrossEntropy in eager mode."""
    with context.eager_mode():
      with tfe.IsolateTest():
        batch_size = 10
        n_features = 5
        logits = np.random.rand(batch_size, n_features).astype(np.float32)
        labels = np.random.randint(0, 2, (batch_size, n_features)).astype(
            np.float32)
        result = layers.SigmoidCrossEntropy()(labels, logits)
        expected = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=labels, logits=logits)
        assert np.allclose(result, expected)
