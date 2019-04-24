import deepchem as dc
import numpy as np
import tensorflow as tf
import deepchem.models.tensorgraph.layers as layers
from tensorflow.python.eager import context
from tensorflow.python.framework import test_util


class TestLayersEager(test_util.TensorFlowTestCase):
  """
  Test that layers function in eager mode.
  """

  def test_conv_1d(self):
    """Test invoking Conv1D in eager mode."""
    with context.eager_mode():
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
      assert len(layer.trainable_variables) == 2

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
      in_dim = 2
      out_dim = 3
      batch_size = 10
      input = np.random.rand(batch_size, in_dim).astype(np.float32)
      layer = layers.Dense(out_dim)
      result = layer(input)
      assert result.shape == (batch_size, out_dim)
      assert len(layer.trainable_variables) == 2

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
      width = 5
      batch_size = 10
      input = np.random.rand(batch_size, width).astype(np.float32)
      layer = layers.Highway()
      result = layer(input)
      assert result.shape == (batch_size, width)
      assert len(layer.trainable_variables) == 4

      # Creating a second layer should produce different results, since it has
      # different random weights.

      layer2 = layers.Highway()
      result2 = layer2(input)
      assert not np.allclose(result, result2)

      # But evaluating the first layer again should produce the same result as before.

      result3 = layer(input)
      assert np.allclose(result, result3)

  def test_flatten(self):
    """Test invoking Flatten in eager mode."""
    with context.eager_mode():
      input = np.random.rand(5, 10, 4).astype(np.float32)
      result = layers.Flatten()(input)
      assert result.shape == (5, 40)

  def test_reshape(self):
    """Test invoking Reshape in eager mode."""
    with context.eager_mode():
      input = np.random.rand(5, 10, 4).astype(np.float32)
      result = layers.Reshape((100, 2))(input)
      assert result.shape == (100, 2)

  def test_cast(self):
    """Test invoking Cast in eager mode."""
    with context.eager_mode():
      input = np.random.rand(5, 3)
      result = layers.Cast(dtype=tf.float32)(input)
      assert result.dtype == tf.float32

  def test_squeeze(self):
    """Test invoking Squeeze in eager mode."""
    with context.eager_mode():
      input = np.random.rand(5, 1, 4).astype(np.float32)
      result = layers.Squeeze()(input)
      assert result.shape == (5, 4)

  def test_transpose(self):
    """Test invoking Transpose in eager mode."""
    with context.eager_mode():
      input = np.random.rand(5, 10, 4).astype(np.float32)
      result = layers.Transpose((1, 2, 0))(input)
      assert result.shape == (10, 4, 5)

  def test_combine_mean_std(self):
    """Test invoking CombineMeanStd in eager mode."""
    with context.eager_mode():
      mean = np.random.rand(5, 3).astype(np.float32)
      std = np.random.rand(5, 3).astype(np.float32)
      layer = layers.CombineMeanStd(training_only=True, noise_epsilon=0.01)
      result1 = layer(mean, std, training=False)
      assert np.array_equal(result1, mean)  # No noise in test mode
      result2 = layer(mean, std, training=True)
      assert not np.array_equal(result2, mean)
      assert np.allclose(result2, mean, atol=0.1)

  def test_repeat(self):
    """Test invoking Repeat in eager mode."""
    with context.eager_mode():
      input = np.random.rand(5, 4).astype(np.float32)
      result = layers.Repeat(3)(input)
      assert result.shape == (5, 3, 4)
      assert np.array_equal(result[:, 0, :], result[:, 1, :])

  def test_gather(self):
    """Test invoking Gather in eager mode."""
    with context.eager_mode():
      input = np.random.rand(5).astype(np.float32)
      indices = [[1], [3]]
      result = layers.Gather()(input, indices)
      assert np.array_equal(result, [input[1], input[3]])

  def test_gru(self):
    """Test invoking GRU in eager mode."""
    with context.eager_mode():
      batch_size = 10
      n_hidden = 7
      in_channels = 4
      n_steps = 6
      input = np.random.rand(batch_size, n_steps,
                             in_channels).astype(np.float32)
      layer = layers.GRU(n_hidden, batch_size)
      result, state = layer(input)
      assert result.shape == (batch_size, n_steps, n_hidden)
      assert len(layer.trainable_variables) == 3

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

  def test_lstm(self):
    """Test invoking LSTM in eager mode."""
    with context.eager_mode():
      batch_size = 10
      n_hidden = 7
      in_channels = 4
      n_steps = 6
      input = np.random.rand(batch_size, n_steps,
                             in_channels).astype(np.float32)
      layer = layers.LSTM(n_hidden, batch_size)
      result, state = layer(input)
      assert result.shape == (batch_size, n_steps, n_hidden)
      assert len(layer.trainable_variables) == 3

      # Creating a second layer should produce different results, since it has
      # different random weights.

      layer2 = layers.LSTM(n_hidden, batch_size)
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
      in_dim = 2
      out_dim = 3
      n_steps = 6
      batch_size = 10
      input = np.random.rand(batch_size, n_steps, in_dim).astype(np.float32)
      layer = layers.TimeSeriesDense(out_dim)
      result = layer(input)
      assert result.shape == (batch_size, n_steps, out_dim)
      assert len(layer.trainable_variables) == 2

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
      input1 = np.random.rand(5, 10).astype(np.float32)
      input2 = np.random.rand(5, 10).astype(np.float32)
      result = layers.L1Loss()(input1, input2)
      expected = np.mean(np.abs(input1 - input2), axis=1)
      assert np.allclose(result, expected)

  def test_l2_loss(self):
    """Test invoking L2Loss in eager mode."""
    with context.eager_mode():
      input1 = np.random.rand(5, 10).astype(np.float32)
      input2 = np.random.rand(5, 10).astype(np.float32)
      result = layers.L2Loss()(input1, input2)
      expected = np.mean((input1 - input2)**2, axis=1)
      assert np.allclose(result, expected)

  def test_softmax(self):
    """Test invoking SoftMax in eager mode."""
    with context.eager_mode():
      input = np.random.rand(5, 10).astype(np.float32)
      result = layers.SoftMax()(input)
      expected = tf.nn.softmax(input)
      assert np.allclose(result, expected)

  def test_sigmoid(self):
    """Test invoking Sigmoid in eager mode."""
    with context.eager_mode():
      input = np.random.rand(5, 10).astype(np.float32)
      result = layers.Sigmoid()(input)
      expected = tf.nn.sigmoid(input)
      assert np.allclose(result, expected)

  def test_relu(self):
    """Test invoking ReLU in eager mode."""
    with context.eager_mode():
      input = np.random.normal(size=(5, 10)).astype(np.float32)
      result = layers.ReLU()(input)
      expected = tf.nn.relu(input)
      assert np.allclose(result, expected)

  def test_concat(self):
    """Test invoking Concat in eager mode."""
    with context.eager_mode():
      input1 = np.random.rand(5, 10).astype(np.float32)
      input2 = np.random.rand(5, 4).astype(np.float32)
      result = layers.Concat()(input1, input2)
      assert result.shape == (5, 14)
      assert np.array_equal(input1, result[:, :10])
      assert np.array_equal(input2, result[:, 10:])

  def test_stack(self):
    """Test invoking Stack in eager mode."""
    with context.eager_mode():
      input1 = np.random.rand(5, 4).astype(np.float32)
      input2 = np.random.rand(5, 4).astype(np.float32)
      result = layers.Stack()(input1, input2)
      assert result.shape == (5, 2, 4)
      assert np.array_equal(input1, result[:, 0, :])
      assert np.array_equal(input2, result[:, 1, :])

  def test_constant(self):
    """Test invoking Constant in eager mode."""
    with context.eager_mode():
      value = np.random.rand(5, 4).astype(np.float32)
      result = layers.Constant(value)()
      assert np.array_equal(result, value)

  def test_variable(self):
    """Test invoking Variable in eager mode."""
    with context.eager_mode():
      value = np.random.rand(5, 4).astype(np.float32)
      layer = layers.Variable(value)
      result = layer()
      assert np.array_equal(result.numpy(), value)
      assert len(layer.trainable_variables) == 1

  def test_add(self):
    """Test invoking Add in eager mode."""
    with context.eager_mode():
      result = layers.Add()([1, 2], [3, 4])
      assert np.array_equal(result, [4, 6])

  def test_multiply(self):
    """Test invoking Multiply in eager mode."""
    with context.eager_mode():
      result = layers.Multiply()([1, 2], [3, 4])
      assert np.array_equal(result, [3, 8])

  def test_divide(self):
    """Test invoking Divide in eager mode."""
    with context.eager_mode():
      result = layers.Divide()([1, 2], [2, 5])
      assert np.allclose(result, [0.5, 0.4])

  def test_log(self):
    """Test invoking Log in eager mode."""
    with context.eager_mode():
      result = layers.Log()(2.5)
      assert np.allclose(result, np.log(2.5))

  def test_exp(self):
    """Test invoking Exp in eager mode."""
    with context.eager_mode():
      result = layers.Exp()(2.5)
      assert np.allclose(result, np.exp(2.5))

  def test_interatomic_l2_distances(self):
    """Test invoking InteratomicL2Distances in eager mode."""
    with context.eager_mode():
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
      batch_size = 10
      n_features = 5
      logits = np.random.rand(batch_size, n_features).astype(np.float32)
      labels = np.random.randint(0, 2,
                                 (batch_size, n_features)).astype(np.float32)
      result = layers.SigmoidCrossEntropy()(labels, logits)
      expected = tf.nn.sigmoid_cross_entropy_with_logits(
          labels=labels, logits=logits)
      assert np.allclose(result, expected)

  def test_reduce_mean(self):
    """Test invoking ReduceMean in eager mode."""
    with context.eager_mode():
      input = np.random.rand(5, 10).astype(np.float32)
      result = layers.ReduceMean(axis=1)(input)
      assert result.shape == (5,)
      assert np.allclose(result, np.mean(input, axis=1))

  def test_reduce_max(self):
    """Test invoking ReduceMax in eager mode."""
    with context.eager_mode():
      input = np.random.rand(5, 10).astype(np.float32)
      result = layers.ReduceMax(axis=1)(input)
      assert result.shape == (5,)
      assert np.allclose(result, np.max(input, axis=1))

  def test_reduce_sum(self):
    """Test invoking ReduceSum in eager mode."""
    with context.eager_mode():
      input = np.random.rand(5, 10).astype(np.float32)
      result = layers.ReduceSum(axis=1)(input)
      assert result.shape == (5,)
      assert np.allclose(result, np.sum(input, axis=1))

  def test_reduce_square_difference(self):
    """Test invoking ReduceSquareDifference in eager mode."""
    with context.eager_mode():
      input1 = np.random.rand(5, 10).astype(np.float32)
      input2 = np.random.rand(5, 10).astype(np.float32)
      result = layers.ReduceSquareDifference(axis=1)(input1, input2)
      assert result.shape == (5,)
      assert np.allclose(result, np.mean((input1 - input2)**2, axis=1))

  def test_conv_2d(self):
    """Test invoking Conv2D in eager mode."""
    with context.eager_mode():
      length = 4
      width = 5
      in_channels = 2
      filters = 3
      kernel_size = 2
      batch_size = 10
      input = np.random.rand(batch_size, length, width,
                             in_channels).astype(np.float32)
      layer = layers.Conv2D(filters, kernel_size=kernel_size)
      result = layer(input)
      assert result.shape == (batch_size, length, width, filters)
      assert len(layer.trainable_variables) == 2

      # Creating a second layer should produce different results, since it has
      # different random weights.

      layer2 = layers.Conv2D(filters, kernel_size=kernel_size)
      result2 = layer2(input)
      assert not np.allclose(result, result2)

      # But evaluating the first layer again should produce the same result as before.

      result3 = layer(input)
      assert np.allclose(result, result3)

  def test_conv_3d(self):
    """Test invoking Conv3D in eager mode."""
    with context.eager_mode():
      length = 4
      width = 5
      depth = 6
      in_channels = 2
      filters = 3
      kernel_size = 2
      batch_size = 10
      input = np.random.rand(batch_size, length, width, depth,
                             in_channels).astype(np.float32)
      layer = layers.Conv3D(filters, kernel_size=kernel_size)
      result = layer(input)
      assert result.shape == (batch_size, length, width, depth, filters)
      assert len(layer.trainable_variables) == 2

      # Creating a second layer should produce different results, since it has
      # different random weights.

      layer2 = layers.Conv3D(filters, kernel_size=kernel_size)
      result2 = layer2(input)
      assert not np.allclose(result, result2)

      # But evaluating the first layer again should produce the same result as before.

      result3 = layer(input)
      assert np.allclose(result, result3)

  def test_conv_2d_transpose(self):
    """Test invoking Conv2DTranspose in eager mode."""
    with context.eager_mode():
      length = 4
      width = 5
      in_channels = 2
      filters = 3
      kernel_size = 2
      stride = 2
      batch_size = 10
      input = np.random.rand(batch_size, length, width,
                             in_channels).astype(np.float32)
      layer = layers.Conv2DTranspose(
          filters, kernel_size=kernel_size, stride=stride)
      result = layer(input)
      assert result.shape == (batch_size, length * stride, width * stride,
                              filters)
      assert len(layer.trainable_variables) == 2

      # Creating a second layer should produce different results, since it has
      # different random weights.

      layer2 = layers.Conv2DTranspose(
          filters, kernel_size=kernel_size, stride=stride)
      result2 = layer2(input)
      assert not np.allclose(result, result2)

      # But evaluating the first layer again should produce the same result as before.

      result3 = layer(input)
      assert np.allclose(result, result3)

  def test_conv_3d_transpose(self):
    """Test invoking Conv3DTranspose in eager mode."""
    with context.eager_mode():
      length = 4
      width = 5
      depth = 6
      in_channels = 2
      filters = 3
      kernel_size = 2
      stride = 2
      batch_size = 10
      input = np.random.rand(batch_size, length, width, depth,
                             in_channels).astype(np.float32)
      layer = layers.Conv3DTranspose(
          filters, kernel_size=kernel_size, stride=stride)
      result = layer(input)
      assert result.shape == (batch_size, length * stride, width * stride,
                              depth * stride, filters)
      assert len(layer.trainable_variables) == 2

      # Creating a second layer should produce different results, since it has
      # different random weights.

      layer2 = layers.Conv3DTranspose(
          filters, kernel_size=kernel_size, stride=stride)
      result2 = layer2(input)
      assert not np.allclose(result, result2)

      # But evaluating the first layer again should produce the same result as before.

      result3 = layer(input)
      assert np.allclose(result, result3)

  def test_max_pool_1d(self):
    """Test invoking MaxPool1D in eager mode."""
    with context.eager_mode():
      input = np.random.rand(4, 6, 8).astype(np.float32)
      result = layers.MaxPool1D(strides=2)(input)
      assert result.shape == (4, 3, 8)

  def test_max_pool_2d(self):
    """Test invoking MaxPool2D in eager mode."""
    with context.eager_mode():
      input = np.random.rand(2, 4, 6, 8).astype(np.float32)
      result = layers.MaxPool2D()(input)
      assert result.shape == (2, 2, 3, 8)

  def test_max_pool_3d(self):
    """Test invoking MaxPool3D in eager mode."""
    with context.eager_mode():
      input = np.random.rand(2, 4, 6, 8, 2).astype(np.float32)
      result = layers.MaxPool3D()(input)
      assert result.shape == (2, 2, 3, 4, 2)

  def test_graph_conv(self):
    """Test invoking GraphConv in eager mode."""
    with context.eager_mode():
      out_channels = 2
      n_atoms = 4  # In CCC and C, there are 4 atoms
      raw_smiles = ['CCC', 'C']
      import rdkit
      mols = [rdkit.Chem.MolFromSmiles(s) for s in raw_smiles]
      featurizer = dc.feat.graph_features.ConvMolFeaturizer()
      mols = featurizer.featurize(mols)
      multi_mol = dc.feat.mol_graphs.ConvMol.agglomerate_mols(mols)
      atom_features = multi_mol.get_atom_features().astype(np.float32)
      degree_slice = multi_mol.deg_slice
      membership = multi_mol.membership
      deg_adjs = multi_mol.get_deg_adjacency_lists()[1:]
      args = [atom_features, degree_slice, membership] + deg_adjs
      layer = layers.GraphConv(out_channels)
      result = layer(*args)
      assert result.shape == (n_atoms, out_channels)
      num_deg = 2 * layer.max_degree + (1 - layer.min_degree)
      assert len(layer.trainable_variables) == 2 * num_deg

  def test_graph_pool(self):
    """Test invoking GraphPool in eager mode."""
    with context.eager_mode():
      n_atoms = 4  # In CCC and C, there are 4 atoms
      raw_smiles = ['CCC', 'C']
      import rdkit
      mols = [rdkit.Chem.MolFromSmiles(s) for s in raw_smiles]
      featurizer = dc.feat.graph_features.ConvMolFeaturizer()
      mols = featurizer.featurize(mols)
      multi_mol = dc.feat.mol_graphs.ConvMol.agglomerate_mols(mols)
      atom_features = multi_mol.get_atom_features().astype(np.float32)
      degree_slice = multi_mol.deg_slice
      membership = multi_mol.membership
      deg_adjs = multi_mol.get_deg_adjacency_lists()[1:]
      args = [atom_features, degree_slice, membership] + deg_adjs
      result = layers.GraphPool()(*args)
      assert result.shape[0] == n_atoms
      # TODO What should shape[1] be?  It's not documented.

  def test_graph_gather(self):
    """Test invoking GraphGather in eager mode."""
    with context.eager_mode():
      batch_size = 2
      n_features = 75
      n_atoms = 4  # In CCC and C, there are 4 atoms
      raw_smiles = ['CCC', 'C']
      import rdkit
      mols = [rdkit.Chem.MolFromSmiles(s) for s in raw_smiles]
      featurizer = dc.feat.graph_features.ConvMolFeaturizer()
      mols = featurizer.featurize(mols)
      multi_mol = dc.feat.mol_graphs.ConvMol.agglomerate_mols(mols)
      atom_features = multi_mol.get_atom_features().astype(np.float32)
      degree_slice = multi_mol.deg_slice
      membership = multi_mol.membership
      deg_adjs = multi_mol.get_deg_adjacency_lists()[1:]
      args = [atom_features, degree_slice, membership] + deg_adjs
      result = layers.GraphGather(batch_size)(*args)
      # TODO(rbharath): Why is it 2*n_features instead of n_features?
      assert result.shape == (batch_size, 2 * n_features)

  def test_lstm_step(self):
    """Test invoking LSTMStep in eager mode."""
    with context.eager_mode():
      max_depth = 5
      n_test = 5
      n_feat = 10
      y = np.random.rand(n_test, 2 * n_feat).astype(np.float32)
      state_zero = np.random.rand(n_test, n_feat).astype(np.float32)
      state_one = np.random.rand(n_test, n_feat).astype(np.float32)
      layer = layers.LSTMStep(n_feat, 2 * n_feat)
      result = layer(y, state_zero, state_one)
      h_out, h_copy_out, c_out = (result[0], result[1][0], result[1][1])
      assert h_out.shape == (n_test, n_feat)
      assert h_copy_out.shape == (n_test, n_feat)
      assert c_out.shape == (n_test, n_feat)
      assert len(layer.trainable_variables) == 3

  def test_attn_lstm_embedding(self):
    """Test invoking AttnLSTMEmbedding in eager mode."""
    with context.eager_mode():
      max_depth = 5
      n_test = 5
      n_support = 11
      n_feat = 10
      test = np.random.rand(n_test, n_feat).astype(np.float32)
      support = np.random.rand(n_support, n_feat).astype(np.float32)
      layer = layers.AttnLSTMEmbedding(n_test, n_support, n_feat, max_depth)
      test_out, support_out = layer(test, support)
      assert test_out.shape == (n_test, n_feat)
      assert support_out.shape == (n_support, n_feat)
      assert len(layer.trainable_variables) == 6

  def test_iter_ref_lstm_embedding(self):
    """Test invoking IterRefLSTMEmbedding in eager mode."""
    with context.eager_mode():
      max_depth = 5
      n_test = 5
      n_support = 11
      n_feat = 10
      test = np.random.rand(n_test, n_feat).astype(np.float32)
      support = np.random.rand(n_support, n_feat).astype(np.float32)
      layer = layers.IterRefLSTMEmbedding(n_test, n_support, n_feat, max_depth)
      test_out, support_out = layer(test, support)
      assert test_out.shape == (n_test, n_feat)
      assert support_out.shape == (n_support, n_feat)
      assert len(layer.trainable_variables) == 12

  def test_batch_norm(self):
    """Test invoking BatchNorm in eager mode."""
    with context.eager_mode():
      batch_size = 10
      n_features = 5
      input = np.random.rand(batch_size, n_features).astype(np.float32)
      layer = layers.BatchNorm()
      result = layer(input)
      assert result.shape == (batch_size, n_features)
      assert len(layer.trainable_variables) == 2

  def test_weighted_error(self):
    """Test invoking WeightedError in eager mode."""
    with context.eager_mode():
      input1 = np.random.rand(5, 10).astype(np.float32)
      input2 = np.random.rand(5, 10).astype(np.float32)
      result = layers.WeightedError()(input1, input2)
      expected = np.sum(input1 * input2)
      assert np.allclose(result, expected)

  def test_vina_free_energy(self):
    """Test invoking VinaFreeEnergy in eager mode."""
    with context.eager_mode():
      n_atoms = 5
      m_nbrs = 1
      ndim = 3
      nbr_cutoff = 1
      start = 0
      stop = 4
      X = np.random.rand(n_atoms, ndim).astype(np.float32)
      Z = np.random.randint(0, 2, (n_atoms)).astype(np.float32)
      layer = layers.VinaFreeEnergy(n_atoms, m_nbrs, ndim, nbr_cutoff, start,
                                    stop)
      result = layer(X, Z)
      assert len(layer.trainable_variables) == 6
      assert result.shape == tuple()

      # Creating a second layer should produce different results, since it has
      # different random weights.

      layer2 = layers.VinaFreeEnergy(n_atoms, m_nbrs, ndim, nbr_cutoff, start,
                                     stop)
      result2 = layer2(X, Z)
      assert not np.allclose(result, result2)

      # But evaluating the first layer again should produce the same result as before.

      result3 = layer(X, Z)
      assert np.allclose(result, result3)

  def test_weighted_linear_combo(self):
    """Test invoking WeightedLinearCombo in eager mode."""
    with context.eager_mode():
      input1 = np.random.rand(5, 10).astype(np.float32)
      input2 = np.random.rand(5, 10).astype(np.float32)
      layer = layers.WeightedLinearCombo()
      result = layer(input1, input2)
      assert len(layer.trainable_variables) == 2
      expected = input1 * layer.trainable_variables[0] + input2 * layer.trainable_variables[1]
      assert np.allclose(result, expected)

  def test_neighbor_list(self):
    """Test invoking NeighborList in eager mode."""
    with context.eager_mode():
      N_atoms = 5
      start = 0
      stop = 12
      nbr_cutoff = 3
      ndim = 3
      M_nbrs = 2
      coords = start + np.random.rand(N_atoms, ndim) * (stop - start)
      coords = tf.cast(tf.stack(coords), tf.float32)
      layer = layers.NeighborList(N_atoms, M_nbrs, ndim, nbr_cutoff, start,
                                  stop)
      result = layer(coords)
      assert result.shape == (N_atoms, M_nbrs)

  def test_dropout(self):
    """Test invoking Dropout in eager mode."""
    with context.eager_mode():
      rate = 0.5
      input = np.random.rand(5, 10).astype(np.float32)
      layer = layers.Dropout(rate)
      result1 = layer(input, training=False)
      assert np.allclose(result1, input)
      result2 = layer(input, training=True)
      assert not np.allclose(result2, input)
      nonzero = result2.numpy() != 0
      assert np.allclose(result2.numpy()[nonzero], input[nonzero] / rate)

  def test_atomic_convolution(self):
    """Test invoking AtomicConvolution in eager mode."""
    with context.eager_mode():
      batch_size = 4
      max_atoms = 5
      max_neighbors = 2
      dimensions = 3
      params = [[5.0, 2.0, 0.5], [10.0, 2.0, 0.5]]
      input1 = np.random.rand(batch_size, max_atoms,
                              dimensions).astype(np.float32)
      input2 = np.random.randint(
          max_atoms, size=(batch_size, max_atoms, max_neighbors))
      input3 = np.random.randint(
          1, 10, size=(batch_size, max_atoms, max_neighbors))
      layer = layers.AtomicConvolution(radial_params=params)
      result = layer(input1, input2, input3)
      assert result.shape == (batch_size, max_atoms, len(params))
      assert len(layer.trainable_variables) == 3

  def test_alpha_share_layer(self):
    """Test invoking AlphaShareLayer in eager mode."""
    with context.eager_mode():
      batch_size = 10
      length = 6
      input1 = np.random.rand(batch_size, length).astype(np.float32)
      input2 = np.random.rand(batch_size, length).astype(np.float32)
      layer = layers.AlphaShareLayer()
      result = layer(input1, input2)
      assert input1.shape == result[0].shape
      assert input2.shape == result[1].shape

      # Creating a second layer should produce different results, since it has
      # different random weights.

      layer2 = layers.AlphaShareLayer()
      result2 = layer2(input1, input2)
      assert not np.allclose(result[0], result2[0])
      assert not np.allclose(result[1], result2[1])

      # But evaluating the first layer again should produce the same result as before.

      result3 = layer(input1, input2)
      assert np.allclose(result[0], result3[0])
      assert np.allclose(result[1], result3[1])

  def test_sluice_loss(self):
    """Test invoking SluiceLoss in eager mode."""
    with context.eager_mode():
      input1 = np.ones((3, 4)).astype(np.float32)
      input2 = np.ones((2, 2)).astype(np.float32)
      result = layers.SluiceLoss()(input1, input2)
      assert np.allclose(result, 40.0)

  def test_beta_share(self):
    """Test invoking BetaShare in eager mode."""
    with context.eager_mode():
      batch_size = 10
      length = 6
      input1 = np.random.rand(batch_size, length).astype(np.float32)
      input2 = np.random.rand(batch_size, length).astype(np.float32)
      layer = layers.BetaShare()
      result = layer(input1, input2)
      assert input1.shape == result.shape
      assert input2.shape == result.shape

      # Creating a second layer should produce different results, since it has
      # different random weights.

      layer2 = layers.BetaShare()
      result2 = layer2(input1, input2)
      assert not np.allclose(result, result2)

      # But evaluating the first layer again should produce the same result as before.

      result3 = layer(input1, input2)
      assert np.allclose(result, result3)

  def test_ani_feat(self):
    """Test invoking ANIFeat in eager mode."""
    with context.eager_mode():
      batch_size = 10
      max_atoms = 5
      input = np.random.rand(batch_size, max_atoms, 4).astype(np.float32)
      layer = layers.ANIFeat(max_atoms=max_atoms)
      result = layer(input)
      # TODO What should the output shape be?  It's not documented, and there
      # are no other test cases for it.

  def test_graph_embed_pool_layer(self):
    """Test invoking GraphEmbedPoolLayer in eager mode."""
    with context.eager_mode():
      V = np.random.uniform(size=(10, 100, 50)).astype(np.float32)
      adjs = np.random.uniform(size=(10, 100, 5, 100)).astype(np.float32)
      layer = layers.GraphEmbedPoolLayer(num_vertices=6)
      result = layer(V, adjs)
      assert result[0].shape == (10, 6, 50)
      assert result[1].shape == (10, 6, 5, 6)

      # Creating a second layer should produce different results, since it has
      # different random weights.

      layer2 = layers.GraphEmbedPoolLayer(num_vertices=6)
      result2 = layer2(V, adjs)
      assert not np.allclose(result[0], result2[0])
      assert not np.allclose(result[1], result2[1])

      # But evaluating the first layer again should produce the same result as before.

      result3 = layer(V, adjs)
      assert np.allclose(result[0], result3[0])
      assert np.allclose(result[1], result3[1])

  def test_graph_cnn(self):
    """Test invoking GraphCNN in eager mode."""
    with context.eager_mode():
      V = np.random.uniform(size=(10, 100, 50)).astype(np.float32)
      adjs = np.random.uniform(size=(10, 100, 5, 100)).astype(np.float32)
      layer = layers.GraphCNN(num_filters=6)
      result = layer(V, adjs)
      assert result.shape == (10, 100, 6)

      # Creating a second layer should produce different results, since it has
      # different random weights.

      layer2 = layers.GraphCNN(num_filters=6)
      result2 = layer2(V, adjs)
      assert not np.allclose(result, result2)

      # But evaluating the first layer again should produce the same result as before.

      result3 = layer(V, adjs)
      assert np.allclose(result, result3)

  def test_hinge_loss(self):
    """Test invoking HingeLoss in eager mode."""
    with context.eager_mode():
      n_labels = 1
      n_logits = 1
      logits = np.random.rand(n_logits).astype(np.float32)
      labels = np.random.rand(n_labels).astype(np.float32)
      result = layers.HingeLoss()(labels, logits)
      assert result.shape == (n_labels,)
