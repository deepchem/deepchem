import numpy as np
import rdkit
import tensorflow as tf
from tensorflow.python.framework import test_util

from deepchem.feat.graph_features import ConvMolFeaturizer
from deepchem.feat.mol_graphs import ConvMol
from deepchem.models.tensorgraph.layers import Add, MaxPool2D, MaxPool3D, AvgPool2D, GraphCNN, GraphEmbedPoolLayer, Cast
from deepchem.models.tensorgraph.layers import AlphaShareLayer
from deepchem.models.tensorgraph.layers import AttnLSTMEmbedding
from deepchem.models.tensorgraph.layers import BatchNorm
from deepchem.models.tensorgraph.layers import BetaShare
from deepchem.models.tensorgraph.layers import CombineMeanStd
from deepchem.models.tensorgraph.layers import Concat
from deepchem.models.tensorgraph.layers import Constant
from deepchem.models.tensorgraph.layers import Conv1D, Squeeze
from deepchem.models.tensorgraph.layers import Conv2D
from deepchem.models.tensorgraph.layers import Conv2DTranspose
from deepchem.models.tensorgraph.layers import Conv3D
from deepchem.models.tensorgraph.layers import Conv3DTranspose
from deepchem.models.tensorgraph.layers import Dense
from deepchem.models.tensorgraph.layers import Exp
from deepchem.models.tensorgraph.layers import Flatten
from deepchem.models.tensorgraph.layers import GRU
from deepchem.models.tensorgraph.layers import Gather
from deepchem.models.tensorgraph.layers import GraphConv
from deepchem.models.tensorgraph.layers import GraphGather
from deepchem.models.tensorgraph.layers import HingeLoss
from deepchem.models.tensorgraph.layers import Input
from deepchem.models.tensorgraph.layers import InputFifoQueue
from deepchem.models.tensorgraph.layers import InteratomicL2Distances
from deepchem.models.tensorgraph.layers import IterRefLSTMEmbedding
from deepchem.models.tensorgraph.layers import L2Loss
from deepchem.models.tensorgraph.layers import LSTM
from deepchem.models.tensorgraph.layers import LSTMStep
from deepchem.models.tensorgraph.layers import Log
from deepchem.models.tensorgraph.layers import Multiply
from deepchem.models.tensorgraph.layers import ReduceMean
from deepchem.models.tensorgraph.layers import ReduceSquareDifference
from deepchem.models.tensorgraph.layers import ReduceSum
from deepchem.models.tensorgraph.layers import ReLU
from deepchem.models.tensorgraph.layers import Repeat
from deepchem.models.tensorgraph.layers import Reshape
from deepchem.models.tensorgraph.layers import SluiceLoss
from deepchem.models.tensorgraph.layers import Sigmoid
from deepchem.models.tensorgraph.layers import SigmoidCrossEntropy
from deepchem.models.tensorgraph.layers import SoftMax
from deepchem.models.tensorgraph.layers import SoftMaxCrossEntropy
from deepchem.models.tensorgraph.layers import SparseSoftMaxCrossEntropy
from deepchem.models.tensorgraph.layers import StopGradient
from deepchem.models.tensorgraph.layers import TensorWrapper
from deepchem.models.tensorgraph.layers import TimeSeriesDense
from deepchem.models.tensorgraph.layers import ToFloat
from deepchem.models.tensorgraph.layers import Transpose
from deepchem.models.tensorgraph.layers import Variable
from deepchem.models.tensorgraph.layers import VinaFreeEnergy
from deepchem.models.tensorgraph.layers import WeightDecay
from deepchem.models.tensorgraph.layers import WeightedError
from deepchem.models.tensorgraph.layers import WeightedLinearCombo
from deepchem.models.tensorgraph.IRV import IRVLayer
from deepchem.models.tensorgraph.IRV import IRVRegularize
from deepchem.models.tensorgraph.IRV import Slice
from deepchem.models.tensorgraph.graph_layers import DTNNEmbedding
from deepchem.models.tensorgraph.graph_layers import DTNNExtract
from deepchem.models.tensorgraph.graph_layers import WeaveGather
from deepchem.models.tensorgraph.graph_layers import DTNNGather
from deepchem.models.tensorgraph.graph_layers import DAGGather
from deepchem.models.layers import GatedRecurrentUnit
from deepchem.models.layers import EdgeNetwork


class TestLayers(test_util.TensorFlowTestCase):
  """
  Test that layers function as intended.
  """

  def test_conv_1D(self):
    """Test that Conv1D can be invoked."""
    width = 5
    in_channels = 2
    filters = 3
    kernel_size = 2
    batch_size = 10
    in_tensor = np.random.rand(batch_size, width, in_channels)
    with self.session() as sess:
      in_tensor = tf.convert_to_tensor(in_tensor, dtype=tf.float32)
      out_tensor = Conv1D(filters, kernel_size)(in_tensor)
      sess.run(tf.global_variables_initializer())
      out_tensor = out_tensor.eval()

      self.assertEqual(out_tensor.shape[0], batch_size)
      self.assertEqual(out_tensor.shape[2], filters)

  def test_dense(self):
    """Test that Dense can be invoked."""
    in_dim = 2
    out_dim = 3
    batch_size = 10
    in_tensor = np.random.rand(batch_size, in_dim)
    with self.session() as sess:
      in_tensor = tf.convert_to_tensor(in_tensor, dtype=tf.float32)
      out_tensor = Dense(out_dim)(in_tensor)
      sess.run(tf.global_variables_initializer())
      out_tensor = out_tensor.eval()
      assert out_tensor.shape == (batch_size, out_dim)

  def test_flatten(self):
    """Test that Flatten can be invoked."""
    in_dim_1 = 2
    in_dim_2 = 2
    out_dim = 4
    batch_size = 10
    in_tensor = np.random.rand(batch_size, in_dim_1, in_dim_2)
    with self.session() as sess:
      in_tensor = tf.convert_to_tensor(in_tensor, dtype=tf.float32)
      out_tensor = Flatten()(in_tensor)
      out_tensor = out_tensor.eval()
      assert out_tensor.shape == (batch_size, out_dim)

  def test_reshape(self):
    """Test that Reshape can be invoked."""
    in_dim_1 = 2
    in_dim_2 = 2
    out_dim = 4
    batch_size = 10
    in_tensor = np.random.rand(batch_size, in_dim_1, in_dim_2)
    with self.session() as sess:
      in_tensor = tf.convert_to_tensor(in_tensor, dtype=tf.float32)
      out_tensor = Reshape((batch_size, out_dim))(in_tensor)
      out_tensor = out_tensor.eval()
      assert out_tensor.shape == (batch_size, out_dim)

  def test_transpose(self):
    """Test that Transpose can be invoked."""
    in_dim_1 = 2
    in_dim_2 = 7
    batch_size = 10
    in_tensor = np.random.rand(batch_size, in_dim_1, in_dim_2)
    with self.session() as sess:
      in_tensor = tf.convert_to_tensor(in_tensor, dtype=tf.float32)
      out_tensor = Transpose((0, 2, 1))(in_tensor)
      out_tensor = out_tensor.eval()
      assert out_tensor.shape == (batch_size, in_dim_2, in_dim_1)

  def test_combine_mean_std(self):
    """Test that Transpose can be invoked."""
    dim = 2
    batch_size = 10
    mean_tensor = np.random.rand(dim)
    std_tensor = np.random.rand(1,)
    with self.session() as sess:
      mean_tensor = tf.convert_to_tensor(mean_tensor, dtype=tf.float32)
      std_tensor = tf.convert_to_tensor(std_tensor, dtype=tf.float32)
      out_tensor = CombineMeanStd()(mean_tensor, std_tensor)
      out_tensor = out_tensor.eval()
      assert out_tensor.shape == (dim,)

  def test_repeat(self):
    """Test that Repeat can be invoked."""
    in_dim = 4
    batch_size = 10
    n_repeat = 2
    in_tensor = np.random.rand(batch_size, in_dim)
    with self.session() as sess:
      in_tensor = tf.convert_to_tensor(in_tensor, dtype=tf.float32)
      out_tensor = Repeat(n_repeat)(in_tensor)
      out_tensor = out_tensor.eval()
      assert out_tensor.shape == (batch_size, n_repeat, in_dim)

  def test_gather(self):
    """Test that Gather can be invoked."""
    in_tensor = np.random.uniform(size=(5, 4)).astype(np.float32)
    with self.session() as sess:
      out_tensor = Gather(indices=[[2], [3]])(in_tensor).eval()
      assert np.array_equal([in_tensor[2], in_tensor[3]], out_tensor)
      out_tensor = Gather()(in_tensor, np.array([[1, 1], [0, 3]])).eval()
      assert np.array_equal([in_tensor[1, 1], in_tensor[0, 3]], out_tensor)

  def test_gru(self):
    """Test that GRU can be invoked."""
    batch_size = 10
    n_hidden = 7
    in_channels = 4
    n_repeat = 2
    n_steps = 6
    in_tensor = np.random.rand(batch_size, n_steps, in_channels)
    with self.session() as sess:
      in_tensor = tf.convert_to_tensor(in_tensor, dtype=tf.float32)
      out_tensor = GRU(n_hidden, batch_size)(in_tensor)
      sess.run(tf.global_variables_initializer())
      out_tensor = out_tensor.eval()
      assert out_tensor.shape == (batch_size, n_steps, n_hidden)

  def test_lstm(self):
    """Test that LSTM can be invoked."""
    batch_size = 10
    n_hidden = 7
    in_channels = 4
    n_repeat = 2
    n_steps = 6
    in_tensor = np.random.rand(batch_size, n_steps, in_channels)
    with self.session() as sess:
      in_tensor = tf.convert_to_tensor(in_tensor, dtype=tf.float32)
      out_tensor = LSTM(n_hidden, batch_size)(in_tensor)
      sess.run(tf.global_variables_initializer())
      out_tensor = out_tensor.eval()
      assert out_tensor.shape == (batch_size, n_steps, n_hidden)

  def test_time_series_dense(self):
    """Test that TimeSeriesDense can be invoked."""
    batch_size = 10
    n_hidden = 7
    in_channels = 4
    out_channels = 5
    n_repeat = 2
    n_steps = 6
    in_tensor = np.random.rand(batch_size, n_steps, in_channels)
    with self.session() as sess:
      in_tensor = tf.convert_to_tensor(in_tensor, dtype=tf.float32)
      out_tensor = TimeSeriesDense(out_channels)(in_tensor)
      assert out_tensor.shape == (batch_size, n_steps, out_channels)

  def test_input(self):
    """Test that Input can be invoked."""
    in_shape = (4, 3)
    with self.session() as sess:
      out_tensor = Input(in_shape)()

  def test_l2_loss(self):
    """Test that L2Loss can be invoked."""
    batch_size = 10
    n_features = 5
    guess_tensor = np.random.rand(batch_size, n_features)
    label_tensor = np.random.rand(batch_size, n_features)
    with self.session() as sess:
      guess_tensor = tf.convert_to_tensor(guess_tensor, dtype=tf.float32)
      label_tensor = tf.convert_to_tensor(label_tensor, dtype=tf.float32)
      out_tensor = L2Loss()(guess_tensor, label_tensor)
      out_tensor = out_tensor.eval()
      assert out_tensor.shape == (batch_size,)

  def test_relu(self):
    """Test that Sigmoid can be invoked."""
    batch_size = 10
    n_features = 5
    in_tensor = np.random.rand(batch_size, n_features)
    with self.session() as sess:
      in_tensor = tf.convert_to_tensor(in_tensor, dtype=tf.float32)
      out_tensor = ReLU()(in_tensor)
      out_tensor = out_tensor.eval()
      assert out_tensor.shape == (batch_size, n_features)

  def test_sigmoid(self):
    """Test that Sigmoid can be invoked."""
    batch_size = 10
    n_features = 5
    in_tensor = np.random.rand(batch_size, n_features)
    with self.session() as sess:
      in_tensor = tf.convert_to_tensor(in_tensor, dtype=tf.float32)
      out_tensor = Sigmoid()(in_tensor)
      out_tensor = out_tensor.eval()
      assert out_tensor.shape == (batch_size, n_features)

  def test_softmax(self):
    """Test that Softmax can be invoked."""
    batch_size = 10
    n_features = 5
    in_tensor = np.random.rand(batch_size, n_features)
    with self.session() as sess:
      in_tensor = tf.convert_to_tensor(in_tensor, dtype=tf.float32)
      out_tensor = SoftMax()(in_tensor)
      out_tensor = out_tensor.eval()
      assert out_tensor.shape == (batch_size, n_features)

  def test_concat(self):
    """Test that Concat can be invoked."""
    batch_size = 10
    n_features = 5
    in_tensor_1 = np.random.rand(batch_size, n_features)
    in_tensor_2 = np.random.rand(batch_size, n_features)
    with self.session() as sess:
      in_tensor_1 = tf.convert_to_tensor(in_tensor_1, dtype=tf.float32)
      in_tensor_2 = tf.convert_to_tensor(in_tensor_2, dtype=tf.float32)
      out_tensor = Concat(axis=1)(in_tensor_1, in_tensor_2)
      out_tensor = out_tensor.eval()
      assert out_tensor.shape == (batch_size, 2 * n_features)

  def test_constant(self):
    """Test that Constant can be invoked."""
    value = np.random.uniform(size=(2, 3)).astype(np.float32)
    with self.session() as sess:
      out_tensor = Constant(value)()
      assert np.array_equal(value, out_tensor.eval())

  def test_variable(self):
    """Test that Variable can be invoked."""
    value = np.random.uniform(size=(2, 3)).astype(np.float32)
    with self.session() as sess:
      out_tensor = Variable(value)()
      sess.run(tf.global_variables_initializer())
      assert np.array_equal(value, out_tensor.eval())

  def test_stop_gradient(self):
    """Test that StopGradient can be invoked."""
    batch_size = 10
    n_features = 5
    in_tensor = np.random.rand(batch_size, n_features)
    with self.session() as sess:
      in_tensor = tf.convert_to_tensor(in_tensor, dtype=tf.float32)
      out_tensor = StopGradient()(in_tensor)
      assert np.array_equal(in_tensor.eval(), out_tensor.eval())

  def test_add(self):
    """Test that Add can be invoked."""
    value1 = np.random.uniform(size=(2, 3)).astype(np.float32)
    value2 = np.random.uniform(size=(2, 3)).astype(np.float32)
    value3 = np.random.uniform(size=(2, 3)).astype(np.float32)
    with self.session() as sess:
      out_tensor = Add(weights=[1, 2, 1])(tf.constant(value1),
                                          tf.constant(value2),
                                          tf.constant(value3))
      assert np.array_equal(value1 + 2 * value2 + value3, out_tensor.eval())

  def test_multiply(self):
    """Test that Multiply can be invoked."""
    value1 = np.random.uniform(size=(2, 3)).astype(np.float32)
    value2 = np.random.uniform(size=(2, 3)).astype(np.float32)
    value3 = np.random.uniform(size=(2, 3)).astype(np.float32)
    with self.session() as sess:
      out_tensor = Multiply()(tf.constant(value1), tf.constant(value2),
                              tf.constant(value3))
      assert np.array_equal(value1 * value2 * value3, out_tensor.eval())

  def test_log(self):
    """Test that Log can be invoked."""
    value = np.random.uniform(size=(2, 3)).astype(np.float32)
    with self.session() as sess:
      result = Log()(value).eval()
      assert np.all(np.isclose(np.log(value), result, atol=0.001))

  def test_exp(self):
    """Test that Exp can be invoked."""
    value = np.random.uniform(size=(2, 3)).astype(np.float32)
    with self.session() as sess:
      result = Exp()(value).eval()
      assert np.array_equal(np.exp(value), result)

  def test_interatomic_distances(self):
    """Test that the interatomic distance calculation works."""
    N_atoms = 5
    M_nbrs = 2
    ndim = 3

    with self.session() as sess:
      coords = np.random.rand(N_atoms, ndim)
      nbr_list = np.random.randint(0, N_atoms, size=(N_atoms, M_nbrs))

      coords_tensor = tf.convert_to_tensor(coords)
      nbr_list_tensor = tf.convert_to_tensor(nbr_list)

      dist_tensor = InteratomicL2Distances(N_atoms, M_nbrs,
                                           ndim)(coords_tensor, nbr_list_tensor)

      dists = dist_tensor.eval()
      assert dists.shape == (N_atoms, M_nbrs)

  def test_sigmoid_cross_entropy(self):
    """Test that SigmoidCrossEntropy can be invoked."""
    batch_size = 10
    n_features = 5
    logit_tensor = np.random.rand(batch_size, n_features)
    label_tensor = np.random.randint(0, 2, (batch_size, n_features))
    with self.session() as sess:
      logit_tensor = tf.convert_to_tensor(logit_tensor, dtype=tf.float32)
      label_tensor = tf.convert_to_tensor(label_tensor, dtype=tf.float32)
      out_tensor = SigmoidCrossEntropy()(label_tensor, logit_tensor)
      out_tensor = out_tensor.eval()
      assert out_tensor.shape == (batch_size, n_features)

  def test_softmax_cross_entropy(self):
    """Test that SoftMaxCrossEntropy can be invoked."""
    batch_size = 10
    n_features = 5
    logit_tensor = np.random.rand(batch_size, n_features)
    label_tensor = np.random.rand(batch_size, n_features)
    with self.session() as sess:
      logit_tensor = tf.convert_to_tensor(logit_tensor, dtype=tf.float32)
      label_tensor = tf.convert_to_tensor(label_tensor, dtype=tf.float32)
      out_tensor = SoftMaxCrossEntropy()(logit_tensor, label_tensor)
      out_tensor = out_tensor.eval()
      assert out_tensor.shape == (batch_size,)

  def test_sparse_softmax_cross_entropy(self):
    batch_size = 10
    n_features = 5
    logit_tensor = np.random.rand(batch_size, n_features)
    label_tensor = np.random.rand(batch_size)
    with self.session() as sess:
      logit_tensor = tf.convert_to_tensor(logit_tensor, dtype=tf.float32)
      label_tensor = tf.convert_to_tensor(label_tensor, dtype=tf.int32)
      out_tensor = SparseSoftMaxCrossEntropy()(label_tensor, logit_tensor)
      out_tensor = out_tensor.eval()
      assert out_tensor.shape == (batch_size,)

  def test_reduce_mean(self):
    """Test that ReduceMean can be invoked."""
    batch_size = 10
    n_features = 5
    in_tensor = np.random.rand(batch_size, n_features)
    with self.session() as sess:
      in_tensor = tf.convert_to_tensor(in_tensor, dtype=tf.float32)
      out_tensor = ReduceMean()(in_tensor)
      out_tensor = out_tensor.eval()
      assert isinstance(out_tensor, np.float32)

  def test_to_float(self):
    """Test that ToFloat can be invoked."""
    batch_size = 10
    n_features = 5
    in_tensor = np.random.rand(batch_size, n_features)
    with self.session() as sess:
      in_tensor = tf.convert_to_tensor(in_tensor, dtype=tf.float32)
      out_tensor = ToFloat()(in_tensor)
      out_tensor = out_tensor.eval()
      assert out_tensor.shape == (batch_size, n_features)

  def test_reduce_sum(self):
    """Test that ReduceSum can be invoked."""
    batch_size = 10
    n_features = 5
    in_tensor = np.random.rand(batch_size, n_features)
    with self.session() as sess:
      in_tensor = tf.convert_to_tensor(in_tensor, dtype=tf.float32)
      out_tensor = ReduceSum()(in_tensor)
      out_tensor = out_tensor.eval()
      assert isinstance(out_tensor, np.float32)

  def test_reduce_square_difference(self):
    """Test that ReduceSquareDifference can be invoked."""
    batch_size = 10
    n_features = 5
    in_tensor_1 = np.random.rand(batch_size, n_features)
    in_tensor_2 = np.random.rand(batch_size, n_features)
    with self.session() as sess:
      in_tensor_1 = tf.convert_to_tensor(in_tensor_1, dtype=tf.float32)
      in_tensor_2 = tf.convert_to_tensor(in_tensor_2, dtype=tf.float32)
      out_tensor = ReduceSquareDifference()(in_tensor_1, in_tensor_2)
      out_tensor = out_tensor.eval()
      assert isinstance(out_tensor, np.float32)

  def test_conv_2D(self):
    """Test that Conv2D can be invoked."""
    length = 4
    width = 5
    in_channels = 2
    out_channels = 3
    batch_size = 20
    in_tensor = np.random.rand(batch_size, length, width, in_channels)
    with self.session() as sess:
      in_tensor = tf.convert_to_tensor(in_tensor, dtype=tf.float32)
      out_tensor = Conv2D(out_channels, kernel_size=1)(in_tensor)
      sess.run(tf.global_variables_initializer())
      out_tensor = out_tensor.eval()
      assert out_tensor.shape == (batch_size, length, width, out_channels)

  def test_conv_3D(self):
    """Test that Conv3D can be invoked."""
    length = 4
    width = 5
    depth = 6
    in_channels = 2
    out_channels = 3
    batch_size = 20
    in_tensor = np.random.rand(batch_size, length, width, depth, in_channels)
    with self.session() as sess:
      in_tensor = tf.convert_to_tensor(in_tensor, dtype=tf.float32)
      out_tensor = Conv3D(out_channels, kernel_size=1)(in_tensor)
      sess.run(tf.global_variables_initializer())
      out_tensor = out_tensor.eval()
      assert out_tensor.shape == (batch_size, length, width, depth,
                                  out_channels)

  def test_conv_2D_transpose(self):
    """Test that Conv2DTranspose can be invoked."""
    length = 4
    width = 5
    in_channels = 2
    out_channels = 3
    batch_size = 20
    in_tensor = np.random.rand(batch_size, length, width, in_channels)
    with self.session() as sess:
      in_tensor = tf.convert_to_tensor(in_tensor, dtype=tf.float32)
      out_tensor = Conv2DTranspose(
          out_channels, kernel_size=1, stride=2)(in_tensor)
      sess.run(tf.global_variables_initializer())
      out_tensor = out_tensor.eval()
      assert out_tensor.shape == (batch_size, 2 * length, 2 * width,
                                  out_channels)

  def test_conv_3D_transpose(self):
    """Test that Conv3DTranspose can be invoked."""
    length = 4
    width = 5
    depth = 6
    in_channels = 2
    out_channels = 3
    batch_size = 20
    in_tensor = np.random.rand(batch_size, length, width, depth, in_channels)
    with self.session() as sess:
      in_tensor = tf.convert_to_tensor(in_tensor, dtype=tf.float32)
      out_tensor = Conv3DTranspose(
          out_channels, kernel_size=1, stride=(2, 3, 1))(in_tensor)
      sess.run(tf.global_variables_initializer())
      out_tensor = out_tensor.eval()
      assert out_tensor.shape == (batch_size, 2 * length, 3 * width, depth,
                                  out_channels)

  def test_maxpool2D(self):
    """Test that MaxPool2D can be invoked."""
    length = 2
    width = 2
    in_channels = 2
    batch_size = 20
    in_tensor = np.random.rand(batch_size, length, width, in_channels)
    with self.session() as sess:
      in_tensor = tf.convert_to_tensor(in_tensor, dtype=tf.float32)
      out_tensor = MaxPool2D()(in_tensor)
      sess.run(tf.global_variables_initializer())
      out_tensor = out_tensor.eval()
      assert out_tensor.shape == (batch_size, 1, 1, in_channels)

  def test_max_pool_3D(self):
    """Test that MaxPool3D can be invoked."""
    length = 2
    width = 2
    depth = 2
    in_channels = 2
    batch_size = 20
    in_tensor = np.random.rand(batch_size, length, width, depth, in_channels)
    with self.session() as sess:
      in_tensor = tf.convert_to_tensor(in_tensor, dtype=tf.float32)
      out_tensor = MaxPool3D()(in_tensor)
      sess.run(tf.global_variables_initializer())
      out_tensor = out_tensor.eval()
      assert out_tensor.shape == (batch_size, 1, 1, 1, in_channels)

  def test_avgpool2D(self):
    """Test that AvgPool2D can be invoked."""
    length = 4
    width = 4
    in_channels = 5
    batch_size = 10
    in_tensor = np.random.rand(batch_size, length, width, in_channels)
    with self.session() as sess:
      in_tensor = tf.convert_to_tensor(in_tensor, dtype=tf.float32)
      out_tensor = AvgPool2D()(in_tensor)
      sess.run(tf.global_variables_initializer())
      out_tensor = out_tensor.eval()
      assert out_tensor.shape == (batch_size, 2, 2, in_channels)

  def test_input_fifo_queue(self):
    """Test InputFifoQueue can be invoked."""
    batch_size = 10
    n_features = 5
    in_tensor = np.random.rand(batch_size, n_features)
    tf.reset_default_graph()
    with self.session() as sess:
      in_tensor = TensorWrapper(
          tf.convert_to_tensor(in_tensor, dtype=tf.float32), name="input")
      InputFifoQueue([(batch_size, n_features)], ["input"])(in_tensor)

  def test_graph_conv(self):
    """Test that GraphConv can be invoked."""
    out_channels = 2
    n_atoms = 4  # In CCC and C, there are 4 atoms
    raw_smiles = ['CCC', 'C']
    mols = [rdkit.Chem.MolFromSmiles(s) for s in raw_smiles]
    featurizer = ConvMolFeaturizer()
    mols = featurizer.featurize(mols)
    multi_mol = ConvMol.agglomerate_mols(mols)
    atom_features = multi_mol.get_atom_features()
    degree_slice = multi_mol.deg_slice
    membership = multi_mol.membership
    deg_adjs = multi_mol.get_deg_adjacency_lists()[1:]

    with self.session() as sess:
      atom_features = tf.convert_to_tensor(atom_features, dtype=tf.float32)
      degree_slice = tf.convert_to_tensor(degree_slice, dtype=tf.int32)
      membership = tf.convert_to_tensor(membership, dtype=tf.int32)
      deg_adjs_tf = []
      for deg_adj in deg_adjs:
        deg_adjs_tf.append(tf.convert_to_tensor(deg_adj, dtype=tf.int32))
      args = [atom_features, degree_slice, membership] + deg_adjs_tf
      out_tensor = GraphConv(out_channels)(*args)
      sess.run(tf.global_variables_initializer())
      out_tensor = out_tensor.eval()
      assert out_tensor.shape == (n_atoms, out_channels)

  def test_lstm_step(self):
    """Test that LSTMStep computation works properly."""
    max_depth = 5
    n_test = 5
    n_feat = 10

    y = np.random.rand(n_test, 2 * n_feat)
    state_zero = np.random.rand(n_test, n_feat)
    state_one = np.random.rand(n_test, n_feat)
    with self.session() as sess:
      y = tf.convert_to_tensor(y, dtype=tf.float32)
      state_zero = tf.convert_to_tensor(state_zero, dtype=tf.float32)
      state_one = tf.convert_to_tensor(state_one, dtype=tf.float32)

      lstm = LSTMStep(n_feat, 2 * n_feat)
      out_tensor = lstm(y, state_zero, state_one)
      sess.run(tf.global_variables_initializer())
      h_out, h_copy_out, c_out = (out_tensor[0].eval(), out_tensor[1][0].eval(),
                                  out_tensor[1][1].eval())
      assert h_out.shape == (n_test, n_feat)
      assert h_copy_out.shape == (n_test, n_feat)
      assert c_out.shape == (n_test, n_feat)

  def test_attn_lstm_embedding(self):
    """Test that attention LSTM computation works properly."""
    max_depth = 5
    n_test = 5
    n_support = 11
    n_feat = 10

    test = np.random.rand(n_test, n_feat)
    support = np.random.rand(n_support, n_feat)
    with self.session() as sess:
      test = tf.convert_to_tensor(test, dtype=tf.float32)
      support = tf.convert_to_tensor(support, dtype=tf.float32)

      attn_embedding_layer = AttnLSTMEmbedding(n_test, n_support, n_feat,
                                               max_depth)
      out_tensor = attn_embedding_layer(test, support)
      sess.run(tf.global_variables_initializer())
      test_out, support_out = out_tensor[0].eval(), out_tensor[1].eval()
      assert test_out.shape == (n_test, n_feat)
      assert support_out.shape == (n_support, n_feat)

  def test_iter_ref_lstm_embedding(self):
    """Test that IterRef LSTM computation works properly."""
    max_depth = 5
    n_test = 5
    n_support = 11
    n_feat = 10

    test = np.random.rand(n_test, n_feat)
    support = np.random.rand(n_support, n_feat)
    with self.session() as sess:
      test = tf.convert_to_tensor(test, dtype=tf.float32)
      support = tf.convert_to_tensor(support, dtype=tf.float32)

      iter_ref_embedding_layer = IterRefLSTMEmbedding(n_test, n_support, n_feat,
                                                      max_depth)
      out_tensor = iter_ref_embedding_layer(test, support)
      sess.run(tf.global_variables_initializer())
      test_out, support_out = out_tensor[0].eval(), out_tensor[1].eval()
      assert test_out.shape == (n_test, n_feat)
      assert support_out.shape == (n_support, n_feat)

  # TODO(rbharath): This test should pass. Fix it!
  # def test_graph_pool(self):
  #  """Test that GraphPool can be invoked."""
  #  out_channels = 2
  #  n_atoms = 4 # In CCC and C, there are 4 atoms
  #  raw_smiles = ['CCC', 'C']
  #  mols = [rdkit.Chem.MolFromSmiles(s) for s in raw_smiles]
  #  featurizer = ConvMolFeaturizer()
  #  mols = featurizer.featurize(mols)
  #  multi_mol = ConvMol.agglomerate_mols(mols)
  #  atom_features = multi_mol.get_atom_features()
  #  degree_slice = multi_mol.deg_slice
  #  membership = multi_mol.membership
  #  deg_adjs = multi_mol.get_deg_adjacency_lists()[1:]

  #  with self.session() as sess:
  #    atom_features = tf.convert_to_tensor(atom_features, dtype=tf.float32)
  #    degree_slice = tf.convert_to_tensor(degree_slice, dtype=tf.int32)
  #    membership = tf.convert_to_tensor(membership, dtype=tf.int32)
  #    deg_adjs_tf = []
  #    for deg_adj in deg_adjs:
  #      deg_adjs_tf.append(tf.convert_to_tensor(deg_adj, dtype=tf.int32))
  #    args = [atom_features, degree_slice, membership] + deg_adjs_tf
  #    out_tensor = GraphPool(out_channels)(*args)
  #    sess.run(tf.global_variables_initializer())
  #    out_tensor = out_tensor.eval()
  #    assert out_tensor.shape == (n_atoms, out_channels)

  def test_graph_gather(self):
    """Test that GraphGather can be invoked."""
    batch_size = 2
    n_features = 75
    n_atoms = 4  # In CCC and C, there are 4 atoms
    raw_smiles = ['CCC', 'C']
    mols = [rdkit.Chem.MolFromSmiles(s) for s in raw_smiles]
    featurizer = ConvMolFeaturizer()
    mols = featurizer.featurize(mols)
    multi_mol = ConvMol.agglomerate_mols(mols)
    atom_features = multi_mol.get_atom_features()
    degree_slice = multi_mol.deg_slice
    membership = multi_mol.membership
    deg_adjs = multi_mol.get_deg_adjacency_lists()[1:]

    with self.session() as sess:
      atom_features = tf.convert_to_tensor(atom_features, dtype=tf.float32)
      degree_slice = tf.convert_to_tensor(degree_slice, dtype=tf.int32)
      membership = tf.convert_to_tensor(membership, dtype=tf.int32)
      deg_adjs_tf = []
      for deg_adj in deg_adjs:
        deg_adjs_tf.append(tf.convert_to_tensor(deg_adj, dtype=tf.int32))
      args = [atom_features, degree_slice, membership] + deg_adjs_tf
      out_tensor = GraphGather(batch_size)(*args)
      sess.run(tf.global_variables_initializer())
      out_tensor = out_tensor.eval()
      # TODO(rbharath): Why is it 2*n_features instead of n_features?
      assert out_tensor.shape == (batch_size, 2 * n_features)

  def test_batch_norm(self):
    """Test that BatchNorm can be invoked."""
    batch_size = 10
    n_features = 5
    in_tensor = np.random.rand(batch_size, n_features)
    with self.session() as sess:
      in_tensor = tf.convert_to_tensor(in_tensor, dtype=tf.float32)
      out_tensor = BatchNorm()(in_tensor)
      sess.run(tf.global_variables_initializer())
      out_tensor = out_tensor.eval()
      assert out_tensor.shape == (batch_size, n_features)

  def test_weighted_error(self):
    """Test that WeightedError can be invoked."""
    batch_size = 10
    n_features = 5
    guess_tensor = np.random.rand(batch_size, n_features)
    label_tensor = np.random.rand(batch_size, n_features)
    with self.session() as sess:
      guess_tensor = tf.convert_to_tensor(guess_tensor, dtype=tf.float32)
      label_tensor = tf.convert_to_tensor(label_tensor, dtype=tf.float32)
      out_tensor = WeightedError()(guess_tensor, label_tensor)
      out_tensor = out_tensor.eval()
      assert isinstance(out_tensor, np.float32)

  def test_vina_free_energy(self):
    """Test that VinaFreeEnergy can be invoked."""
    n_atoms = 5
    m_nbrs = 1
    ndim = 3
    nbr_cutoff = 1
    start = 0
    stop = 4
    X_tensor = np.random.rand(n_atoms, ndim)
    Z_tensor = np.random.randint(0, 2, (n_atoms))
    with self.session() as sess:
      X_tensor = tf.convert_to_tensor(X_tensor, dtype=tf.float32)
      Z_tensor = tf.convert_to_tensor(Z_tensor, dtype=tf.float32)
      out_tensor = VinaFreeEnergy(n_atoms, m_nbrs, ndim, nbr_cutoff, start,
                                  stop)(X_tensor, Z_tensor)
      sess.run(tf.global_variables_initializer())
      out_tensor = out_tensor.eval()
      assert isinstance(out_tensor, np.float32)

  def test_weighted_linear_combo(self):
    """Test that WeightedLinearCombo can be invoked."""
    batch_size = 10
    n_features = 5
    in_tensor_1 = np.random.rand(batch_size, n_features)
    in_tensor_2 = np.random.rand(batch_size, n_features)
    with self.session() as sess:
      in_tensor_1 = tf.convert_to_tensor(in_tensor_1, dtype=tf.float32)
      in_tensor_2 = tf.convert_to_tensor(in_tensor_2, dtype=tf.float32)
      out_tensor = WeightedLinearCombo()(in_tensor_1, in_tensor_2)
      sess.run(tf.global_variables_initializer())
      out_tensor = out_tensor.eval()
      assert out_tensor.shape == (batch_size, n_features)

  def test_reshape_inputs(self):
    """Test that layers can automatically reshape inconsistent inputs."""
    value1 = np.random.uniform(size=(2, 3)).astype(np.float32)
    value2 = np.random.uniform(size=(1, 6, 1)).astype(np.float32)
    with self.session() as sess:
      out_tensor = ReduceSquareDifference()(tf.constant(value1),
                                            tf.constant(value2))
      result = out_tensor.eval()
      diff = value1.reshape((1, 6, 1)) - value2
      loss = np.mean(diff**2)
      assert (loss - result) / loss < 1e-6

  def test_cast(self):
    """Test that layers can automatically reshape inconsistent inputs."""
    value1 = np.random.uniform(size=(2, 1)).astype(np.float32)
    with self.session() as sess:
      out_tensor = Cast(dtype=tf.int32)(tf.constant(value1))
      result = out_tensor.eval()
      assert result.dtype == np.int32

  def test_squeeze_inputs(self):
    """Test that layers can automatically reshape inconsistent inputs."""
    value1 = np.random.uniform(size=(2, 1)).astype(np.float32)
    with self.session() as sess:
      out_tensor = Squeeze(squeeze_dims=1)(tf.constant(value1))
      result = out_tensor.eval()
      assert result.shape == (2,)

  def test_convert_to_tensor(self):
    """Test implicit conversion of Layers to Tensors."""
    v = Variable(np.array([1.5]))
    v.create_tensor()
    with self.session() as sess:
      sess.run(tf.global_variables_initializer())
      result = sess.run(v)
      assert result == 1.5
      result = sess.run(tf.gradients(v, v))
      assert result[0] == 1.0

  def test_alpha_share_layer(self):
    """Test that alpha share works correctly"""
    batch_size = 50
    length = 10
    test_1 = np.random.rand(batch_size, length)
    test_2 = np.random.rand(batch_size, length)

    with self.session() as sess:
      test_1 = tf.convert_to_tensor(test_1, dtype=tf.float32)
      test_2 = tf.convert_to_tensor(test_2, dtype=tf.float32)

      out_tensor = AlphaShareLayer()(test_1, test_2)
      sess.run(tf.global_variables_initializer())
      test_1_out_tensor = out_tensor[0].eval()
      test_2_out_tensor = out_tensor[1].eval()
      assert test_1.shape == test_1_out_tensor.shape
      assert test_2.shape == test_2_out_tensor.shape

  def test_beta_share(self):
    """Test that beta share works correctly"""
    batch_size = 50
    length = 10
    test_1 = np.random.rand(batch_size, length)
    test_2 = np.random.rand(batch_size, length)

    with self.session() as sess:
      test_1 = tf.convert_to_tensor(test_1, dtype=tf.float32)
      test_2 = tf.convert_to_tensor(test_2, dtype=tf.float32)

      out_tensor = BetaShare()(test_1, test_2)
      sess.run(tf.global_variables_initializer())
      out_tensor.eval()
      assert test_1.shape == out_tensor.shape
      assert test_2.shape == out_tensor.shape

  def test_sluice_loss(self):
    """Test the sluice loss function"""
    input1 = np.ones((3, 4))
    input2 = np.ones((2, 2))
    with self.session() as sess:
      input1 = tf.convert_to_tensor(input1, dtype=tf.float32)
      input2 = tf.convert_to_tensor(input2, dtype=tf.float32)
      output_tensor = SluiceLoss()(input1, input2)
      sess.run(tf.global_variables_initializer())
      assert output_tensor.eval() == 40.0

  def test_graphcnn(self):
    """ Test GraphCNN Layer From https://arxiv.org/abs/1703.00792"""
    V = np.random.uniform(size=(10, 100, 50)).astype(np.float32)
    adjs = np.random.uniform(size=(10, 100, 5, 100)).astype(np.float32)
    with self.session() as sess:
      out_tensor = GraphCNN(num_filters=6)(V, adjs)
      sess.run(tf.global_variables_initializer())
      result = out_tensor.eval()
      assert result.shape == (10, 100, 6)

  def test_graphcnnpool(self):
    """ Test GraphCNNPool Layer From https://arxiv.org/abs/1703.00792"""
    V = np.random.uniform(size=(10, 100, 50)).astype(np.float32)
    adjs = np.random.uniform(size=(10, 100, 5, 100)).astype(np.float32)
    with self.session() as sess:
      vertex_props, adjs = GraphEmbedPoolLayer(num_vertices=6)(V, adjs)
      sess.run(tf.global_variables_initializer())
      vertex_props, adjs = vertex_props.eval(), adjs.eval()
      assert vertex_props.shape == (10, 6, 50)
      assert adjs.shape == (10, 6, 5, 6)

  def test_slice(self):
    """Test that Slice can be invoked."""
    batch_size = 10
    n_features = 5
    test_tensor_input = np.random.rand(batch_size, n_features)
    with self.session() as sess:
      test_tensor = tf.convert_to_tensor(test_tensor_input, dtype=tf.float32)
      out_tensor = Slice(1)(test_tensor)
      out_tensor = out_tensor.eval()
      assert np.allclose(out_tensor, test_tensor_input[:, 1:2])

  def test_IRV(self):
    """Test that IRVLayer and IRVRegularize can be invoked."""
    batch_size = 10
    n_tasks = 5
    K = 10
    n_features = 2 * K * n_tasks
    test_tensor_input = np.random.rand(batch_size, n_features)
    with self.session() as sess:
      test_tensor = tf.convert_to_tensor(test_tensor_input, dtype=tf.float32)
      irv_layer = IRVLayer(n_tasks, K)
      irv_layer.create_tensor(in_layers=[test_tensor])
      out_tensor = irv_layer.out_tensor
      sess.run(tf.global_variables_initializer())
      out_tensor = out_tensor.eval()
      assert out_tensor.shape == (batch_size, n_tasks)
      irv_reg = IRVRegularize(irv_layer, 1.)()
      assert irv_reg.eval() >= 0

  def test_hingeloss(self):
    separation = 0.25
    labels = [1, 1, 0, 0]
    logits = [.3, .1, -0.3, -0.1]
    losses = np.array([0, 0.15, 0, 0.15], dtype=np.float32)

    with self.session() as sess:
      logits_tensor = tf.convert_to_tensor(logits, dtype=tf.float32)
      labels_tensor = tf.convert_to_tensor(labels, dtype=tf.float32)
      out_tensor = HingeLoss(separation=separation)(labels_tensor,
                                                    logits_tensor)
      out_tensor = out_tensor.eval()
      retval = np.all(losses == np.array(out_tensor))
      self.assertTrue(retval)

  def test_weight_decay(self):
    """Test that WeightDecay can be invoked."""
    values = np.random.rand(5, 5).astype(np.float32)
    variable = tf.Variable(values)
    with self.session() as sess:
      sess.run(tf.global_variables_initializer())
      cost = WeightDecay(3.0, 'l2')(0.0)
      assert np.allclose(3.0 * np.sum(values * values) / 2, cost.eval())

  def test_dtnn_embedding(self):
    """Test that DTNNEmbedding can be invoked."""
    n_embedding = 10
    periodic_table_length = 20
    test_tensor_input = np.random.permutation(
        np.arange(0, periodic_table_length // 2, dtype=np.int32))
    with self.session() as sess:
      test_tensor = tf.convert_to_tensor(test_tensor_input, dtype=tf.int32)
      dtnn_embedding = DTNNEmbedding(
          n_embedding=n_embedding, periodic_table_length=periodic_table_length)
      dtnn_embedding.create_tensor(in_layers=[test_tensor])

      # Layer is wrapper around embedding lookup, tested that then
      sess.run(tf.global_variables_initializer())
      out_tensor = dtnn_embedding.out_tensor.eval()
      embedding_val = dtnn_embedding.trainable_variables[0].eval()
      expected_output = embedding_val[test_tensor_input]
      self.assertAllClose(out_tensor, expected_output)
      self.assertAllClose(out_tensor.shape,
                          (periodic_table_length // 2, n_embedding))

  def test_dtnn_extract(self):
    """Test that DTNNExtract can be invoked."""
    num_samples = 20
    num_features = 30
    task_id = 15
    test_tensor_input = np.random.randn(num_samples, num_features)
    test_output = test_tensor_input[:, task_id:task_id + 1]
    with self.session() as sess:
      test_tensor = tf.convert_to_tensor(test_tensor_input)
      dtnn_extract = DTNNExtract(task_id=task_id)
      dtnn_extract.create_tensor(in_layers=[test_tensor])
      sess.run(tf.global_variables_initializer())
      out_tensor = dtnn_extract.out_tensor.eval()
      self.assertAllClose(test_output, out_tensor)
      self.assertEqual(out_tensor.shape, (num_samples, 1))

  def test_weave_gather(self):
    """Test that WeaveGather can be invoked."""
    batch_size = 1
    num_samples = 2
    num_atoms_per_sample = 5
    num_features = 5
    gaussian_expand = False

    atom_split_np = list()
    for i in range(num_samples):
      atom_split_np.extend([i] * num_atoms_per_sample)
    atom_split_np = np.array(atom_split_np)
    tensor_input_np = np.random.randn(num_samples * num_atoms_per_sample,
                                      num_features)

    # Expected output
    expected_output = list()
    for i in range(num_samples):
      expected_output.append(
          np.sum(
              tensor_input_np[i * num_atoms_per_sample:(i + 1) *
                              num_atoms_per_sample],
              axis=0))
    expected_output = np.array(expected_output)

    with self.session() as sess:
      tensor_input_tf = tf.convert_to_tensor(tensor_input_np)
      atom_split_tf = tf.convert_to_tensor(atom_split_np)
      weave_gather = WeaveGather(
          batch_size=batch_size, gaussian_expand=gaussian_expand)
      weave_gather.create_tensor(in_layers=[tensor_input_tf, atom_split_tf])

      sess.run(tf.global_variables_initializer())
      out_tensor = weave_gather.out_tensor.eval()

      self.assertAllClose(expected_output, out_tensor)
      self.assertEqual(expected_output.shape, out_tensor.shape)

  def test_gated_recurrent_unit(self):
    """Test that Gated Recurrent Unit can be invoked."""
    n_hidden = 2
    init_method = 'one'

    messages_np = np.array([[1, -1]])
    inputs_np = np.array([[1, -1]])
    expected_output = np.array([[0.5, -0.5]])

    with self.session() as sess:
      messages_tf = tf.convert_to_tensor(messages_np, dtype=tf.float32)
      inputs_tf = tf.convert_to_tensor(inputs_np, dtype=tf.float32)
      gru = GatedRecurrentUnit(n_hidden=n_hidden, init=init_method)
      h = gru([inputs_tf, messages_tf])

      sess.run(tf.global_variables_initializer())
      h = h.eval()
      self.assertAllClose(h, expected_output)
      self.assertEqual(h.shape, expected_output.shape)

  def test_edge_network(self):
    """Test that Edge Network can be invoked."""
    n_pair_features = 2
    n_hidden = 2
    init_method = 'one'

    pair_features_np = np.array([[0.25, 0.25], [0.25, 0.25], [0.25, 0.25]])
    atom_features_np = np.array([[1, 1], [2, 2], [3, 3]])
    atom_to_pair_np = np.array([[0, 0], [0, 1], [1, 1]])
    expected_output = np.array([[3, 3], [2, 2]])

    with self.session() as sess:
      pair_features_tf = tf.convert_to_tensor(
          pair_features_np, dtype=tf.float32)
      atom_features_tf = tf.convert_to_tensor(
          atom_features_np, dtype=tf.float32)
      atom_to_pair_tf = tf.convert_to_tensor(atom_to_pair_np, dtype=tf.int32)
      edge_network = EdgeNetwork(
          n_pair_features=n_pair_features, n_hidden=n_hidden, init=init_method)
      edge_network.build([])
      sess.run(tf.global_variables_initializer())
      output = edge_network(
          [pair_features_tf, atom_features_tf, atom_to_pair_tf]).eval()
      self.assertAllClose(output, expected_output)
      self.assertEqual(output.shape, expected_output.shape)

  def test_dtnn_gather(self):
    """Test that DTNNGather can be invoked."""
    n_embedding = 2
    n_outputs = 2
    layer_sizes = [2]
    output_activation = False
    init_method = 'one'
    activation = 'sigmoid'

    inputs_np = np.array([[1, -1], [1, -1], [1, -1]])
    atom_membership_np = np.array([0, 0, 1])
    expected_output = np.array([[2, 2], [1, 1]])

    with self.session() as sess:
      inputs_tf = tf.convert_to_tensor(inputs_np, dtype=tf.float32)
      atom_membership_tf = tf.convert_to_tensor(
          atom_membership_np, dtype=tf.int32)
      dtnn_gather = DTNNGather(
          n_embedding=n_embedding,
          n_outputs=n_outputs,
          layer_sizes=layer_sizes,
          output_activation=output_activation,
          init=init_method,
          activation=activation)
      dtnn_gather.create_tensor(in_layers=[inputs_tf, atom_membership_tf])

      sess.run(tf.global_variables_initializer())
      output = dtnn_gather.out_tensor.eval()
      self.assertAllClose(expected_output, output)
      self.assertEqual(expected_output.shape, output.shape)

  def test_dag_gather(self):
    """Test that DAGGather can be invoked."""
    n_graph_feat = 2
    n_outputs = 2
    layer_sizes = [2]
    init_method = 'one'
    activation = 'sigmoid'

    def sigmoid(x):
      return 1 / (1 + np.exp(-x))

    atom_features_np = np.array([[1, -1], [1, -1], [1, -1]])
    membership_np = np.array([0, 0, 1])
    expected_output = sigmoid(np.array([[1, 1], [1, 1]]))

    with self.session() as sess:
      atom_features_tf = tf.convert_to_tensor(
          atom_features_np, dtype=tf.float32)
      membership_tf = tf.convert_to_tensor(membership_np, dtype=tf.int32)
      dag_gather = DAGGather(
          n_graph_feat=n_graph_feat,
          n_outputs=n_outputs,
          activation=activation,
          init=init_method,
          layer_sizes=layer_sizes)
      dag_gather.create_tensor(in_layers=[atom_features_tf, membership_tf])

      sess.run(tf.global_variables_initializer())
      output = dag_gather.out_tensor.eval()
      self.assertAllClose(output, expected_output)
      self.assertEqual(output.shape, expected_output.shape)
