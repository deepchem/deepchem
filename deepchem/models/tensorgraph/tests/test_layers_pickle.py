import numpy as np
import tensorflow as tf

from deepchem.models import TensorGraph
from deepchem.models.tensorgraph.graph_layers import WeaveLayer, WeaveGather, DTNNEmbedding, DTNNGather, DTNNStep, \
  DTNNExtract, DAGLayer, DAGGather, MessagePassing, SetGather
from deepchem.models.tensorgraph.layers import Feature, Conv1D, Dense, Flatten, Reshape, Squeeze, Transpose, \
  CombineMeanStd, Repeat, Gather, GRU, L2Loss, Concat, SoftMax, \
  Constant, Variable, StopGradient, Add, Multiply, Log, Exp, InteratomicL2Distances, \
  SoftMaxCrossEntropy, ReduceMean, ToFloat, ReduceSquareDifference, Conv2D, MaxPool2D, ReduceSum, GraphConv, GraphPool, \
  GraphGather, BatchNorm, WeightedError, \
  Conv3D, MaxPool3D, Conv2DTranspose, Conv3DTranspose, \
  LSTMStep, AttnLSTMEmbedding, IterRefLSTMEmbedding, GraphEmbedPoolLayer, GraphCNN
from deepchem.models.tensorgraph.symmetry_functions import AtomicDifferentiatedDense


def test_Conv1D_pickle():
  tg = TensorGraph()
  feature = Feature(shape=(tg.batch_size, 1, 1))
  conv = Conv1D(2, 1, in_layers=feature)
  tg.add_output(conv)
  tg.set_loss(conv)
  tg.build()
  tg.save()


def test_Dense_pickle():
  tg = TensorGraph()
  feature = Feature(shape=(tg.batch_size, 1))
  dense = Dense(out_channels=1, in_layers=feature)
  tg.add_output(dense)
  tg.set_loss(dense)
  tg.build()
  tg.save()


def test_Flatten_pickle():
  tg = TensorGraph()
  feature = Feature(shape=(tg.batch_size, 1))
  layer = Flatten(in_layers=feature)
  tg.add_output(layer)
  tg.set_loss(layer)
  tg.build()
  tg.save()


def test_Reshape_pickle():
  tg = TensorGraph()
  feature = Feature(shape=(tg.batch_size, 1))
  layer = Reshape(shape=(None, 2), in_layers=feature)
  tg.add_output(layer)
  tg.set_loss(layer)
  tg.build()
  tg.save()


def test_Squeeze_pickle():
  tg = TensorGraph()
  feature = Feature(shape=(tg.batch_size, 1))
  layer = Squeeze(in_layers=feature)
  tg.add_output(layer)
  tg.set_loss(layer)
  tg.build()
  tg.save()


def test_Transpose_pickle():
  tg = TensorGraph()
  feature = Feature(shape=(tg.batch_size, 1))
  layer = Transpose(perm=(1, 0), in_layers=feature)
  tg.add_output(layer)
  tg.set_loss(layer)
  tg.build()
  tg.save()


def test_CombineMeanStd_pickle():
  tg = TensorGraph()
  feature = Feature(shape=(tg.batch_size, 1))
  layer = CombineMeanStd(in_layers=[feature, feature])
  tg.add_output(layer)
  tg.set_loss(layer)
  tg.build()
  tg.save()


def test_Repeat_pickle():
  tg = TensorGraph()
  feature = Feature(shape=(tg.batch_size, 1))
  layer = Repeat(n_times=10, in_layers=feature)
  tg.add_output(layer)
  tg.set_loss(layer)
  tg.build()
  tg.save()


def test_Gather_pickle():
  tg = TensorGraph()
  feature = Feature(shape=(tg.batch_size, 1))
  layer = Gather(indices=[[0], [2], [3]], in_layers=feature)
  tg.add_output(layer)
  tg.set_loss(layer)
  tg.build()
  tg.save()


def test_GRU_pickle():
  tg = TensorGraph()
  feature = Feature(shape=(tg.batch_size, 10, 10))
  layer = GRU(n_hidden=10, batch_size=tg.batch_size, in_layers=feature)
  tg.add_output(layer)
  tg.set_loss(layer)
  tg.build()
  tg.save()


def test_L2loss_pickle():
  tg = TensorGraph()
  feature = Feature(shape=(tg.batch_size, 1))
  layer = L2Loss(in_layers=[feature, feature])
  tg.add_output(layer)
  tg.set_loss(layer)
  tg.build()
  tg.save()


def test_Softmax_pickle():
  tg = TensorGraph()
  feature = Feature(shape=(tg.batch_size, 1))
  layer = SoftMax(in_layers=feature)
  tg.add_output(layer)
  tg.set_loss(layer)
  tg.build()
  tg.save()


def test_Concat_pickle():
  tg = TensorGraph()
  feature = Feature(shape=(tg.batch_size, 1))
  layer = Concat(in_layers=[feature, feature])
  tg.add_output(layer)
  tg.set_loss(layer)
  tg.build()
  tg.save()


def test_Constant_pickle():
  tg = TensorGraph()
  feature = Feature(shape=(tg.batch_size, 1))
  layer = Constant(np.array([15.0]))
  output = Add(in_layers=[feature, layer])
  tg.add_output(output)
  tg.set_loss(output)
  tg.build()
  tg.save()


def test_Variable_pickle():
  tg = TensorGraph()
  feature = Feature(shape=(tg.batch_size, 1))
  layer = Variable(np.array([15.0]))
  output = Multiply(in_layers=[feature, layer])
  tg.add_output(output)
  tg.set_loss(output)
  tg.build()
  tg.save()


def test_StopGradient_pickle():
  tg = TensorGraph()
  feature = Feature(shape=(tg.batch_size, 1))
  output = StopGradient(feature)
  tg.add_output(output)
  tg.set_loss(output)
  tg.build()
  tg.save()


def test_Log_pickle():
  tg = TensorGraph()
  feature = Feature(shape=(tg.batch_size, 1))
  layer = Log(feature)
  tg.add_output(layer)
  tg.set_loss(layer)
  tg.build()
  tg.save()


def test_Exp_pickle():
  tg = TensorGraph()
  feature = Feature(shape=(tg.batch_size, 1))
  layer = Exp(feature)
  tg.add_output(layer)
  tg.set_loss(layer)
  tg.build()
  tg.save()


def testInteratomicL2Distances():
  """
    TODO(LESWING) what is ndim here?
    :return:
    """
  tg = TensorGraph()
  n_atoms = tg.batch_size
  M_nbrs = 4
  n_dim = 3
  feature = Feature(shape=(tg.batch_size, 3))
  neighbors = Feature(shape=(tg.batch_size, M_nbrs), dtype=tf.int32)
  layer = InteratomicL2Distances(
      N_atoms=n_atoms,
      M_nbrs=M_nbrs,
      ndim=n_dim,
      in_layers=[feature, neighbors])
  tg.add_output(layer)
  tg.set_loss(layer)
  tg.build()
  tg.save()


def test_SoftmaxCrossEntropy_pickle():
  tg = TensorGraph()
  feature = Feature(shape=(tg.batch_size, 1))
  layer = SoftMaxCrossEntropy(in_layers=[feature, feature])
  tg.add_output(layer)
  tg.set_loss(layer)
  tg.build()
  tg.save()


def test_ReduceMean_pickle():
  tg = TensorGraph()
  feature = Feature(shape=(tg.batch_size, 1))
  layer = ReduceMean(in_layers=[feature])
  tg.add_output(layer)
  tg.set_loss(layer)
  tg.build()
  tg.save()


def test_ToFloat_pickle():
  tg = TensorGraph()
  feature = Feature(shape=(tg.batch_size, 1))
  layer = ToFloat(in_layers=[feature])
  tg.add_output(layer)
  tg.set_loss(layer)
  tg.build()
  tg.save()


def test_ReduceSum_pickle():
  tg = TensorGraph()
  feature = Feature(shape=(tg.batch_size, 1))
  layer = ReduceSum(in_layers=[feature])
  tg.add_output(layer)
  tg.set_loss(layer)
  tg.build()
  tg.save()


def test_ReduceSquareDifference_pickle():
  tg = TensorGraph()
  feature = Feature(shape=(tg.batch_size, 1))
  layer = ReduceSquareDifference(in_layers=[feature, feature])
  tg.add_output(layer)
  tg.set_loss(layer)
  tg.build()
  tg.save()


def test_Conv2D_pickle():
  tg = TensorGraph()
  feature = Feature(shape=(tg.batch_size, 10, 10, 1))
  layer = Conv2D(num_outputs=3, in_layers=feature)
  tg.add_output(layer)
  tg.set_loss(layer)
  tg.build()
  tg.save()


def test_Conv3D_pickle():
  tg = TensorGraph()
  feature = Feature(shape=(tg.batch_size, 10, 10, 10, 1))
  layer = Conv3D(num_outputs=3, in_layers=feature)
  tg.add_output(layer)
  tg.set_loss(layer)
  tg.build()
  tg.save()


def test_Conv2DTranspose_pickle():
  tg = TensorGraph()
  feature = Feature(shape=(tg.batch_size, 10, 10, 1))
  layer = Conv2DTranspose(num_outputs=3, in_layers=feature)
  tg.add_output(layer)
  tg.set_loss(layer)
  tg.build()
  tg.save()


def test_Conv3DTranspose_pickle():
  tg = TensorGraph()
  feature = Feature(shape=(tg.batch_size, 10, 10, 10, 1))
  layer = Conv3DTranspose(num_outputs=3, in_layers=feature)
  tg.add_output(layer)
  tg.set_loss(layer)
  tg.build()
  tg.save()


def test_MaxPool2D_pickle():
  tg = TensorGraph()
  feature = Feature(shape=(tg.batch_size, 10, 10, 10))
  layer = MaxPool2D(in_layers=feature)
  tg.add_output(layer)
  tg.set_loss(layer)
  tg.build()
  tg.save()


def test_MaxPool3D_pickle():
  tg = TensorGraph()
  feature = Feature(shape=(tg.batch_size, 10, 10, 10, 10))
  layer = MaxPool3D(in_layers=feature)
  tg.add_output(layer)
  tg.set_loss(layer)
  tg.build()
  tg.save()


def test_GraphConv_pickle():
  tg = TensorGraph()
  atom_features = Feature(shape=(None, 75))
  degree_slice = Feature(shape=(None, 2), dtype=tf.int32)
  membership = Feature(shape=(None,), dtype=tf.int32)

  deg_adjs = []
  for i in range(0, 10 + 1):
    deg_adj = Feature(shape=(None, i + 1), dtype=tf.int32)
    deg_adjs.append(deg_adj)
  layer = GraphConv(
      64,
      activation_fn=tf.nn.relu,
      in_layers=[atom_features, degree_slice, membership] + deg_adjs)
  tg.add_output(layer)
  tg.set_loss(layer)
  tg.build()
  tg.save()


def test_GraphPool_Pickle():
  tg = TensorGraph()
  atom_features = Feature(shape=(None, 75))
  degree_slice = Feature(shape=(None, 2), dtype=tf.int32)
  membership = Feature(shape=(None,), dtype=tf.int32)
  deg_adjs = []
  for i in range(0, 10 + 1):
    deg_adj = Feature(shape=(None, i + 1), dtype=tf.int32)
    deg_adjs.append(deg_adj)
  layer = GraphPool(
      in_layers=[atom_features, degree_slice, membership] + deg_adjs)
  tg.set_loss(layer)
  tg.build()
  tg.save()


def test_GraphGather_Pickle():
  tg = TensorGraph()
  atom_features = Feature(shape=(None, 75))
  degree_slice = Feature(shape=(None, 2), dtype=tf.int32)
  membership = Feature(shape=(None,), dtype=tf.int32)
  deg_adjs = []
  for i in range(0, 10 + 1):
    deg_adj = Feature(shape=(None, i + 1), dtype=tf.int32)
    deg_adjs.append(deg_adj)
  layer = GraphGather(
      batch_size=tg.batch_size,
      activation_fn=tf.nn.tanh,
      in_layers=[atom_features, degree_slice, membership] + deg_adjs)
  tg.set_loss(layer)
  tg.build()
  tg.save()


def test_BatchNorm_pickle():
  tg = TensorGraph()
  feature = Feature(shape=(tg.batch_size, 10))
  layer = BatchNorm(in_layers=feature)
  tg.add_output(layer)
  tg.set_loss(layer)
  tg.build()
  tg.save()


def test_WeightedError_pickle():
  tg = TensorGraph()
  feature = Feature(shape=(tg.batch_size, 10))
  layer = WeightedError(in_layers=[feature, feature])
  tg.add_output(layer)
  tg.set_loss(layer)
  tg.build()
  tg.save()


def test_Weave_pickle():
  tg = TensorGraph()
  atom_feature = Feature(shape=(None, 75))
  pair_feature = Feature(shape=(None, 14))
  pair_split = Feature(shape=(None,), dtype=tf.int32)
  atom_to_pair = Feature(shape=(None, 2), dtype=tf.int32)
  weave = WeaveLayer(
      in_layers=[atom_feature, pair_feature, pair_split, atom_to_pair])
  tg.add_output(weave)
  tg.set_loss(weave)
  tg.build()
  tg.save()


def test_WeaveGather_pickle():
  tg = TensorGraph()
  atom_feature = Feature(shape=(None, 75))
  atom_split = Feature(shape=(None,), dtype=tf.int32)
  weave_gather = WeaveGather(
      32, gaussian_expand=True, in_layers=[atom_feature, atom_split])
  tg.add_output(weave_gather)
  tg.set_loss(weave_gather)
  tg.build()
  tg.save()


def test_DTNNEmbedding_pickle():
  tg = TensorGraph()
  atom_numbers = Feature(shape=(None, 23), dtype=tf.int32)
  Embedding = DTNNEmbedding(in_layers=[atom_numbers])
  tg.add_output(Embedding)
  tg.set_loss(Embedding)
  tg.build()
  tg.save()


def test_DTNNStep_pickle():
  tg = TensorGraph()
  atom_features = Feature(shape=(None, 30))
  distance = Feature(shape=(None, 100))
  distance_membership_i = Feature(shape=(None,), dtype=tf.int32)
  distance_membership_j = Feature(shape=(None,), dtype=tf.int32)
  DTNN = DTNNStep(in_layers=[
      atom_features, distance, distance_membership_i, distance_membership_j
  ])
  tg.add_output(DTNN)
  tg.set_loss(DTNN)
  tg.build()
  tg.save()


def test_DTNNGather_pickle():
  tg = TensorGraph()
  atom_features = Feature(shape=(None, 30))
  atom_membership = Feature(shape=(None,), dtype=tf.int32)
  Gather = DTNNGather(in_layers=[atom_features, atom_membership])
  tg.add_output(Gather)
  tg.set_loss(Gather)
  tg.build()
  tg.save()


def test_DTNNExtract_pickle():
  tg = TensorGraph()
  atom_features = Feature(shape=(None, 30))
  Ext = DTNNExtract(0, in_layers=[atom_features])
  tg.add_output(Ext)
  tg.set_loss(Ext)
  tg.build()
  tg.save()


def test_DAGLayer_pickle():
  tg = TensorGraph(use_queue=False)
  atom_features = Feature(shape=(None, 75))
  parents = Feature(shape=(None, 50, 50), dtype=tf.int32)
  calculation_orders = Feature(shape=(None, 50), dtype=tf.int32)
  calculation_masks = Feature(shape=(None, 50), dtype=tf.bool)
  n_atoms = Feature(shape=(), dtype=tf.int32)
  DAG = DAGLayer(in_layers=[
      atom_features, parents, calculation_orders, calculation_masks, n_atoms
  ])
  tg.add_output(DAG)
  tg.set_loss(DAG)
  tg.build()
  tg.save()


def test_DAGGather_pickle():
  tg = TensorGraph()
  atom_features = Feature(shape=(None, 30))
  membership = Feature(shape=(None,), dtype=tf.int32)
  Gather = DAGGather(in_layers=[atom_features, membership])
  tg.add_output(Gather)
  tg.set_loss(Gather)
  tg.build()
  tg.save()


def test_MP_pickle():
  tg = TensorGraph()
  atom_feature = Feature(shape=(None, 75))
  pair_feature = Feature(shape=(None, 14))
  atom_to_pair = Feature(shape=(None, 2), dtype=tf.int32)
  MP = MessagePassing(5, in_layers=[atom_feature, pair_feature, atom_to_pair])
  tg.add_output(MP)
  tg.set_loss(MP)
  tg.build()
  tg.save()


def test_AttnLSTM_pickle():
  """Tests that AttnLSTM can be pickled."""
  max_depth = 5
  n_test = 5
  n_support = 5
  n_feat = 10

  tg = TensorGraph(batch_size=n_test)
  test = Feature(shape=(None, n_feat))
  support = Feature(shape=(None, n_feat))
  out = AttnLSTMEmbedding(
      n_test, n_support, n_feat, max_depth, in_layers=[test, support])
  tg.add_output(out)
  tg.set_loss(out)
  tg.build()
  tg.save()


def test_LSTMStep_pickle():
  """Tests that LSTMStep can be pickled."""
  n_feat = 10
  tg = TensorGraph()
  y = Feature(shape=(None, 2 * n_feat))
  state_zero = Feature(shape=(None, n_feat))
  state_one = Feature(shape=(None, n_feat))
  lstm = LSTMStep(n_feat, 2 * n_feat, in_layers=[y, state_zero, state_one])
  tg.add_output(lstm)
  tg.set_loss(lstm)
  tg.build()
  tg.save()


def test_IterRefLSTM_pickle():
  """Tests that IterRefLSTM can be pickled."""
  n_feat = 10
  max_depth = 5
  n_test = 5
  n_support = 5
  tg = TensorGraph()
  test = Feature(shape=(None, n_feat))
  support = Feature(shape=(None, n_feat))
  lstm = IterRefLSTMEmbedding(
      n_test, n_support, n_feat, max_depth, in_layers=[test, support])
  tg.add_output(lstm)
  tg.set_loss(lstm)
  tg.build()
  tg.save()


def test_SetGather_pickle():
  tg = TensorGraph()
  atom_feature = Feature(shape=(None, 100))
  atom_split = Feature(shape=(None,), dtype=tf.int32)
  Gather = SetGather(5, 16, in_layers=[atom_feature, atom_split])
  tg.add_output(Gather)
  tg.set_loss(Gather)
  tg.build()
  tg.save()


def test_AtomicDifferentialDense_pickle():
  max_atoms = 23
  atom_features = 100
  tg = TensorGraph()
  atom_feature = Feature(shape=(None, max_atoms, atom_features))
  atom_numbers = Feature(shape=(None, max_atoms))
  atomic_differential_dense = AtomicDifferentiatedDense(
      max_atoms=23, out_channels=5, in_layers=[atom_feature, atom_numbers])
  tg.add_output(atomic_differential_dense)
  tg.set_loss(atomic_differential_dense)
  tg.build()
  tg.save()


def testGraphCNN_pickle():
  V = Feature(shape=(None, 200, 50))
  A = Feature(shape=(None, 200, 1, 200))
  gcnn = GraphCNN(32, in_layers=[V, A])
  tg = TensorGraph()
  tg.add_output(gcnn)
  tg.set_loss(gcnn)
  tg.build()
  tg.save()


def testGraphCNNPoolLayer_pickle():
  V = Feature(shape=(None, 200, 50))
  A = Feature(shape=(None, 200, 1, 200))
  gcnnpool = GraphEmbedPoolLayer(32, in_layers=[V, A])
  tg = TensorGraph()
  tg.add_output(gcnnpool)
  tg.set_loss(gcnnpool)
  tg.build()
  tg.save()
