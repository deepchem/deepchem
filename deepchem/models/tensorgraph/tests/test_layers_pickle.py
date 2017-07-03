import numpy as np
import tensorflow as tf
from deepchem.models import TensorGraph
from deepchem.models.tensorgraph.layers import Feature, Conv1D, Dense, Flatten, Reshape, Squeeze, Transpose, \
    CombineMeanStd, Repeat, GRU, L2Loss, Concat, SoftMax, Constant, Variable, Add, Multiply, InteratomicL2Distances, \
    SoftMaxCrossEntropy, ReduceMean, ToFloat, ReduceSquareDifference, Conv2D, MaxPool, ReduceSum


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
    layer = Reshape(shape=(-1, 2), in_layers=feature)
    tg.add_output(layer)
    tg.set_loss(layer)
    tg.build()
    tg.save()


def test_Squeeze_pickle():
    tg = TensorGraph()
    feature = Feature(shape=(tg.batch_size, 1))
    layer = Squeeze(squeeze_dims=-1, in_layers=feature)
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
    layer = Constant(np.expand_dims([17] * tg.batch_size, -1))
    output = Add(in_layers=[feature, layer])
    tg.add_output(output)
    tg.set_loss(output)
    tg.build()
    tg.save()


def test_Variable_pickle():
    tg = TensorGraph()
    feature = Feature(shape=(tg.batch_size, 1))
    layer = Variable(np.expand_dims([17] * tg.batch_size, -1))
    output = Multiply(in_layers=[feature, layer])
    tg.add_output(output)
    tg.set_loss(output)
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
    layer = InteratomicL2Distances(N_atoms=n_atoms, M_nbrs=M_nbrs, ndim=n_dim, in_layers=[feature, neighbors])
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
    feature = Feature(shape=(tg.batch_size, 10, 10))
    layer = Conv2D(num_outputs=3, in_layers=feature)
    tg.add_output(layer)
    tg.set_loss(layer)
    tg.build()
    tg.save()

def test_MaxPool_pickle():
    tg = TensorGraph()
    feature = Feature(shape=(tg.batch_size, 10, 10, 10))
    layer = MaxPool(in_layers=feature)
    tg.add_output(layer)
    tg.set_loss(layer)
    tg.build()
    tg.save()
