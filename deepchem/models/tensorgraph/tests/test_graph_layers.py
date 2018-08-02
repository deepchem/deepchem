import numpy as np
import tensorflow as tf
import six

from deepchem.data import NumpyDataset, pad_features
from deepchem.feat.mol_graphs import ConvMol
from deepchem.metrics import to_one_hot
from deepchem.models.tensorgraph.graph_layers import WeaveGather, DTNNEmbedding, DTNNStep, DTNNGather, DAGLayer, DAGGather, DTNNExtract, MessagePassing, SetGather, WeaveLayer, EdgeNetwork, GatedRecurrentUnit

from tensorflow.python.framework import test_util


class TestGraphLayers(test_util.TensorFlowTestCase):
    def Weavegather(self):
        batch_size = 10
        in_tensor = np.random.rand(batch_size)
        with self.test_session() as sess:
            in_tensor = tf.convert_to_tensor(in_tensor, dtype=tf.float32)
            out_tensor = WeaveGather()(in_tensor)
            sess.run(tf.global_variables_initializer())
            out_tensor = out_tensor.eval()

            self.assertEqual(out_tensor.shape[0], batch_size)
