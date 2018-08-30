import numpy as np
import tensorflow as tf
import six

from deepchem.data import NumpyDataset, pad_features
from deepchem.feat.mol_graphs import ConvMol
from deepchem.metrics import to_one_hot
from deepchem.models.tensorgraph.graph_layers import WeaveGather, DTNNEmbedding, DTNNStep, DTNNGather, DAGLayer, DAGGather, DTNNExtract, MessagePassing, SetGather, WeaveLayer, EdgeNetwork, GatedRecurrentUnit
from deepchem.models.tensorgraph.layers import Feature
from tensorflow.python.framework import test_util


class TestGraphLayers(test_util.TensorFlowTestCase):
    def test_weave_gather(self):
        batch_size = 10
        n_graph_feat = 128
        atom_split = np.random.rand(batch_size)
        in_tensor = np.random.rand(batch_size)
        with self.test_session() as sess:
            in_tensor1 = tf.convert_to_tensor(in_tensor, dtype=tf.float32)
            atom_split = tf.convert_to_tensor(atom_split,dtype = tf.int32)
            out_tensor = WeaveGather(batch_size)(in_tensor,atom_split)
            sess.run(tf.global_variables_initializer())
            out_tensor = out_tensor.eval()


    def test_dtnn_embedding(self):
        n_embedding = 30
        atom_number = 10
        in_tensor = np.random.rand(atom_number)
        with self.test_session() as sess:
            in_tensor = tf.convert_to_tensor(in_tensor,dtype = tf.float32)
            out_tensor = DTNNEmbedding(n_embedding = n_embedding)(in_tensor)
            sess.run(tf.global_variabled_initializer())
            out_tensor = out_tensor.eval()


