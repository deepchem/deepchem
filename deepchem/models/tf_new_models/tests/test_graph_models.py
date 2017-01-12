"""
Testing construction of graph models.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Han Altae-Tran and Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "GPL"

import unittest
import tensorflow as tf
from keras import backend as K
from tensorflow.python.framework import test_util
from deepchem.nn.copy import Dense
from deepchem.nn.copy import BatchNormalization
from deepchem.nn.layers import GraphConv
from deepchem.nn.layers import GraphPool
from deepchem.nn.layers import GraphGather
from deepchem.nn.layers import AttnLSTMEmbedding
from deepchem.models.tf_new_models.graph_models import SequentialGraph
from deepchem.models.tf_new_models.graph_models import SequentialSupportGraph

class TestGraphModels(test_util.TensorFlowTestCase):
  """
  Test Container usage.
  """
  def setUp(self):
    super(TestGraphModels, self).setUp()
    self.root = '/tmp'

  def test_sequential_graph_model(self):
    """Simple test that SequentialGraph can be initialized."""
    n_atoms = 5
    n_feat = 10
    batch_size = 3
    graph_model = SequentialGraph(n_feat)
    assert len(graph_model.layers) == 0

  def test_sample_sequential_architecture(self):
    """Tests that a representative architecture can be created."""
    n_atoms = 5
    n_feat = 10
    batch_size = 3
    graph_model = SequentialGraph(n_feat)

    graph_model.add(GraphConv(64, activation='relu'))
    graph_model.add(BatchNormalization(epsilon=1e-5, mode=1))
    graph_model.add(GraphPool())

    # Gather Projection
    graph_model.add(Dense(128, activation='relu'))
    graph_model.add(BatchNormalization(epsilon=1e-5, mode=1))
    graph_model.add(GraphGather(batch_size, activation="tanh"))

    # There should be 8 layers in graph_model
    assert len(graph_model.layers) == 6

  def test_sample_attn_lstm_architecture(self):
    """Tests that an attention architecture can be created without crash."""
    g = tf.Graph()
    sess = tf.Session(graph=g)
    K.set_session(sess)
    with g.as_default():
      max_depth = 5
      n_test = 5
      n_support = 11 
      n_feat = 10
      batch_size = 3

      support_model = SequentialSupportGraph(n_feat)
      
      # Add layers
      support_model.add(GraphConv(64, activation='relu'))
      # Need to add batch-norm separately to test/support due to differing
      # shapes.
      support_model.add_test(BatchNormalization(epsilon=1e-5, mode=1))
      support_model.add_support(BatchNormalization(epsilon=1e-5, mode=1))
      support_model.add(GraphPool())

      # Apply an attention lstm layer
      support_model.join(AttnLSTMEmbedding(n_test, n_support, max_depth))

      # Gather Projection
      support_model.add(Dense(128, activation='relu'))
      support_model.add_test(BatchNormalization(epsilon=1e-5, mode=1))
      support_model.add_support(BatchNormalization(epsilon=1e-5, mode=1))
      support_model.add(GraphGather(batch_size, activation="tanh"))
