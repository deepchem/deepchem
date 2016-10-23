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
from tensorflow.python.framework import test_util
from keras.layers import Dense, BatchNormalization
from deepchem.models.tf_keras_models.graph_models import SequentialGraphModel
from deepchem.models.tf_keras_models.graph_models import SequentialSupportGraphModel
from deepchem.models.tf_keras_models.keras_layers import GraphConv
from deepchem.models.tf_keras_models.keras_layers import GraphPool
from deepchem.models.tf_keras_models.keras_layers import GraphGather

class TestGraphModels(test_util.TensorFlowTestCase):
  """
  Test Container usage.
  """
  def setUp(self):
    super(TestGraphModels, self).setUp()
    self.root = '/tmp'

  def test_sequential_graph_model(self):
    """Simple test that SequentialGraphModel can be initialized."""
    n_atoms = 5
    n_feat = 10
    batch_size = 3
    graph_model = SequentialGraphModel(n_atoms, n_feat, batch_size)
    assert len(graph_model.layers) == 0

  def test_sample_sequential_architecture(self):
    """Tests that a representative architecture can be created."""
    n_atoms = 5
    n_feat = 10
    batch_size = 3
    graph_model = SequentialGraphModel(n_atoms, n_feat, batch_size)

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
    """Tests that a representative attention architecture can be created."""
    max_depth = 5
    n_test = 5
    n_support = 11 
    n_feat = 10
    nb_filter = 7

    support_model = SequentialSupportGraphModel(n_test, n_support, n_feat)
