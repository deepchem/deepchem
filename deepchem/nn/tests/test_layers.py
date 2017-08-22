"""
Test that Layers work as advertised.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Han Altae-Tran and Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "MIT"

import numpy as np
import unittest
import deepchem as dc
from tensorflow.python.framework import test_util


class TestLayers(test_util.TensorFlowTestCase):
  """
  Test Layers.

  The tests in this class only do basic sanity checks to make sure that
  produced tensors have the right shape.
  """

  def setUp(self):
    super(TestLayers, self).setUp()
    self.root = '/tmp'

  def test_dense(self):
    """Tests dense layer class can be initialized."""
    with self.test_session() as sess:
      dense = dc.nn.Dense(32, input_dim=16)

  def test_dropout(self):
    """Tests that dropout can be initialized."""
    with self.test_session() as sess:
      dropout = dc.nn.Dropout(.5)

  def test_input(self):
    """Tests that inputs can be created."""
    with self.test_session() as sess:
      input_layer = dc.nn.Input(shape=(32,))

  def test_graph_convolution(self):
    """Tests that Graph Convolution transforms shapes correctly."""
    n_atoms = 5
    n_feat = 10
    nb_filter = 7
    with self.test_session() as sess:
      graph_topology = dc.nn.GraphTopology(n_feat)
      graph_conv_layer = dc.nn.GraphConv(nb_filter, n_feat)

      X = graph_topology.get_input_placeholders()
      out = graph_conv_layer(X)
      # Output should be of shape (?, nb_filter)
      assert out.get_shape()[1] == nb_filter

  def test_graph_gather(self):
    """Tests that GraphGather transforms shapes correctly."""
    n_atoms = 5
    n_feat = 10
    batch_size = 3
    nb_filter = 7
    with self.test_session() as sess:
      graph_topology = dc.nn.GraphTopology(n_feat)
      graph_gather_layer = dc.nn.GraphGather(batch_size)

      X = graph_topology.get_input_placeholders()
      out = graph_gather_layer(X)
      # Output should be of shape (batch_size, n_feat)
      assert out.get_shape() == (batch_size, n_feat)

  def test_graph_pool(self):
    """Tests that GraphPool transforms shapes correctly."""
    n_atoms = 5
    n_feat = 10
    batch_size = 3
    nb_filter = 7
    with self.test_session() as sess:
      graph_topology = dc.nn.GraphTopology(n_feat)
      graph_pool_layer = dc.nn.GraphPool()

      X = graph_topology.get_input_placeholders()
      out = graph_pool_layer(X)

  def test_attn_lstm_embedding(self):
    """Test that attention LSTM computation works properly."""
    max_depth = 5
    n_test = 5
    n_support = 11
    n_feat = 10
    nb_filter = 7
    with self.test_session() as sess:
      graph_topology_test = dc.nn.GraphTopology(n_feat)
      graph_topology_support = dc.nn.GraphTopology(n_feat)

      test = graph_topology_test.get_input_placeholders()[0]
      support = graph_topology_support.get_input_placeholders()[0]

      attn_embedding_layer = dc.nn.AttnLSTMEmbedding(n_test, n_support, n_feat,
                                                     max_depth)
      # Try concatenating the two lists of placeholders
      feed_dict = {
          test: np.zeros((n_test, n_feat)),
          support: np.zeros((n_support, n_feat))
      }
      test_out, support_out = attn_embedding_layer([test, support])
      assert test_out.get_shape() == (n_test, n_feat)
      assert support_out.get_shape()[1] == (n_feat)

  def test_resi_lstm_embedding(self):
    """Test that attention LSTM computation works properly."""
    max_depth = 5
    n_test = 5
    n_support = 11
    n_feat = 10
    nb_filter = 7
    with self.test_session() as sess:
      graph_topology_test = dc.nn.GraphTopology(n_feat)
      graph_topology_support = dc.nn.GraphTopology(n_feat)

      test = graph_topology_test.get_input_placeholders()[0]
      support = graph_topology_support.get_input_placeholders()[0]

      resi_embedding_layer = dc.nn.ResiLSTMEmbedding(n_test, n_support, n_feat,
                                                     max_depth)
      # Try concatenating the two lists of placeholders
      feed_dict = {
          test: np.zeros((n_test, n_feat)),
          support: np.zeros((n_support, n_feat))
      }
      test_out, support_out = resi_embedding_layer([test, support])
      assert test_out.get_shape() == (n_test, n_feat)
      assert support_out.get_shape()[1] == (n_feat)
