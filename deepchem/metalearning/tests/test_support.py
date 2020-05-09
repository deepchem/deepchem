"""
Testing construction of graph models.
"""
import unittest
import tensorflow as tf
import deepchem as dc
import deepchem.models.layers as layers
from deepchem.metalearning.support import SequentialGraph
from deepchem.metalearning.support import SequentialSupportGraph


class TestSupport(unittest.TestCase):
  """
  Test metalearning support models 
  """

  def test_sequential_graph_model(self):
    """Simple test that SequentialGraph can be initialized."""
    n_atoms = 5
    n_feat = 10
    batch_size = 3
    graph_model = tf.keras.Sequential([])
    assert len(graph_model.layers) == 0

  def test_sample_sequential_architecture(self):
    """Tests that a representative architecture can be created."""
    n_atoms = 5
    n_feat = 10
    batch_size = 3
    graph_model = tf.keras.Sequential([
      layers.GraphConv(64, n_feat, activation_fn=tf.nn.relu),
      layers.BatchNormalization(epsilon=1e-5, mode=1),
      layers.GraphPool(),
      ## Gather Projection
      #graph_model.add(dc.nn.Dense(128, activation='relu'))
      layers.BatchNormalization(epsilon=1e-5, mode=1),
      layers.GraphGather(batch_size, activation="tanh")])

    # There should be 8 layers in graph_model
    #assert len(graph_model.layers) == 6
    assert len(graph_model.layers) == 5

  #def test_sample_attn_lstm_architecture(self):
  #  """Tests that an attention architecture can be created without crash."""
  #  max_depth = 5
  #  n_test = 5
  #  n_support = 11
  #  n_feat = 10
  #  batch_size = 3

  #  support_model = SequentialSupportGraph(n_feat)

  #  # Add layers
  #  support_model.add(dc.nn.GraphConv(64, n_feat, activation='relu'))
  #  # Need to add batch-norm separately to test/support due to differing
  #  # shapes.
  #  support_model.add_test(dc.nn.BatchNormalization(epsilon=1e-5, mode=1))
  #  support_model.add_support(dc.nn.BatchNormalization(epsilon=1e-5, mode=1))
  #  support_model.add(dc.nn.GraphPool())

  #  # Apply an attention lstm layer
  #  support_model.join(
  #      dc.nn.AttnLSTMEmbedding(n_test, n_support, 64, max_depth))

  #  # Gather Projection
  #  support_model.add(dc.nn.Dense(128, 64))
  #  support_model.add_test(dc.nn.BatchNormalization(epsilon=1e-5, mode=1))
  #  support_model.add_support(dc.nn.BatchNormalization(epsilon=1e-5, mode=1))
  #  support_model.add(dc.nn.GraphGather(batch_size, activation="tanh"))

  #def test_sample_resi_lstm_architecture(self):
  #  """Tests that an attention architecture can be created without crash."""
  #  max_depth = 5
  #  n_test = 5
  #  n_support = 11
  #  n_feat = 10
  #  batch_size = 3

  #  support_model = SequentialSupportGraph(n_feat)

  #  # Add layers
  #  support_model.add(dc.nn.GraphConv(64, n_feat, activation='relu'))
  #  # Need to add batch-norm separately to test/support due to differing
  #  # shapes.
  #  support_model.add_test(dc.nn.BatchNormalization(epsilon=1e-5, mode=1))
  #  support_model.add_support(dc.nn.BatchNormalization(epsilon=1e-5, mode=1))
  #  support_model.add(dc.nn.GraphPool())

  #  # Apply an attention lstm layer
  #  support_model.join(
  #      dc.nn.ResiLSTMEmbedding(n_test, n_support, 64, max_depth))

  #  # Gather Projection
  #  support_model.add(dc.nn.Dense(128, 64))
  #  support_model.add_test(dc.nn.BatchNormalization(epsilon=1e-5, mode=1))
  #  support_model.add_support(dc.nn.BatchNormalization(epsilon=1e-5, mode=1))
  #  support_model.add(dc.nn.GraphGather(batch_size, activation="tanh"))
