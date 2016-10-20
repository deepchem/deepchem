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
from deepchem.models.tf_keras_models.containers import GraphContainer
from deepchem.models.tf_keras_models.graph_topology import GraphTopology
from deepchem.models.tf_keras_models.graph_models import SequentialGraphModel

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
    n_atom_feat = 10
    batch_size = 3
    graph_model = SequentialGraphModel(n_atoms, n_feat, batch_size)
    assert len(graph_model.layers) == 0
