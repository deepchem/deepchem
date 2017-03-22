"""
Simple Tests for Graph Topologies 
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Han Altae-Tran and Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "MIT"

import unittest
import tensorflow as tf
from deepchem.models.tf_new_models.graph_topology import GraphTopology


class TestGraphTopology(unittest.TestCase):
  """
  Test that graph topologies work correctly.
  """

  def test_shapes(self):
    """Simple test that Graph topology placeholders have correct shapes."""
    n_atoms = 5
    n_feat = 10
    batch_size = 3
    max_deg = 10
    min_deg = 0
    topology = GraphTopology(n_feat)

    # Degrees from 1 to max_deg inclusive 
    # TODO(rbharath): Should this be 0 to max_deg inclusive?
    deg_adj_lists_placeholders = topology.get_deg_adjacency_lists_placeholders()
    assert len(deg_adj_lists_placeholders) == max_deg
    for ind, deg_adj_list in enumerate(deg_adj_lists_placeholders):
      deg = ind + 1
      # Should have shape (?, deg)
      assert deg_adj_list.get_shape()[1] == deg

    # Shape of atom_features should be (?, n_feat)
    atom_features = topology.get_atom_features_placeholder()
    assert atom_features.get_shape()[1] == n_feat

    # Shape of deg_slice placeholder should be (max_deg+1-min_deg, 2)
    deg_slice = topology.get_deg_slice_placeholder()
    print("deg_slice.get_shape()")
    print(deg_slice.get_shape())
    assert deg_slice.get_shape() == (max_deg + 1 - min_deg, 2)
