"""
Tests for Molecular Graph data structures. 
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Han Altae-Tran and Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "MIT"

import unittest
import os
import sys
import numpy as np
import rdkit
from deepchem.feat.mol_graphs import ConvMol
from deepchem.feat.mol_graphs import MultiConvMol


class TestMolGraphs(unittest.TestCase):
  """
  Test mol graphs.
  """

  def test_construct_conv_mol(self):
    """Tests that ConvMols can be constructed without crash."""
    N_feat = 4
    # Artificial feature array.
    atom_features = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    adj_list = [[1], [0, 2], [1]]
    mol = ConvMol(atom_features, adj_list)

  def test_conv_mol_deg_slice(self):
    """Tests that deg_slice works properly."""
    atom_features = np.array([[20, 21, 22, 23], [24, 25, 26, 27],
                              [28, 29, 30, 31], [32, 33, 34, 35]])
    adj_list = [[1, 2], [0, 3], [0, 3], [1, 2]]
    mol = ConvMol(atom_features, adj_list)

    assert np.array_equal(
        mol.get_deg_slice(),
        # 0 atoms of degree 0
        # 0 atoms of degree 1
        # 4 atoms of degree 2
        # 0 atoms of degree 3
        # 0 atoms of degree 4
        # 0 atoms of degree 5
        # 0 atoms of degree 6
        # 0 atoms of degree 7
        # 0 atoms of degree 8
        # 0 atoms of degree 9
        # 0 atoms of degree 10
        np.array([[0, 0], [0, 0], [0, 4], [0, 0], [0, 0], [0, 0], [0, 0],
                  [0, 0], [0, 0], [0, 0], [0, 0]]))

  def test_get_atom_features(self):
    """Test that the atom features are computed properly."""
    atom_features = np.array([[40, 41, 42, 43], [44, 45, 46, 47],
                              [48, 49, 50, 51], [52, 53, 54, 55],
                              [56, 57, 58, 59]])
    canon_adj_list = [[1, 2], [0, 3], [0, 3], [1, 2, 4], [3]]
    mol = ConvMol(atom_features, canon_adj_list)
    # atom 4 has 0 neighbors
    # atom 0 has 2 neighbors
    # atom 1 has 2 neighbors
    # atom 2 has 2 neighbors
    # atom 3 has 3 neighbors.
    # Verify that atom features have been sorted by atom degree.
    assert np.array_equal(mol.get_atom_features(),
                          np.array([[56, 57, 58, 59], [40, 41, 42, 43],
                                    [44, 45, 46, 47], [48, 49, 50, 51],
                                    [52, 53, 54, 55]]))

  def test_get_adjacency_list(self):
    """Tests that adj-list is canonicalized properly."""
    atom_features = np.array([[40, 41, 42, 43], [44, 45, 46, 47],
                              [48, 49, 50, 51], [52, 53, 54, 55],
                              [56, 57, 58, 59]])
    canon_adj_list = [[1, 2], [0, 3], [0, 3], [1, 2, 4], [3]]
    mol = ConvMol(atom_features, canon_adj_list)
    # Sorting is done by atom degree as before. So the ordering goes
    # 4, 0, 1, 2, 3 now in terms of the original ordering. The mapping
    # from new position to old position is 
    # {(4, 0), (0, 1), (1, 2), (2, 3), (3, 4)}. Check that adjacency
    # list respects this reordering and returns correct adjacency list.
    assert (
        mol.get_adjacency_list() == [[4], [2, 3], [1, 4], [1, 4], [2, 3, 0]])

  def test_agglomerate_molecules(self):
    """Test AggrMol.agglomerate_mols."""
    molecules = []

    #### First example molecule
    N_feat = 4
    # Artificial feature array.
    atom_features = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    adj_list = [[1], [0, 2], [1]]
    molecules.append(ConvMol(atom_features, adj_list))

    #### Second example molecule
    atom_features = np.array([[20, 21, 22, 23], [24, 25, 26, 27],
                              [28, 29, 30, 31], [32, 33, 34, 35]])
    adj_list = [[1, 2], [0, 3], [0, 3], [1, 2]]
    molecules.append(ConvMol(atom_features, adj_list))

    ### Third example molecule
    atom_features = np.array([[40, 41, 42, 43], [44, 45, 46, 47],
                              [48, 49, 50, 51], [52, 53, 54, 55],
                              [56, 57, 58, 59]])
    adj_list = [[1, 2], [0, 3], [0, 3], [1, 2, 4], [3]]
    molecules.append(ConvMol(atom_features, adj_list))

    # Test agglomerate molecule method
    concat_mol = ConvMol.agglomerate_mols(molecules)

    assert concat_mol.get_num_atoms() == 12
    assert concat_mol.get_num_molecules() == 3

    atom_features = concat_mol.get_atom_features()
    assert np.array_equal(atom_features[0, :], [1, 2, 3, 4])
    assert np.array_equal(atom_features[2, :], [56, 57, 58, 59])
    assert np.array_equal(atom_features[11, :], [52, 53, 54, 55])
    assert np.array_equal(atom_features[4, :], [20, 21, 22, 23])

    deg_adj_lists = concat_mol.get_deg_adjacency_lists()
    # No atoms of degree 0
    assert np.array_equal(deg_adj_lists[0], np.zeros([0, 0]))
    # 3 atoms of degree 1
    assert np.array_equal(deg_adj_lists[1], [[3], [3], [11]])
    # 8 atoms of degree 2
    assert np.array_equal(
        deg_adj_lists[2],
        [[0, 1], [5, 6], [4, 7], [4, 7], [5, 6], [9, 10], [8, 11], [8, 11]])
    # 1 atom of degree 3
    assert np.array_equal(deg_adj_lists[3], [[9, 10, 2]])
    # 0 atoms of degree 4
    assert np.array_equal(deg_adj_lists[4], np.zeros([0, 4]))
    # 0 atoms of degree 5
    assert np.array_equal(deg_adj_lists[5], np.zeros([0, 5]))

  def test_null_conv_mol(self):
    """Running Null AggrMol Test. Only works when max_deg=6 and min_deg=0"""
    num_feat = 4
    min_deg = 0
    null_mol = ConvMol.get_null_mol(num_feat)

    deg_adj_lists = null_mol.get_deg_adjacency_lists()

    # Check that atoms are only connected to themselves.
    assert np.array_equal(deg_adj_lists[10],
                          [[10, 10, 10, 10, 10, 10, 10, 10, 10, 10]])
    assert np.array_equal(deg_adj_lists[1], [[1]])
    # Check that there's one atom of each degree.
    assert np.array_equal(null_mol.get_deg_slice(),
                          [[0, 1], [1, 1], [2, 1], [3, 1], [4, 1], [5, 1],
                           [6, 1], [7, 1], [8, 1], [9, 1], [10, 1]])
