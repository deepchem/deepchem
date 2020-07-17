"""
Tests for ConvMolFeaturizer. 
"""
__author__ = "Han Altae-Tran and Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "MIT"

import unittest
import os
import numpy as np
import pytest

from deepchem.feat.graph_features import ConvMolFeaturizer, AtomicConvFeaturizer


class TestConvMolFeaturizer(unittest.TestCase):
  """
  Test ConvMolFeaturizer featurizes properly.
  """

  def test_carbon_nitrogen(self):
    """Test on carbon nitrogen molecule"""
    # Note there is a central nitrogen of degree 4, with 4 carbons
    # of degree 1 (connected only to central nitrogen).
    raw_smiles = ['C[N+](C)(C)C']
    import rdkit.Chem
    mols = [rdkit.Chem.MolFromSmiles(s) for s in raw_smiles]
    featurizer = ConvMolFeaturizer()
    mols = featurizer.featurize(mols)
    mol = mols[0]

    # 5 atoms in compound
    assert mol.get_num_atoms() == 5

    # Get the adjacency lists grouped by degree
    deg_adj_lists = mol.get_deg_adjacency_lists()
    assert np.array_equal(deg_adj_lists[0], np.zeros([0, 0], dtype=np.int32))
    # The 4 outer atoms connected to central nitrogen
    assert np.array_equal(deg_adj_lists[1],
                          np.array([[4], [4], [4], [4]], dtype=np.int32))
    assert np.array_equal(deg_adj_lists[2], np.zeros([0, 2], dtype=np.int32))
    assert np.array_equal(deg_adj_lists[3], np.zeros([0, 3], dtype=np.int32))
    # Central nitrogen connected to everything else.
    assert np.array_equal(deg_adj_lists[4],
                          np.array([[0, 1, 2, 3]], dtype=np.int32))
    assert np.array_equal(deg_adj_lists[5], np.zeros([0, 5], dtype=np.int32))
    assert np.array_equal(deg_adj_lists[6], np.zeros([0, 6], dtype=np.int32))

  def test_single_carbon(self):
    """Test that single carbon atom is featurized properly."""
    raw_smiles = ['C']
    import rdkit
    mols = [rdkit.Chem.MolFromSmiles(s) for s in raw_smiles]
    featurizer = ConvMolFeaturizer()
    mol_list = featurizer.featurize(mols)
    mol = mol_list[0]

    # Only one carbon
    assert mol.get_num_atoms() == 1

    # No bonds, so degree adjacency lists are empty
    deg_adj_lists = mol.get_deg_adjacency_lists()
    assert np.array_equal(deg_adj_lists[0], np.zeros([1, 0], dtype=np.int32))
    assert np.array_equal(deg_adj_lists[1], np.zeros([0, 1], dtype=np.int32))
    assert np.array_equal(deg_adj_lists[2], np.zeros([0, 2], dtype=np.int32))
    assert np.array_equal(deg_adj_lists[3], np.zeros([0, 3], dtype=np.int32))
    assert np.array_equal(deg_adj_lists[4], np.zeros([0, 4], dtype=np.int32))
    assert np.array_equal(deg_adj_lists[5], np.zeros([0, 5], dtype=np.int32))
    assert np.array_equal(deg_adj_lists[6], np.zeros([0, 6], dtype=np.int32))

  def test_alkane(self):
    """Test on simple alkane"""
    raw_smiles = ['CCC']
    import rdkit.Chem
    mols = [rdkit.Chem.MolFromSmiles(s) for s in raw_smiles]
    featurizer = ConvMolFeaturizer()
    mol_list = featurizer.featurize(mols)
    mol = mol_list[0]

    # 3 carbonds in alkane
    assert mol.get_num_atoms() == 3

    deg_adj_lists = mol.get_deg_adjacency_lists()
    assert np.array_equal(deg_adj_lists[0], np.zeros([0, 0], dtype=np.int32))
    # Outer two carbonds are connected to central carbon
    assert np.array_equal(deg_adj_lists[1], np.array(
        [[2], [2]], dtype=np.int32))
    # Central carbon connected to outer two
    assert np.array_equal(deg_adj_lists[2], np.array([[0, 1]], dtype=np.int32))
    assert np.array_equal(deg_adj_lists[3], np.zeros([0, 3], dtype=np.int32))
    assert np.array_equal(deg_adj_lists[4], np.zeros([0, 4], dtype=np.int32))
    assert np.array_equal(deg_adj_lists[5], np.zeros([0, 5], dtype=np.int32))
    assert np.array_equal(deg_adj_lists[6], np.zeros([0, 6], dtype=np.int32))
