"""
Tests for Molecular Graph data structures. 
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Han Altae-Tran and Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "GPL"

import unittest
import os
import sys
import numpy as np
import rdkit
from deepchem.featurizers.mol_graphs import ConvMol
from deepchem.featurizers.mol_graphs import MultiConvMol
from deepchem.featurizers.graph_features import ConvMolFeaturizer

class TestMolGraphs(unittest.TestCase):
  """
  Test DataStructures for Low Data experiments.
  """
  def test_construct_conv_mol(self):
    """Tests that ConvMols can be constructed without crash."""
    N_feat = 4
    # Artificial feature array.
    nodes = np.array([[1, 2, 3, 4],
                      [5, 6, 7, 8],
                      [9, 10, 11, 12]])
    canon_adj_list = [[1],[0,2],[1]]
    mol = ConvMol(nodes, canon_adj_list)

  def test_conv_mol_deg_slice(self):
    """Tests that deg_slice works properly."""
    nodes = np.array([[20, 21, 22, 23],
                      [24, 25, 26, 27],
                      [28, 29, 30, 31],
                      [32, 33, 34, 35]])
    canon_adj_list = [[1, 2], [0, 3], [0, 3], [1, 2]]
    mol = ConvMol(nodes, canon_adj_list)

    assert np.array_equal(
        mol.get_deg_slice(),
        # 0 atoms of degree 0
        # 0 atoms of degree 1
        # 4 atoms of degree 2
        # 0 atoms of degree 3
        # 0 atoms of degree 4
        # 0 atoms of degree 5
        # 0 atoms of degree 6
        np.array([[0, 0], [0, 0], [0, 4], [0, 0], [0, 0], [0, 0], [0,0]]))

  def test_agglomerate_molecules(self):
    """Test AggrMol.agglomerate_mols."""
    molecules = []

    #### First example molecule
    N_feat = 4
    # Artificial feature array.
    nodes = np.array([[1, 2, 3, 4],
                      [5, 6, 7, 8],
                      [9, 10, 11, 12]])
    canon_adj_list = [[1],[0,2],[1]]
    molecules.append(ConvMol(nodes, canon_adj_list))
    
    #### Second example molecule
    nodes = np.array([[20, 21, 22, 23],
                      [24, 25, 26, 27],
                      [28, 29, 30, 31],
                      [32, 33, 34, 35]])
    canon_adj_list = [[1, 2], [0, 3], [0, 3], [1, 2]]
    molecules.append(ConvMol(nodes, canon_adj_list))

    ### Third example molecule
    nodes = np.array([[40, 41, 42, 43],
                      [44, 45, 46, 47],
                      [48, 49, 50, 51],
                      [52, 53, 54, 55],
                      [56, 57, 58, 59]])
    canon_adj_list = [[1, 2], [0, 3], [0, 3], [1, 2, 4], [3]]
    mol = ConvMol(nodes, canon_adj_list)
    assert mol.canon_adj_list == [[4], [2, 3], [1, 4], [1, 4], [2, 3, 0]]
    assert np.array_equal(
      mol.nodes,
      np.array([[56, 57, 58, 59],
                [40, 41, 42, 43],
                [44, 45, 46, 47],
                [48, 49, 50, 51],
                [52, 53, 54, 55]]))
    molecules.append(mol)    
    
    # Test agglomerate molecule method
    concat_mol = ConvMol.agglomerate_mols(molecules)

    assert concat_mol.N_nodes == 12
    assert concat_mol.N_mols == 3
    assert np.array_equal(
      concat_mol.nodes[0,:], [1, 2, 3, 4])
    assert np.array_equal(
      concat_mol.nodes[2,:], [56, 57, 58, 59])
    assert np.array_equal(
      concat_mol.nodes[11,:], [52, 53, 54, 55])    
    assert np.array_equal(
      concat_mol.nodes[4,:], [20, 21, 22, 23])        

    assert np.array_equal(
      concat_mol.deg_adj_lists[0], np.zeros([0,0]))
    assert np.array_equal(
      concat_mol.deg_adj_lists[1], [[3], [3], [11]])
    assert np.array_equal(
      concat_mol.deg_adj_lists[2],
      [[0, 1], [5, 6], [4, 7], [4, 7], [5, 6], [9, 10], [8, 11], [8, 11]])
    assert np.array_equal(
      concat_mol.deg_adj_lists[3], [[9, 10, 2]])
    assert np.array_equal(
      concat_mol.deg_adj_lists[4], np.zeros([0, 4]))
    assert np.array_equal(
      concat_mol.deg_adj_lists[5], np.zeros([0, 5]))

  
  def test_null_conv_mol(self):
    """Running Null AggrMol Test. Only works when max_deg=6 and min_deg=0"""
    N_feat = 4
    min_deg = 0
    null_mol = ConvMol.get_null_mol(N_feat)

    assert np.array_equal(
        null_mol.deg_adj_lists[6-min_deg], [[6, 6, 6, 6, 6, 6]])
    assert np.array_equal(
        null_mol.deg_adj_lists[1-min_deg], [[1]])
    assert np.array_equal(
        null_mol.deg_slice,
        [[0, 1], [1, 1], [2, 1], [3, 1], [4, 1], [5, 1], [6, 1]])


  def test_featurizer(self):
    """Running AggrMol SmilesDataManager Test"""
    raw_smiles = ['C[N+](C)(C)C', 'CCC', 'C']
    mols = [rdkit.Chem.MolFromSmiles(s) for s in raw_smiles]
    featurizer = ConvMolFeaturizer()
    mol_list = featurizer.featurize(mols)

    mol1 = mol_list[0]
    assert np.array_equal(mol1.deg_adj_lists[0],
                          np.zeros([0,0], dtype=np.int32))
    assert np.array_equal(mol1.deg_adj_lists[1],
                          np.array([[4], [4], [4], [4]], dtype=np.int32))
    assert np.array_equal(mol1.deg_adj_lists[2],
                          np.zeros([0,2], dtype=np.int32))
    assert np.array_equal(mol1.deg_adj_lists[3],
                          np.zeros([0,3], dtype=np.int32))
    assert np.array_equal(mol1.deg_adj_lists[4],
                          np.array([[0, 1, 2, 3]], dtype=np.int32))
    assert np.array_equal(mol1.deg_adj_lists[5],
                          np.zeros([0,5], dtype=np.int32))
    assert np.array_equal(mol1.deg_adj_lists[6],
                          np.zeros([0,6], dtype=np.int32))

    mol2 = mol_list[1]
    assert np.array_equal(mol2.deg_adj_lists[0],
                          np.zeros([0,0], dtype=np.int32))
    assert np.array_equal(mol2.deg_adj_lists[1],
                          np.array([[2], [2]], dtype=np.int32))
    assert np.array_equal(mol2.deg_adj_lists[2],
                          np.array([[0,1]], dtype=np.int32))
    assert np.array_equal(mol2.deg_adj_lists[3],
                          np.zeros([0,3], dtype=np.int32))
    assert np.array_equal(mol2.deg_adj_lists[4],
                          np.zeros([0,4], dtype=np.int32))
    assert np.array_equal(mol2.deg_adj_lists[5],
                          np.zeros([0,5], dtype=np.int32))
    assert np.array_equal(mol2.deg_adj_lists[6],
                          np.zeros([0,6], dtype=np.int32))
    
    mol3 = mol_list[2]
    assert np.array_equal(mol3.deg_adj_lists[0],
                          np.zeros([1,0], dtype=np.int32))
    assert np.array_equal(mol3.deg_adj_lists[1],
                          np.zeros([0,1], dtype=np.int32))
    assert np.array_equal(mol3.deg_adj_lists[2],
                          np.zeros([0,2], dtype=np.int32))
    assert np.array_equal(mol3.deg_adj_lists[3],
                          np.zeros([0,3], dtype=np.int32))
    assert np.array_equal(mol3.deg_adj_lists[4],
                          np.zeros([0,4], dtype=np.int32))
    assert np.array_equal(mol3.deg_adj_lists[5],
                          np.zeros([0,5], dtype=np.int32))
    assert np.array_equal(mol3.deg_adj_lists[6],
                          np.zeros([0,6], dtype=np.int32))
    
