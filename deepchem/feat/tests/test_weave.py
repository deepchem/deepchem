"""
Tests for weave featurizer.
"""
import numpy as np
import deepchem as dc
from deepchem.feat.graph_features import max_pair_distance_pairs


def test_max_pair_distance_pairs():
  """Test that max pair distance pairs are computed properly."""
  from rdkit import Chem
  # Carbon
  mol = Chem.MolFromSmiles('C')
  # Test distance 1
  pair_edges = max_pair_distance_pairs(mol, 1)
  assert pair_edges.shape == (2, 1)
  assert np.all(pair_edges.flatten() == np.array([0, 0]))
  # Test distance 2
  pair_edges = max_pair_distance_pairs(mol, 2)
  assert pair_edges.shape == (2, 1)
  assert np.all(pair_edges.flatten() == np.array([0, 0]))

  # Test alkane
  mol = Chem.MolFromSmiles('CCC')
  # Test distance 1
  pair_edges = max_pair_distance_pairs(mol, 1)
  # 3 self connections and 2 bonds which are both counted twice because of
  # symmetry for 7 total
  assert pair_edges.shape == (2, 7)
  # Test distance 2
  pair_edges = max_pair_distance_pairs(mol, 2)
  # Everything is connected at this distance
  assert pair_edges.shape == (2, 9)


def test_single_carbon():
  """Test that single carbon atom is featurized properly."""
  mols = ['C']
  featurizer = dc.feat.WeaveFeaturizer()
  #from rdkit import Chem
  mol_list = featurizer.featurize(mols)
  mol = mol_list[0]
  #mol = featurizer._featurize(Chem.MolFromSmiles("C"))

  # Only one carbon
  assert mol.get_num_atoms() == 1

  # Test feature sizes
  assert mol.get_num_features() == 75

  # No bonds, so only 1 pair feature (for the self interaction)
  assert mol.get_pair_features().shape == (1 * 1, 14)


def test_alkane():
  """Test on simple alkane"""
  mols = ['CCC']
  featurizer = dc.feat.WeaveFeaturizer()
  mol_list = featurizer.featurize(mols)
  mol = mol_list[0]

  # 3 carbonds in alkane
  assert mol.get_num_atoms() == 3

  # Test feature sizes
  assert mol.get_num_features() == 75

  # Should be a 3x3 interaction grid
  assert mol.get_pair_features().shape == (3 * 3, 14)


def test_carbon_nitrogen():
  """Test on carbon nitrogen molecule"""
  # Note there is a central nitrogen of degree 4, with 4 carbons
  # of degree 1 (connected only to central nitrogen).
  mols = ['C[N+](C)(C)C']
  #import rdkit.Chem
  #mols = [rdkit.Chem.MolFromSmiles(s) for s in raw_smiles]
  featurizer = dc.feat.WeaveFeaturizer()
  mols = featurizer.featurize(mols)
  mol = mols[0]

  # 5 atoms in compound
  assert mol.get_num_atoms() == 5

  # Test feature sizes
  assert mol.get_num_features() == 75

  # Should be a 3x3 interaction grid
  assert mol.get_pair_features().shape == (5 * 5, 14)


#def test_alkane_max_pair_distance():
#  """Test on simple alkane with max_pair_distance < infinity"""
#  mols = ['CCC']
#  featurizer = dc.feat.WeaveFeaturizer(max_pair_distance=1)
#  mol_list = featurizer.featurize(mols)
#  mol = mol_list[0]
#
#  # 3 carbonds in alkane
#  assert mol.get_num_atoms() == 3
#
#  # Test feature sizes
#  assert mol.get_num_features() == 75
#
#  # Should be a 3x3 interaction grid
#  assert mol.get_pair_features().shape == (3, 1, 14)
