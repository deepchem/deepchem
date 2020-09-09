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


def test_max_pair_distance_infinity():
  """Test that max pair distance pairs are computed properly with infinity distance."""
  from rdkit import Chem
  # Test alkane
  mol = Chem.MolFromSmiles('CCC')
  # Test distance infinity
  pair_edges = max_pair_distance_pairs(mol, None)
  # Everything is connected at this distance
  assert pair_edges.shape == (2, 9)

  # Test pentane
  mol = Chem.MolFromSmiles('CCCCC')
  # Test distance infinity
  pair_edges = max_pair_distance_pairs(mol, None)
  # Everything is connected at this distance
  assert pair_edges.shape == (2, 25)


def test_weave_single_carbon():
  """Test that single carbon atom is featurized properly."""
  mols = ['C']
  featurizer = dc.feat.WeaveFeaturizer()
  mol_list = featurizer.featurize(mols)
  mol = mol_list[0]

  # Only one carbon
  assert mol.get_num_atoms() == 1

  # Test feature sizes
  assert mol.get_num_features() == 75

  # No bonds, so only 1 pair feature (for the self interaction)
  assert mol.get_pair_features().shape == (1 * 1, 14)


def test_chiral_weave():
  """Test weave features on a molecule with chiral structure."""
  mols = ["F\C=C\F"]  # noqa: W605
  featurizer = dc.feat.WeaveFeaturizer(use_chirality=True)
  mol_list = featurizer.featurize(mols)
  mol = mol_list[0]

  # Only 4 atoms
  assert mol.get_num_atoms() == 4

  # Test feature sizes for chirality
  assert mol.get_num_features() == 78


def test_weave_alkane():
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


def test_weave_alkane_max_pairs():
  """Test on simple alkane with max pairs distance cutoff"""
  mols = ['CCC']
  featurizer = dc.feat.WeaveFeaturizer(max_pair_distance=1)
  # mol_list = featurizer.featurize(mols)
  # mol = mol_list[0]
  from rdkit import Chem
  mol = featurizer._featurize(Chem.MolFromSmiles(mols[0]))

  # 3 carbonds in alkane
  assert mol.get_num_atoms() == 3

  # Test feature sizes
  assert mol.get_num_features() == 75

  # Should be a 7x14 interaction grid since there are 7 pairs within graph
  # distance 1 (3 self interactions plus 2 bonds counted twice because of
  # symmetry)
  assert mol.get_pair_features().shape == (7, 14)


def test_carbon_nitrogen():
  """Test on carbon nitrogen molecule"""
  # Note there is a central nitrogen of degree 4, with 4 carbons
  # of degree 1 (connected only to central nitrogen).
  mols = ['C[N+](C)(C)C']
  # import rdkit.Chem
  # mols = [rdkit.Chem.MolFromSmiles(s) for s in raw_smiles]
  featurizer = dc.feat.WeaveFeaturizer()
  mols = featurizer.featurize(mols)
  mol = mols[0]

  # 5 atoms in compound
  assert mol.get_num_atoms() == 5

  # Test feature sizes
  assert mol.get_num_features() == 75

  # Should be a 3x3 interaction grid
  assert mol.get_pair_features().shape == (5 * 5, 14)
