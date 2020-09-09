"""
Tests for Coulomb matrix calculation.
"""
import numpy as np
import unittest

from deepchem.feat import CoulombMatrix, CoulombMatrixEig
from deepchem.utils import conformers


class TestCoulombMatrix(unittest.TestCase):
  """
  Tests for CoulombMatrix.
  """

  def setUp(self):
    """
    Set up tests.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem
    smiles = 'CC(=O)OC1=CC=CC=C1C(=O)O'
    mol = Chem.MolFromSmiles(smiles)
    self.mol_with_no_conf = mol

    # with one conformer
    mol_with_one_conf = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol_with_one_conf, AllChem.ETKDG())
    self.mol_with_one_conf = mol_with_one_conf

    # with multiple conformers
    self.num_confs = 4
    engine = conformers.ConformerGenerator(max_conformers=self.num_confs)
    self.mol_with_multi_conf = engine.generate_conformers(mol)

    # include explicit hydrogens
    self.num_atoms = mol_with_one_conf.GetNumAtoms()
    assert self.num_atoms == 21
    assert self.mol_with_one_conf.GetNumConformers() == 1
    assert self.mol_with_multi_conf.GetNumConformers() == self.num_confs

  def test_coulomb_matrix(self):
    """
    Test CoulombMatrix.
    """
    f = CoulombMatrix(self.num_atoms)
    rval = f([self.mol_with_no_conf])
    assert rval.shape == (1, self.num_atoms, self.num_atoms)
    rval = f([self.mol_with_one_conf])
    assert rval.shape == (1, self.num_atoms, self.num_atoms)
    rval = f([self.mol_with_multi_conf])
    assert rval.shape == (1, self.num_confs, self.num_atoms, self.num_atoms)

  def test_coulomb_matrix_padding(self):
    """
    Test CoulombMatrix with padding.
    """
    max_atoms = self.num_atoms * 2
    f = CoulombMatrix(max_atoms=max_atoms)
    rval = f([self.mol_with_no_conf])
    assert rval.shape == (1, max_atoms, max_atoms)
    rval = f([self.mol_with_one_conf])
    assert rval.shape == (1, max_atoms, max_atoms)
    rval = f([self.mol_with_multi_conf])
    assert rval.shape == (1, self.num_confs, max_atoms, max_atoms)

  def test_upper_tri_coulomb_matrix(self):
    """
    Test upper triangular CoulombMatrix.
    """
    f = CoulombMatrix(self.num_atoms, upper_tri=True)
    size = np.triu_indices(self.num_atoms)[0].size
    rval = f([self.mol_with_no_conf])
    assert rval.shape == (1, size)
    rval = f([self.mol_with_one_conf])
    assert rval.shape == (1, size)
    rval = f([self.mol_with_multi_conf])
    assert rval.shape == (1, self.num_confs, size)

  def test_upper_tri_coulomb_matrix_padding(self):
    """
        Test upper triangular CoulombMatrix with padding.
        """
    max_atoms = self.num_atoms * 2
    f = CoulombMatrix(max_atoms=max_atoms, upper_tri=True)
    size = np.triu_indices(max_atoms)[0].size
    rval = f([self.mol_with_no_conf])
    assert rval.shape == (1, size)
    rval = f([self.mol_with_one_conf])
    assert rval.shape == (1, size)
    rval = f([self.mol_with_multi_conf])
    assert rval.shape == (1, self.num_confs, size)

  def test_coulomb_matrix_no_hydrogens(self):
    """
    Test hydrogen removal.
    """
    num_atoms_with_no_H = self.mol_with_no_conf.GetNumAtoms()
    assert num_atoms_with_no_H < self.num_atoms
    f = CoulombMatrix(
        max_atoms=num_atoms_with_no_H, remove_hydrogens=True, upper_tri=True)
    size = np.triu_indices(num_atoms_with_no_H)[0].size
    rval = f([self.mol_with_no_conf])
    assert rval.shape == (1, size)
    rval = f([self.mol_with_one_conf])
    assert rval.shape == (1, size)
    rval = f([self.mol_with_multi_conf])
    assert rval.shape == (1, self.num_confs, size)

  def test_coulomb_matrix_hydrogens(self):
    """
    Test no hydrogen removal.
    """
    f = CoulombMatrix(
        max_atoms=self.num_atoms, remove_hydrogens=False, upper_tri=True)
    size = np.triu_indices(self.num_atoms)[0].size
    rval = f([self.mol_with_no_conf])
    assert rval.shape == (1, size)
    rval = f([self.mol_with_one_conf])
    assert rval.shape == (1, size)
    rval = f([self.mol_with_multi_conf])
    assert rval.shape == (1, self.num_confs, size)


class TestCoulombMatrixEig(unittest.TestCase):
  """
  Tests for CoulombMatrixEig.
  """

  def setUp(self):
    """
    Set up tests.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem
    smiles = 'CC(=O)OC1=CC=CC=C1C(=O)O'
    mol = Chem.MolFromSmiles(smiles)
    self.mol_with_no_conf = mol

    # with one conformer
    mol_with_one_conf = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol_with_one_conf, AllChem.ETKDG())
    self.mol_with_one_conf = mol_with_one_conf

    # with multiple conformers
    self.num_confs = 4
    engine = conformers.ConformerGenerator(max_conformers=self.num_confs)
    self.mol_with_multi_conf = engine.generate_conformers(mol)

    # include explicit hydrogens
    self.num_atoms = mol_with_one_conf.GetNumAtoms()
    assert self.num_atoms == 21
    assert self.mol_with_one_conf.GetNumConformers() == 1
    assert self.mol_with_multi_conf.GetNumConformers() == self.num_confs

  def test_coulomb_matrix_eig(self):
    """
    Test CoulombMatrixEig.
    """
    f = CoulombMatrixEig(self.num_atoms)
    rval = f([self.mol_with_one_conf])
    assert rval.shape == (1, self.num_atoms)
    rval = f([self.mol_with_one_conf])
    assert rval.shape == (1, self.num_atoms)
    rval = f([self.mol_with_multi_conf])
    assert rval.shape == (1, self.num_confs, self.num_atoms)

  def test_coulomb_matrix_eig_padding(self):
    """
        Test padding of CoulombMatixEig
        """
    max_atoms = 2 * self.num_atoms
    f = CoulombMatrixEig(max_atoms=max_atoms)
    rval = f([self.mol_with_one_conf])
    assert rval.shape == (1, max_atoms)
    rval = f([self.mol_with_one_conf])
    assert rval.shape == (1, max_atoms)
    rval = f([self.mol_with_multi_conf])
    assert rval.shape == (1, self.num_confs, max_atoms)
