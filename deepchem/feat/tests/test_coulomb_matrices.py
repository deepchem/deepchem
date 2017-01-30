"""
Tests for Coulomb matrix calculation.
"""
import numpy as np
import unittest

from rdkit import Chem

from deepchem.feat import coulomb_matrices as cm
from deepchem.utils import conformers


class TestCoulombMatrix(unittest.TestCase):
    """
    Tests for CoulombMatrix.
    """
    def setUp(self):
        """
        Set up tests.
        """
        smiles = 'CC(=O)OC1=CC=CC=C1C(=O)O'
        mol = Chem.MolFromSmiles(smiles)
        engine = conformers.ConformerGenerator(max_conformers=1)
        self.mol = engine.generate_conformers(mol)
        assert self.mol.GetNumConformers() > 0

    def test_coulomb_matrix(self):
        """
        Test CoulombMatrix.
        """
        f = cm.CoulombMatrix(self.mol.GetNumAtoms())
        rval = f([self.mol])
        assert rval.shape == (1, self.mol.GetNumConformers(), self.mol.GetNumAtoms(), self.mol.GetNumAtoms())

    def test_coulomb_matrix_padding(self):
        """
        Test CoulombMatrix with padding.
        """
        max_atoms = self.mol.GetNumAtoms() * 2
        f = cm.CoulombMatrix(max_atoms=max_atoms)
        rval = f([self.mol])
        assert rval.shape == (1, self.mol.GetNumConformers(), max_atoms, max_atoms)

    def test_upper_tri_coulomb_matrix(self):
        """
        Test upper triangular CoulombMatrix.
        """
        f = cm.CoulombMatrix(self.mol.GetNumAtoms(), upper_tri=True)
        rval = f([self.mol])
        size = np.triu_indices(self.mol.GetNumAtoms())[0].size
        assert rval.shape == (1, self.mol.GetNumConformers(), size)

    def test_upper_tri_coulomb_matrix_padding(self):
        """
        Test upper triangular CoulombMatrix with padding.
        """
        f = cm.CoulombMatrix(max_atoms=self.mol.GetNumAtoms() * 2, upper_tri=True)
        rval = f([self.mol])
        size = np.triu_indices(self.mol.GetNumAtoms() * 2)[0].size
        assert rval.shape == (1, self.mol.GetNumConformers(), size)

    def test_coulomb_matrix_no_hydrogens(self):
        """
        Test hydrogen removal.
        """
        mol = Chem.RemoveHs(self.mol)
        assert mol.GetNumAtoms() < self.mol.GetNumAtoms()
        f = cm.CoulombMatrix(max_atoms=mol.GetNumAtoms(),
                             remove_hydrogens=True, upper_tri=True)
        rval = f([self.mol])  # use the version with hydrogens
        size = np.triu_indices(mol.GetNumAtoms())[0].size
        assert rval.shape == (1, mol.GetNumConformers(), size)

    def test_coulomb_matrix_hydrogens(self):
        """
        Test no hydrogen removal.
        """
        f = cm.CoulombMatrix(max_atoms=self.mol.GetNumAtoms(),
                             remove_hydrogens=False, upper_tri=True)
        rval = f([self.mol])
        size = np.triu_indices(self.mol.GetNumAtoms())[0].size
        assert rval.shape == (1, self.mol.GetNumConformers(), size)

class TestCoulombMatrixEig(unittest.TestCase):
    """
    Tests for CoulombMatrixEig.
    """
    def setUp(self):
        """
        Set up tests.
        """
        smiles = '[H]C([H])([H])[H]'
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        engine = conformers.ConformerGenerator(max_conformers=1)
        self.mol = engine.generate_conformers(mol)
        assert self.mol.GetNumConformers() > 0

    def test_coulomb_matrix_eig(self):
        """
        Test CoulombMatrixEig.
        """
        f = cm.CoulombMatrixEig(self.mol.GetNumAtoms())
        rval = f([self.mol])
        assert rval.shape == (1, self.mol.GetNumConformers(), self.mol.GetNumAtoms())

    def test_coulomb_matrix_eig_padding(self):
        """
        Test padding of CoulombMatixEig
        """
        self.max_atoms = 29
        f = cm.CoulombMatrixEig(self.max_atoms)
        rval = f([self.mol])
        assert rval.shape == (1, self.mol.GetNumConformers(), self.max_atoms)
