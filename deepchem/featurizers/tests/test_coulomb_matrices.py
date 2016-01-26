"""
Tests for Coulomb matrix calculation.
"""
import numpy as np
import unittest

from rdkit import Chem

from deepchem.featurizers import coulomb_matrices as cm
from vs_utils.utils.rdkit_utils import conformers


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
        size = np.triu_indices(self.mol.GetNumAtoms())[0].size
        assert rval.shape == (1, self.mol.GetNumConformers(), size)

    def test_coulomb_matrix_padding(self):
        """
        Test CoulombMatrix with padding.
        """
        f = cm.CoulombMatrix(max_atoms=self.mol.GetNumAtoms() * 2)
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
                             remove_hydrogens=True)
        rval = f([self.mol])  # use the version with hydrogens
        size = np.triu_indices(mol.GetNumAtoms())[0].size
        assert rval.shape == (1, mol.GetNumConformers(), size)

    def test_coulomb_matrix_hydrogens(self):
        """
        Test no hydrogen removal.
        """
        f = cm.CoulombMatrix(max_atoms=self.mol.GetNumAtoms(),
                             remove_hydrogens=False)
        rval = f([self.mol])
        size = np.triu_indices(self.mol.GetNumAtoms())[0].size
        assert rval.shape == (1, self.mol.GetNumConformers(), size)
