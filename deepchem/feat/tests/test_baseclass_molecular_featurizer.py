from deepchem.feat.base_classes import MolecularFeaturizer
from deepchem.feat import MolGraphConvFeaturizer
from rdkit import Chem
import unittest
import numpy as np


class DummyTestClass(MolecularFeaturizer):
    """
    Dummy test class derived from MolecularFeaturizer where `use_original_atoms_order` parameter is not initialised
    """

    def __init__(self):
        pass

    def _featurize(self, datapoint, **kwargs):
        """
        Returns mapping of atomic number and atom ranks as feature vector (only for testing purposes)
        """
        if isinstance(datapoint, Chem.rdchem.Mol):
            atoms_order = []
            for atom in datapoint.GetAtoms():
                atoms_order.append((atom.GetAtomicNum(), atom.GetIdx()))
            return atoms_order


class DummyTestClass2(MolecularFeaturizer):
    """
    Dummy test class derived from MolecularFeaturizer where `use_original_atoms_order` parameter is initialised
    """

    def __init__(self, use_original_atoms_order=False):
        self.use_original_atoms_order = use_original_atoms_order

    def _featurize(self, datapoint, **kwargs):
        """
        Returns mapping of atomic number and atom ranks as feature vector (only for testing purposes)
        """
        if isinstance(datapoint, Chem.rdchem.Mol):
            atoms_order = []
            for atom in datapoint.GetAtoms():
                atoms_order.append((atom.GetAtomicNum(), atom.GetIdx()))
            return atoms_order


def get_edge_index(mol):
    # construct edge (bond) index
    src, dest = [], []
    for bond in mol.GetBonds():
        # add edge list considering a directed graph
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        src += [start, end]
        dest += [end, start]
    return np.asarray([src, dest], dtype=int)


class TestUpdatedMolecularFeaturizer(unittest.TestCase):
    """
    Test for `use_original_atoms_order` boolean condition added to `MolecularFeaturizer` base class
    """

    def setUp(self):
        """
        Set up tests.
        """
        from rdkit.Chem import rdmolfiles
        from rdkit.Chem import rdmolops
        self.smile = "C1=CC=CN=C1"
        mol = Chem.MolFromSmiles(self.smile)

        self.original_atoms_order = []
        for atom in mol.GetAtoms():
            self.original_atoms_order.append(
                (atom.GetAtomicNum(), atom.GetIdx()
                ))  # mapping of atomic number and original atom ordering

        new_order = rdmolfiles.CanonicalRankAtoms(mol)
        canonical_mol = rdmolops.RenumberAtoms(mol, new_order)
        self.canonical_atoms_order = []
        for atom in canonical_mol.GetAtoms():
            self.canonical_atoms_order.append(
                (atom.GetAtomicNum(), atom.GetIdx()
                ))  # mapping of atomic number and canonical atom ordering

        self.bond_index = get_edge_index(
            mol)  # bond index based on original atom order
        self.canonical_bond_index = get_edge_index(
            canonical_mol)  # bond index based on canonical atom order

    def test_without_init(self):
        """
        Test without use_original_atoms_order being initialised
        """
        featurizer = DummyTestClass()
        datapoint_atoms_order = featurizer.featurize(
            self.smile)  # should be canonical mapping
        assert (datapoint_atoms_order == np.asarray(
            [self.canonical_atoms_order])).all()

    def test_with_canonical_order(self):
        """
        Test with use_original_atoms_order = False
        """
        featurizer = DummyTestClass2(use_original_atoms_order=False)
        datapoint_atoms_order = featurizer.featurize(
            self.smile)  # should be canonical mapping
        assert (datapoint_atoms_order == np.asarray(
            [self.canonical_atoms_order])).all()

    def test_with_original_order(self):
        """
        Test with use_original_atoms_order = True
        """
        featurizer = DummyTestClass2(use_original_atoms_order=True)
        datapoint_atoms_order = featurizer.featurize(
            self.smile)  # should be canonical mapping
        assert (datapoint_atoms_order == np.asarray([self.original_atoms_order
                                                    ])).all()

    def test_on_derived_featurizers(self):
        """
        Test for atom order in featurizer classes derived from 'MolecularFeaturizer' base class
        """
        # test for 'MolGraphConvFeaturizer' class
        featurizer = MolGraphConvFeaturizer()
        graph_feat = featurizer.featurize(self.smile)
        # for "C1=CC=CN=C1" original bond index is not equal to canonical bond index
        assert not (self.bond_index == self.canonical_bond_index).all()
        assert (graph_feat[0].edge_index == self.canonical_bond_index).all()
