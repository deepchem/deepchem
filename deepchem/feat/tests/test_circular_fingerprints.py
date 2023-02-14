"""
Test topological fingerprints.
"""
import unittest
from deepchem.feat import CircularFingerprint
import numpy as np


class TestCircularFingerprint(unittest.TestCase):
    """
    Tests for CircularFingerprint.
    """

    def setUp(self):
        """
        Set up tests.
        """
        from rdkit import Chem
        smiles = 'CC(=O)OC1=CC=CC=C1C(=O)O'
        self.mol = Chem.MolFromSmiles(smiles)

    def test_circular_fingerprints(self):
        """
        Test CircularFingerprint.
        """
        featurizer = CircularFingerprint()
        rval = featurizer([self.mol])
        assert rval.shape == (1, 2048)

        # number of indices, where feature count is more than 1, should be 0
        assert len(np.where(rval[0] > 1.0)[0]) == 0

    def test_count_based_circular_fingerprints(self):
        """
        Test CircularFingerprint with counts-based encoding
        """
        featurizer = CircularFingerprint(is_counts_based=True)
        rval = featurizer([self.mol])
        assert rval.shape == (1, 2048)

        # number of indices where feature count is more than 1
        assert len(np.where(rval[0] > 1.0)[0]) == 8

    def test_circular_fingerprints_with_1024(self):
        """
        Test CircularFingerprint with 1024 size.
        """
        featurizer = CircularFingerprint(size=1024)
        rval = featurizer([self.mol])
        assert rval.shape == (1, 1024)

    def test_sparse_circular_fingerprints(self):
        """
        Test CircularFingerprint with sparse encoding.
        """
        featurizer = CircularFingerprint(sparse=True)
        rval = featurizer([self.mol])
        assert rval.shape == (1,)
        assert isinstance(rval[0], dict)
        assert len(rval[0])

    def test_sparse_circular_fingerprints_with_smiles(self):
        """
        Test CircularFingerprint with sparse encoding and SMILES for each
        fragment.
        """
        featurizer = CircularFingerprint(sparse=True, smiles=True)
        rval = featurizer([self.mol])
        assert rval.shape == (1,)
        assert isinstance(rval[0], dict)
        assert len(rval[0])

        # check for separate count and SMILES entries for each fragment
        for fragment_id, value in rval[0].items():
            assert 'count' in value
            assert 'smiles' in value
