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


class TestCircularFingerprintEquality(unittest.TestCase):
    """
    Tests that equality reflects the featurizer's configuration.
    """

    def test_counts_based_and_bit_vector_are_not_equal(self):
        """
        is_counts_based selects a different RDKit call, so it must be part of
        equality. Two featurizers differing only in that flag produce different
        feature vectors, and declaring them equal lets them collapse in a set
        or a dict key.
        """
        bit_vector = CircularFingerprint(size=64, is_counts_based=False)
        counts = CircularFingerprint(size=64, is_counts_based=True)

        assert bit_vector != counts
        assert len({bit_vector, counts}) == 2
        assert len({bit_vector: 1, counts: 2}) == 2

    def test_equal_configuration_is_still_equal_and_hashable(self):
        """
        The fix must not make equal configurations compare unequal, and equal
        objects must still hash equally for the dict/set contract to hold.
        """
        first = CircularFingerprint(size=64, is_counts_based=True)
        second = CircularFingerprint(size=64, is_counts_based=True)

        assert first == second
        assert hash(first) == hash(second)
        assert len({first, second}) == 1

    def test_a_different_flag_changes_the_features(self):
        """
        The reason equality has to distinguish them: the two settings do not
        produce the same numbers.
        """
        bit_vector = CircularFingerprint(size=64, is_counts_based=False)
        counts = CircularFingerprint(size=64, is_counts_based=True)

        as_bits = bit_vector([self.mol_for_equality()])[0]
        as_counts = counts([self.mol_for_equality()])[0]

        assert not np.array_equal(as_bits, as_counts)
        assert as_bits.sum() < as_counts.sum()

    def test_comparison_with_another_featurizer_type_is_false(self):
        """
        Guard the pre-existing isinstance check so the added clause does not
        change it.
        """
        from deepchem.feat import MACCSKeysFingerprint
        assert CircularFingerprint() != MACCSKeysFingerprint()

    @staticmethod
    def mol_for_equality():
        from rdkit import Chem
        return Chem.MolFromSmiles('CC(=O)OC1=CC=CC=C1C(=O)O')
