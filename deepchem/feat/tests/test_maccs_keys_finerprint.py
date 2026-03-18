import unittest

from deepchem.feat import MACCSKeysFingerprint


class TestMACCSKeysFingerprint(unittest.TestCase):
    """
    Test MACCSKeyFingerprint.
    """

    def setUp(self):
        """
        Set up tests.
        """
        from rdkit import Chem
        smiles = 'CC(=O)OC1=CC=CC=C1C(=O)O'
        self.mol = Chem.MolFromSmiles(smiles)

    def test_maccs_key_fingerprint(self):
        """
        Test simple fingerprint.
        """
        featurizer = MACCSKeysFingerprint()
        feature_sum = featurizer([self.mol])
        assert feature_sum.shape == (1, 167)
