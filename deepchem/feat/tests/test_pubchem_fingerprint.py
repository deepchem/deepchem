import unittest

from deepchem.feat import PubChemFingerprint


class TestPubChemFingerprint(unittest.TestCase):
    """
    Test PubChemFingerprint.
    """

    def setUp(self):
        """
        Set up tests.
        """
        from rdkit import Chem
        smiles = 'CC(=O)OC1=CC=CC=C1C(=O)O'
        self.mol = Chem.MolFromSmiles(smiles)

    def test_pubchem_fingerprint(self):
        """
        Test simple fingerprint.
        """
        featurizer = PubChemFingerprint()
        feature_sum = featurizer([self.mol])
        assert feature_sum.shape == (1, 881)
