import unittest
import numpy as np
from rdkit import Chem
import deepchem as dc

class TestDummyFeaturizer(unittest.TestCase):
    """
    Tests for the DummyFeaturizer.
    """

    def setUp(self):
        # Sample SMILES strings for testing.
        self.smiles_list = [
            "C1=CC=CC=C1",   # Benzene
            "O=C(O)C",       # Acetic acid (non-canonical, may be standardized)
            "CC(=O)O"        # Acetic acid alternative representation
        ]
        # Create a 2D array of SMILES for shape testing.
        self.smiles_array = np.array([self.smiles_list, self.smiles_list])

    def _get_expected_canonical(self, smiles):
        """
        Compute the expected canonical SMILES using RDKit directly.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string provided: {smiles}")
        return Chem.MolToSmiles(mol, canonical=True)

    def test_featurize_no_canonicalization(self):
        """
        When canonicalize is False, the featurizer should return the input unchanged.
        """
        featurizer = dc.feat.DummyFeaturizer(canonicalize=False)
        output = featurizer.featurize(self.smiles_array)
        # Output should be a numpy array with the same shape as the input.
        self.assertIsInstance(output, np.ndarray)
        self.assertEqual(output.shape, self.smiles_array.shape)
        # Check that the output is identical to the input.
        self.assertTrue(np.array_equal(output, self.smiles_array))

    def test_featurize_with_canonicalization(self):
        """
        When canonicalize is True, the featurizer should return canonical SMILES strings.
        """
        featurizer = dc.feat.DummyFeaturizer(canonicalize=True)
        output = featurizer.featurize(self.smiles_array)
        # Ensure the shape is preserved.
        self.assertEqual(output.shape, self.smiles_array.shape)
        # Compare each element with RDKit's canonicalization.
        for original, result in zip(self.smiles_array.flatten(), output.flatten()):
            expected = self._get_expected_canonical(original)
            self.assertEqual(result, expected)

    def test_invalid_smiles_raises_error(self):
        """
        When canonicalization is enabled, an invalid SMILES string should raise a ValueError.
        """
        invalid_smiles = "invalid_smiles"
        # Create an array that includes one invalid SMILES.
        test_array = np.array([["C1=CC=CC=C1", invalid_smiles]])
        featurizer = dc.feat.DummyFeaturizer(canonicalize=True)
        with self.assertRaises(ValueError):
            # This should raise a ValueError due to the invalid SMILES.
            featurizer.featurize(test_array)

if __name__ == '__main__':
    unittest.main()
