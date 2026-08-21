import unittest
import deepchem as dc
import numpy as np

class TestDummyFeaturizer(unittest.TestCase):
    """
    Test for DummyFeaturizer.
    """

    def test_featurize(self):
        """
        Test the featurize method on a 2D array of SMILES inputs.
        """
        input_array = np.array(
            [[
                "N#C[S-].O=C(CBr)c1ccc(C(F)(F)F)cc1>CCO.[K+]",
                "N#CSCC(=O)c1ccc(C(F)(F)F)cc1"
            ],
             [
                 "C1COCCN1.FCC(Br)c1cccc(Br)n1>CCN(C(C)C)C(C)C.CN(C)C=O.O",
                 "FCC(c1cccc(Br)n1)N1CCOCC1"
             ]]
        )
        featurizer = dc.feat.DummyFeaturizer(canonicalize=False)
        out = featurizer.featurize(input_array)
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.shape, input_array.shape)
        self.assertTrue(np.array_equal(out, input_array))  # check actual values
    

    #This is to check the canonicalize=True
    def test_featurize_with_canonicalization(self):
        smiles = ["C(C(=O)O)N", "OC(=O)CN"]
        featurizer = dc.feat.DummyFeaturizer(canonicalize=True)
        output = featurizer.featurize(smiles)
        expected = np.array(["NCC(=O)O", "NCC(=O)O"])
        self.assertTrue(np.array_equal(output, expected))


if __name__ == '__main__':
    unittest.main()
