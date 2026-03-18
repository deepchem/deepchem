import unittest
import deepchem as dc
import numpy as np


class TestDummyFeaturizer(unittest.TestCase):
    """
    Test for DummyFeaturizer.
    """

    def test_featurize(self):
        """
        Test the featurize method on an array of inputs.
        """
        input_array = np.array(
            [[
                "N#C[S-].O=C(CBr)c1ccc(C(F)(F)F)cc1>CCO.[K+]",
                "N#CSCC(=O)c1ccc(C(F)(F)F)cc1"
            ],
             [
                 "C1COCCN1.FCC(Br)c1cccc(Br)n1>CCN(C(C)C)C(C)C.CN(C)C=O.O",
                 "FCC(c1cccc(Br)n1)N1CCOCC1"
             ]])
        featurizer = dc.feat.DummyFeaturizer()
        out = featurizer.featurize(input_array)
        assert isinstance(out, np.ndarray)
        assert (out.shape == input_array.shape)
