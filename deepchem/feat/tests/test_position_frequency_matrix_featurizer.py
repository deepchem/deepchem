import unittest
import numpy as np
from deepchem.feat.sequence_featurizers.position_frequency_matrix_featurizer import PFMFeaturizer, CHARSET, PFM_to_PPM


class TestPFMFeaturizer(unittest.TestCase):
    """
    Test PFMFeaturizer.
    """

    def setUp(self):
        """
        Set up test.
        """
        self.msa = np.array([['ABC', 'BCD'], ['AAA', 'AAB']])
        self.featurizer = PFMFeaturizer()
        self.max_length = 100

    def test_PFMFeaturizer_arbitrary(self):
        """
        Test PFM featurizer for simple MSA.
        """
        pfm = self.featurizer.featurize(self.msa)
        assert pfm.shape == (2, len(CHARSET) + 1, self.max_length)
        assert pfm[0][0][0] == 1

    def test_PFM_to_PPM(self):
        """
        Test PFM_to_PPM.
        """
        pfm = self.featurizer.featurize(self.msa)
        ppm = PFM_to_PPM(pfm[0])
        assert ppm.shape == (len(CHARSET) + 1, self.max_length)
        assert ppm[0][0] == .5
