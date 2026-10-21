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

    def test_PFMFeaturizer_custom_charset(self):
        """
        Test that a user-supplied charset is used for encoding.
        """
        charset = ['A', 'C', 'G', 'T']
        featurizer = PFMFeaturizer(charset=charset, max_length=4)
        msa = np.array([['ACGT', 'ACGA']])
        pfm = featurizer.featurize(msa)
        # rows: one per charset entry plus the unknown row
        assert pfm.shape == (1, len(charset) + 1, 4)
        expected = np.array([
            [2., 0., 0., 1.],  # A
            [0., 2., 0., 0.],  # C
            [0., 0., 2., 0.],  # G
            [0., 0., 0., 1.],  # T
            [0., 0., 0., 0.],  # unknown
        ])
        np.testing.assert_array_equal(pfm[0], expected)

    def test_PFMFeaturizer_custom_charset_unknown(self):
        """
        Test that characters outside a custom charset land in the unknown row.
        """
        featurizer = PFMFeaturizer(charset=['A', 'C', 'G', 'T'], max_length=3)
        pfm = featurizer.featurize(np.array([['ANA']]))
        assert pfm[0][0][0] == 1
        assert pfm[0][-1][1] == 1  # 'N' is not in the charset

    def test_PFM_to_PPM(self):
        """
        Test PFM_to_PPM.
        """
        pfm = self.featurizer.featurize(self.msa)
        ppm = PFM_to_PPM(pfm[0])
        assert ppm.shape == (len(CHARSET) + 1, self.max_length)
        assert ppm[0][0] == .5
