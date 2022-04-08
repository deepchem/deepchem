import unittest

import numpy as np

from deepchem.feat.sequence_featurizers.position_frequency_matrix_featurizer import PFMFeaturizer
from deepchem.feat.sequence_featurizers.position_frequency_matrix_featurizer import CHARSET


class TestPFMFeaturizer(unittest.TestCase):
  """
  Test PFMFeaturizer.
  """
  def test_PFMFeaturizer_arbitrary(self):
    """
    Test shape of PFM for simple MSA.
    """
    msa = np.array([['ABC', 'BCD'], ['AAA', 'AAB']])
    featurizer = PFMFeaturizer()
    pfm = featurizer.featurize(msa)
    assert pfm.shape == (2, len(CHARSET) + 1, self.max_length)
