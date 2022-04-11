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
    default_max_length = 100
    msa = np.array([['ABC', 'BCD'], ['AAA', 'AAB']])
    featurizer = PFMFeaturizer(max_length=default_max_length)
    pfm = featurizer.featurize(msa)
    assert pfm.shape == (2, len(CHARSET) + 1, default_max_length)
