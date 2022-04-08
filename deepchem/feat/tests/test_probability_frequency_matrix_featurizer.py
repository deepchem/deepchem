import unittest

import numpy as np

from deepchem.feat.sequence_featurizers.probability_frequency_matrix_featurizer import PFMFeaturizer
from deepchem.feat.sequence_featurizers.probability_frequency_matrix_featurizer import CHARSET


class TestPFMFeaturizer(unittest.TestCase):
  """
  Test PFMFeaturizer.
  """