"""
Test Raw Featurizations.
"""
import numpy as np
import unittest
from deepchem.feat.raw_featurizer import RawReactionFeaturizer

class TestRawReactionFeaturizer(unittest.TestCase):
  """
  Test Raw Reaction Featurizer.
  """
  def testRawReactionFeaturizer(self):
    """
    Test featurization of Raw reactions.
    """
    smarts = '[C:1]=[O,N:2]>>[C:1][*:2]'
    featurizer = RawReactionFeaturizer()
    rxn = featurizer._featurize(smarts)
