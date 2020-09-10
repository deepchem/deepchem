import unittest

import numpy as np

from deepchem.feat import Mol2VecFingerprint


class TestMol2VecFingerprint(unittest.TestCase):
  """
  Test Mol2VecFingerprint.
  """

  def setUp(self):
    """
    Set up tests.
    """
    from rdkit import Chem
    smiles = 'CC(=O)OC1=CC=CC=C1C(=O)O'
    self.mol = Chem.MolFromSmiles(smiles)

  def test_mol2vec_fingerprint(self):
    """
    Test simple fingerprint.
    """
    featurizer = Mol2VecFingerprint()
    feature_sum = featurizer([self.mol])
    assert feature_sum.shape == (1, 300)
    featurizer = Mol2VecFingerprint(gather_method='mean')
    feature_mean = featurizer([self.mol])
    assert feature_mean.shape == (1, 300)
    assert not np.allclose(feature_sum, feature_mean)
