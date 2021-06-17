import unittest
from deepchem.feat import MATFeaturizer
import numpy as np


class TestMATFeaturizer(unittest.TestCase):
  """
    Test MATFeaturizer.
    """

  def setUp(self):
    """
        Set up tests.
        """
    from rdkit import Chem
    smiles = 'CC(C)CC(=O)'
    self.mol = Chem.MolFromSmiles(smiles)

  def test_mat_featurizer(self):
    """
        Test featurizer.py
        """
    featurizer = MATFeaturizer()
    out = featurizer.featurize(self.mol)
    assert (type(out) == np.ndarray)
    assert (out.shape == (1, 3, 6, 27))
