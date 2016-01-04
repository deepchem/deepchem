"""
Tests for dataset classes. 
"""
import unittest
import numpy as np
import pandas as pd
from deep_chem.utils.dataset import FeaturizedDataset

__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2015, Stanford University"
__license__ = "LGPL"

def featurize_compound(smiles, split=None):
  """
  Featurizes a compound given smiles.
  """
  return {"mol_id": smiles,
          "smiles": smiles,
          "split": split,
          "features": np.zeros(10),
          "descriptors": np.zeros(10),
          "fingerprints": np.zeros(10),
          "task": 1.0}

class TestFeaturizedDataset(unittest.TestCase):
  """
  Test FeaturizedDataset.
  """

  def setUp(self):
    """
    Set up tests.
    """
    smiles_strs = ["CC(=O)OC1=CC=CC=C1C(=O)O",
                   "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
                   "CN=C=O",
                   "O=Cc1ccc(O)c(OC)c1",
                   "CC(=O)NCCC1=CNc2c1cc(OC)cc2"]
    train_compounds = [
        featurize_compound(s, split="train") for s in smiles_strs[:-1]]
    test_compounds = [
        featurize_compound(s, split="test") for s in smiles_strs[-1:]]
    self.compound_df = pd.DataFrame(train_compounds + test_compounds)

  
  def test_train_test_split(self):
    """
    Basic sanity test of train/test split.
    """
    dataset = FeaturizedDataset(compound_df=self.compound_df)

    train, test = dataset.train_test_split(splittype="random")
    assert len(train.compound_df) == .8 * len(self.compound_df)
    assert len(test.compound_df) == .2 * len(self.compound_df)
  
    train, test = dataset.train_test_split(splittype="scaffold")
    assert len(train.compound_df) == .8 * len(self.compound_df)
    assert len(test.compound_df) == .2 * len(self.compound_df)

    train, test = dataset.train_test_split(splittype="specified")
    assert len(train.compound_df) == .8 * len(self.compound_df)
    assert len(test.compound_df) == .2 * len(self.compound_df)
