"""
Tests for dataset classes. 
"""
import os
import unittest
import numpy as np
import pandas as pd
import tempfile
import shutil
from deepchem.utils.dataset import FeaturizedSamples
from deepchem.utils.save import save_to_disk

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

def featurized_dataset_from_data(data_df, out_dir):
  """
  Writes featurized data to disk and returns a FeaturizedData object.
  """
  data_loc = os.path.join(out_dir, "data.joblib")
  save_to_disk(data_df, data_loc)
  return FeaturizedSamples(paths=[out_dir])

class TestFeaturizedSamples(unittest.TestCase):
  """
  Test FeaturizedSamples.
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
    dataset = FeaturizedSamples(compound_df=self.compound_df)

    train, test = dataset.train_test_split(splittype="random")
    assert len(train.compound_df) == .8 * len(self.compound_df)
    assert len(test.compound_df) == .2 * len(self.compound_df)
  
    train, test = dataset.train_test_split(splittype="scaffold")
    assert len(train.compound_df) == .8 * len(self.compound_df)
    assert len(test.compound_df) == .2 * len(self.compound_df)

    train, test = dataset.train_test_split(splittype="specified")
    assert len(train.compound_df) == .8 * len(self.compound_df)
    assert len(test.compound_df) == .2 * len(self.compound_df)

  def test_to_arrays(self):
    """
    Basic sanity test of to_arrays function.
    """
    dataset = FeaturizedSamples(compound_df=self.compound_df)
    # Test singletask mode writing runs
    dirpath = tempfile.mkdtemp()
    arrays = dataset.to_arrays(dirpath, "singletask", ["fingerprints"])
    shutil.rmtree(dirpath)

    # Test multitask mode writing runs
    dirpath = tempfile.mkdtemp()
    arrays = dataset.to_arrays(dirpath, "multitask", ["fingerprints"])
    shutil.rmtree(dirpath)

  def test_transform_data(self):
    """
    Basic sanity test of data transforms.
    """
    featurepath = tempfile.mkdtemp()
    dataset = featurized_dataset_from_data(self.compound_df, featurepath)
    # Test normalization transforms. 
    dirpath = tempfile.mkdtemp()
    arrays = dataset.to_arrays(dirpath, "singletask", ["fingerprints"])
    input_transforms = ["normalize"]
    output_transforms = ["normalize"]
    arrays.transform_data(input_transforms, output_transforms)
    shutil.rmtree(dirpath)
    shutil.rmtree(featurepath)
