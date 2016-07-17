"""
Testing singletask/multitask dataset shuffling 
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "GPL"

import os
import shutil
import tempfile
import numpy as np
from deepchem.models.tests import TestAPI
from deepchem.utils.save import load_from_disk
from deepchem.featurizers.fingerprints import CircularFingerprint
from deepchem.featurizers.featurize import DataFeaturizer
from deepchem.datasets import Dataset

class TestShuffle(TestAPI):
  """
  Test singletask/multitask dataset shuffling.
  """
  def test_shuffle(self):
    """Test that datasets can be merged."""
    verbosity = "high"
    current_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(self.base_dir, "dataset")

    dataset_file = os.path.join(
        current_dir, "../../models/tests/example.csv")

    featurizer = CircularFingerprint(size=1024)
    tasks = ["log-solubility"]
    loader = DataFeaturizer(tasks=tasks,
                            smiles_field="smiles",
                            featurizer=featurizer,
                            verbosity=verbosity)
    dataset = loader.featurize(
        dataset_file, data_dir, shard_size=2)

    X_orig, y_orig, w_orig, orig_ids = dataset.to_numpy()
    orig_len = len(dataset)

    dataset.shuffle(iterations=5)
    X_new, y_new, w_new, new_ids = dataset.to_numpy()
    
    assert len(dataset) == orig_len
    # The shuffling should have switched up the ordering
    assert not np.array_equal(orig_ids, new_ids)
    # But all the same entries should still be present
    assert sorted(orig_ids) == sorted(new_ids)
    # All the data should have same shape
    assert X_orig.shape == X_new.shape
    assert y_orig.shape == y_new.shape
    assert w_orig.shape == w_new.shape
