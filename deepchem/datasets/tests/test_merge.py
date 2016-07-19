"""
Testing singletask/multitask dataset merging
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
from deepchem.featurizers.featurize import DataLoader
from deepchem.datasets import Dataset

class TestMerge(TestAPI):
  """
  Test singletask/multitask dataset merging.
  """
  def test_merge(self):
    """Test that datasets can be merged."""
    verbosity = "high"
    current_dir = os.path.dirname(os.path.realpath(__file__))
    first_data_dir = os.path.join(self.base_dir, "first_dataset")
    second_data_dir = os.path.join(self.base_dir, "second_dataset")
    merged_data_dir = os.path.join(self.base_dir, "merged_data")

    dataset_file = os.path.join(
        current_dir, "../../models/tests/example.csv")

    featurizer = CircularFingerprint(size=1024)
    tasks = ["log-solubility"]
    loader = DataLoader(tasks=tasks,
                        smiles_field="smiles",
                        featurizer=featurizer,
                        verbosity=verbosity)
    first_dataset = loader.featurize(
        dataset_file, first_data_dir)
    second_dataset = loader.featurize(
        dataset_file, second_data_dir)

    merged_dataset = Dataset.merge(
        merged_data_dir, [first_dataset, second_dataset])

    assert len(merged_dataset) == len(first_dataset) + len(second_dataset)

  def test_subset(self):
    """Tests that subsetting of datasets works."""
    verbosity = "high"
    current_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(self.base_dir, "dataset")
    subset_dir = os.path.join(self.base_dir, "subset")

    dataset_file = os.path.join(
        current_dir, "../../models/tests/example.csv")

    featurizer = CircularFingerprint(size=1024)
    tasks = ["log-solubility"]
    loader = DataLoader(tasks=tasks,
                        smiles_field="smiles",
                        featurizer=featurizer,
                        verbosity=verbosity)
    dataset = loader.featurize(
        dataset_file, data_dir, shard_size=2)

    shard_nums = [1, 2]

    orig_ids = dataset.get_ids()
    _, _, _, ids_1 = dataset.get_shard(1)
    _, _, _, ids_2 = dataset.get_shard(2)

    subset = dataset.subset(subset_dir, shard_nums)
    after_ids = dataset.get_ids()

    assert len(subset) == 4
    assert sorted(subset.get_ids()) == sorted(np.concatenate([ids_1, ids_2]))
    assert list(orig_ids) == list(after_ids)
