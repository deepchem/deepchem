"""
Testing singletask/multitask dataset merging
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "MIT"

import os
import shutil
import tempfile
import unittest
import deepchem as dc
import numpy as np


class TestMerge(unittest.TestCase):
  """
  Test singletask/multitask dataset merging.
  """

  def test_merge(self):
    """Test that datasets can be merged."""
    current_dir = os.path.dirname(os.path.realpath(__file__))

    dataset_file = os.path.join(current_dir, "../../models/tests/example.csv")

    featurizer = dc.feat.CircularFingerprint(size=1024)
    tasks = ["log-solubility"]
    loader = dc.data.CSVLoader(
        tasks=tasks, smiles_field="smiles", featurizer=featurizer)
    first_dataset = loader.featurize(dataset_file)
    second_dataset = loader.featurize(dataset_file)

    merged_dataset = dc.data.DiskDataset.merge([first_dataset, second_dataset])

    assert len(merged_dataset) == len(first_dataset) + len(second_dataset)

  def test_subset(self):
    """Tests that subsetting of datasets works."""
    current_dir = os.path.dirname(os.path.realpath(__file__))

    dataset_file = os.path.join(current_dir, "../../models/tests/example.csv")

    featurizer = dc.feat.CircularFingerprint(size=1024)
    tasks = ["log-solubility"]
    loader = dc.data.CSVLoader(
        tasks=tasks, smiles_field="smiles", featurizer=featurizer)
    dataset = loader.featurize(dataset_file, shard_size=2)

    shard_nums = [1, 2]

    orig_ids = dataset.ids
    _, _, _, ids_1 = dataset.get_shard(1)
    _, _, _, ids_2 = dataset.get_shard(2)

    subset = dataset.subset(shard_nums)
    after_ids = dataset.ids

    assert len(subset) == 4
    assert sorted(subset.ids) == sorted(np.concatenate([ids_1, ids_2]))
    assert list(orig_ids) == list(after_ids)
