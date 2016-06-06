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
from deepchem.featurizers.featurize import DataFeaturizer
from deepchem.datasets import Dataset

class TestMerge(TestAPI):
  """
  Test singletask/multitask dataset merging.
  """
  def test_move_load(self):
    """Test that datasets can be moved and loaded."""
    verbosity = "high"
    current_dir = os.path.dirname(os.path.realpath(__file__))
    first_data_dir = os.path.join(self.base_dir, "first_dataset")
    second_data_dir = os.path.join(self.base_dir, "second_dataset")
    merged_data_dir = os.path.join(self.base_dir, "merged_data")

    dataset_file = os.path.join(
        current_dir, "../../models/tests/example.csv")

    featurizers = [CircularFingerprint(size=1024)]
    tasks = ["log-solubility"]
    featurizer = DataFeaturizer(tasks=tasks,
                                smiles_field="smiles",
                                featurizers=featurizers,
                                verbosity=verbosity)
    first_dataset = featurizer.featurize(
        dataset_file, first_data_dir)
    second_dataset = featurizer.featurize(
        dataset_file, second_data_dir)

    merged_dataset = Dataset.merge(
        merged_data_dir, [first_dataset, second_dataset])

    assert len(merged_dataset) == len(first_dataset) + len(second_dataset)
