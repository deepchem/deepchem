"""
Tests for FeaturizedSamples class
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "GPL"

import os
import unittest
import tempfile
import shutil
from deepchem.datasets import Dataset
from deepchem.models.tests import TestAPI
from deepchem.splits import RandomSplitter
from deepchem.splits import ScaffoldSplitter
from deepchem.splits import SpecifiedSplitter
from deepchem.featurizers.featurize import DataFeaturizer
from deepchem.featurizers.fingerprints import CircularFingerprint
#from deepchem.featurizers.featurize import FeaturizedSamples

class TestFeaturizedSamples(TestAPI):
  """
  Test Featurized Samples class.
  """

  def scaffold_test_train_valid_test_split(self):
    """Test of singletask RF ECFP regression API."""
    splittype = "scaffold"
    input_transforms = []
    output_transforms = ["normalize"]
    model_params = {}
    tasks = ["log-solubility"]
    task_type = "regression"
    task_types = {task: task_type for task in tasks}
    input_file = os.path.join(self.current_dir, "example.csv")
    featurizers = [CircularFingerprint(size=1024)]

    input_file = os.path.join(self.current_dir, input_file)
    featurizer = DataFeaturizer(tasks=tasks,
                                smiles_field=self.smiles_field,
                                featurizers=featurizers,
                                verbosity="low")

    dataset = featurizer.featurize(input_file, self.data_dir)

    # Splits featurized samples into train/test
    splitter = ScaffoldSplitter()
    train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
        dataset, self.train_dir, self.valid_dir, self.test_dir)
    assert len(train_dataset) == 8
    assert len(valid_dataset) == 1
    assert len(test_dataset) == 1

  def scaffold_test_train_test_split(self):
    """Test of singletask RF ECFP regression API."""
    splittype = "scaffold"
    input_transforms = []
    output_transforms = ["normalize"]
    model_params = {}
    tasks = ["log-solubility"]
    task_type = "regression"
    task_types = {task: task_type for task in tasks}
    input_file = os.path.join(self.current_dir, "example.csv")
    featurizers = [CircularFingerprint(size=1024)]

    input_file = os.path.join(self.current_dir, input_file)
    featurizer = DataFeaturizer(tasks=tasks,
                                smiles_field=self.smiles_field,
                                featurizers=featurizers,
                                verbosity="low")

    dataset = featurizer.featurize(input_file, self.data_dir)

    # Splits featurized samples into train/test
    splitter = ScaffoldSplitter()
    train_dataset, test_dataset = splitter.train_test_split(
        dataset, self.train_dir, self.test_dir)
    assert len(train_dataset) == 8
    assert len(test_dataset) == 2

  def random_test_train_valid_test_split(self):
    """Test of singletask RF ECFP regression API."""
    input_transforms = []
    output_transforms = ["normalize"]
    model_params = {}
    tasks = ["log-solubility"]
    task_type = "regression"
    task_types = {task: task_type for task in tasks}
    input_file = os.path.join(self.current_dir, "example.csv")
    featurizers = [CircularFingerprint(size=1024)]

    input_file = os.path.join(self.current_dir, input_file)
    featurizer = DataFeaturizer(tasks=tasks,
                                smiles_field=self.smiles_field,
                                featurizers=featurizers,
                                verbosity="low")

    dataset = featurizer.featurize(input_file, self.data_dir)

    # Splits featurized samples into train/test
    splitter = RandomSplitter()
    train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
        dataset, self.train_dir, self.valid_dir, self.test_dir)
    assert len(train_dataset) == 8
    assert len(valid_dataset) == 1
    assert len(test_dataset) == 1

  def random_test_train_test_split(self):
    """Test of singletask RF ECFP regression API."""
    #splittype = "random"
    model_params = {}
    tasks = ["log-solubility"]
    task_type = "regression"
    task_types = {task: task_type for task in tasks}
    input_file = os.path.join(self.current_dir, "example.csv")
    featurizers = [CircularFingerprint(size=1024)]
    featurizer = DataFeaturizer(tasks=tasks,
                                smiles_field=self.smiles_field,
                                featurizers=featurizers,
                                verbosity="low")

    dataset = featurizer.featurize(input_file, self.data_dir)

    # Splits featurized samples into train/test
    splitter = RandomSplitter()
    train_dataset, test_dataset = splitter.train_test_split(
        dataset, self.train_dir, self.test_dir)
    assert len(train_dataset) == 8
    assert len(test_dataset) == 2

  def test_samples_move(self):
    """Test that featurized samples can be moved and reloaded."""
    verbosity = "high"
    data_dir = os.path.join(self.base_dir, "data")
    moved_data_dir = os.path.join(self.base_dir, "moved_data")
    dataset_file = os.path.join(
        self.current_dir, "example.csv")

    featurizers = [CircularFingerprint(size=1024)]
    tasks = ["log-solubility"]
    featurizer = DataFeaturizer(tasks=tasks,
                                smiles_field="smiles",
                                featurizers=featurizers,
                                verbosity=verbosity)
    featurized_dataset = featurizer.featurize(
        dataset_file, data_dir)
    n_dataset = len(featurized_dataset)
  
    # Now perform move
    shutil.move(data_dir, moved_data_dir)

    moved_featurized_dataset = Dataset(
        data_dir=moved_data_dir, reload=True)

    assert len(moved_featurized_dataset) == n_dataset
