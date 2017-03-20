"""
Tests for FeaturizedSamples class
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "MIT"

import os
import unittest
import tempfile
import shutil
import deepchem as dc


class TestDataLoader(unittest.TestCase):
  """
  Test DataLoader 
  """

  def setUp(self):
    super(TestDataLoader, self).setUp()
    self.current_dir = os.path.dirname(os.path.abspath(__file__))

  def unlabelled_test(self):
    input_file = os.path.join(self.current_dir,
                              "../../data/tests/no_labels.csv")
    featurizer = dc.feat.CircularFingerprint(size=1024)
    loader = dc.data.CSVLoader(
        tasks=[], smiles_field="smiles", featurizer=featurizer)
    loader.featurize(input_file)

  def scaffold_test_train_valid_test_split(self):
    """Test of singletask RF ECFP regression API."""
    splittype = "scaffold"
    input_transforms = []
    output_transforms = ["normalize"]
    model_params = {}
    tasks = ["log-solubility"]
    task_type = "regression"
    task_types = {task: task_type for task in tasks}
    input_file = os.path.join(self.current_dir,
                              "../../models/tests/example.csv")
    featurizer = dc.feat.CircularFingerprint(size=1024)

    input_file = os.path.join(self.current_dir, input_file)
    loader = dc.data.CSVLoader(
        tasks=tasks, smiles_field="smiles", featurizer=featurizer)

    dataset = loader.featurize(input_file)

    # Splits featurized samples into train/test
    splitter = dc.splits.ScaffoldSplitter()
    train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
        dataset)
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
    input_file = os.path.join(self.current_dir,
                              "../../models/tests/example.csv")
    featurizer = dc.feat.CircularFingerprint(size=1024)

    input_file = os.path.join(self.current_dir, input_file)
    loader = dc.data.CSVLoader(
        tasks=tasks, smiles_field="smiles", featurizer=featurizer)

    dataset = loader.featurize(input_file)

    # Splits featurized samples into train/test
    splitter = dc.splits.ScaffoldSplitter()
    train_dataset, test_dataset = splitter.train_test_split(dataset)
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
    input_file = os.path.join(self.current_dir,
                              "../../models/tests/example.csv")
    featurizer = dc.feat.CircularFingerprint(size=1024)

    input_file = os.path.join(self.current_dir, input_file)
    loader = dc.data.CSVLoader(
        tasks=tasks, smiles_field="smiles", featurizer=featurizer)

    dataset = loader.featurize(input_file)

    # Splits featurized samples into train/test
    splitter = dc.splits.RandomSplitter()
    train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
        dataset)
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
    input_file = os.path.join(self.current_dir,
                              "../../models/tests/example.csv")
    featurizer = dc.feat.CircularFingerprint(size=1024)
    loader = dc.data.CSVLoader(
        tasks=tasks, smiles_field="smiles", featurizer=featurizer)

    dataset = loader.featurize(input_file)

    # Splits featurized samples into train/test
    splitter = dc.splits.RandomSplitter()
    train_dataset, test_dataset = splitter.train_test_split(dataset)
    assert len(train_dataset) == 8
    assert len(test_dataset) == 2

  def test_log_solubility_dataset(self):
    """Test of loading for simple log-solubility dataset."""
    current_dir = os.path.dirname(os.path.realpath(__file__))
    input_file = "../../models/tests/example.csv"
    input_file = os.path.join(current_dir, input_file)

    tasks = ["log-solubility"]
    smiles_field = "smiles"
    loader = dc.data.CSVLoader(
        tasks=tasks,
        smiles_field="smiles",
        featurizer=dc.feat.CircularFingerprint(size=1024))
    dataset = loader.featurize(input_file)

    assert len(dataset) == 10

  def test_dataset_move(self):
    """Test that dataset can be moved and reloaded."""
    base_dir = tempfile.mkdtemp()
    data_dir = os.path.join(base_dir, "data")
    moved_data_dir = os.path.join(base_dir, "moved_data")
    dataset_file = os.path.join(self.current_dir,
                                "../../models/tests/example.csv")

    featurizer = dc.feat.CircularFingerprint(size=1024)
    tasks = ["log-solubility"]
    loader = dc.data.CSVLoader(
        tasks=tasks, smiles_field="smiles", featurizer=featurizer)
    featurized_dataset = loader.featurize(dataset_file, data_dir)
    n_dataset = len(featurized_dataset)

    # Now perform move
    shutil.move(data_dir, moved_data_dir)

    moved_featurized_dataset = dc.data.DiskDataset(moved_data_dir)

    assert len(moved_featurized_dataset) == n_dataset
