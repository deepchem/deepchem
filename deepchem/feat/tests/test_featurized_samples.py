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
import deepchem as dc

class TestFeaturizedSamples(unittest.TestCase):
  """
  Test Featurized Samples class.
  """
  def setUp(self):
    super(TestFeaturizedSamples, self).setUp()
    self.current_dir = os.path.dirname(os.path.abspath(__file__))

  def scaffold_test_train_valid_test_split(self):
    """Test of singletask RF ECFP regression API."""
    splittype = "scaffold"
    input_transforms = []
    output_transforms = ["normalize"]
    model_params = {}
    tasks = ["log-solubility"]
    task_type = "regression"
    task_types = {task: task_type for task in tasks}
    input_file = os.path.join(
        self.current_dir, "../../models/tests/example.csv")
    featurizer = dc.feat.CircularFingerprint(size=1024)

    input_file = os.path.join(self.current_dir, input_file)
    loader = dc.load.DataLoader(
        tasks=tasks, smiles_field="smiles",
        featurizer=featurizer, verbosity="low")

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
    input_file = os.path.join(
        self.current_dir, "../../models/tests/example.csv")
    featurizer = dc.feat.CircularFingerprint(size=1024)

    input_file = os.path.join(self.current_dir, input_file)
    loader = dc.load.DataLoader(
        tasks=tasks, smiles_field="smiles",
        featurizer=featurizer, verbosity="low")

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
    input_file = os.path.join(
        self.current_dir, "../../models/tests/example.csv")
    featurizer = dc.feat.CircularFingerprint(size=1024)

    input_file = os.path.join(self.current_dir, input_file)
    loader = dc.load.DataLoader(
        tasks=tasks, smiles_field="smiles",
        featurizer=featurizer, verbosity="low")

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
    input_file = os.path.join(
        self.current_dir, "../../models/tests/example.csv")
    featurizer = dc.feat.CircularFingerprint(size=1024)
    loader = dc.load.DataLoader(
        tasks=tasks, smiles_field="smiles",
        featurizer=featurizer, verbosity="low")

    dataset = loader.featurize(input_file)

    # Splits featurized samples into train/test
    splitter = dc.splits.RandomSplitter()
    train_dataset, test_dataset = splitter.train_test_split(dataset)
    assert len(train_dataset) == 8
    assert len(test_dataset) == 2

  def test_samples_move(self):
    """Test that featurized samples can be moved and reloaded."""
    verbosity = "high"
    base_dir = tempfile.mkdtemp()
    data_dir = os.path.join(base_dir, "data")
    moved_data_dir = os.path.join(base_dir, "moved_data")
    dataset_file = os.path.join(
        self.current_dir, "../../models/tests/example.csv")

    featurizer = dc.feat.CircularFingerprint(size=1024)
    tasks = ["log-solubility"]
    loader = dc.load.DataLoader(
        tasks=tasks, smiles_field="smiles",
        featurizer=featurizer, verbosity=verbosity)
    featurized_dataset = loader.featurize(
        dataset_file, data_dir)
    n_dataset = len(featurized_dataset)
  
    # Now perform move
    shutil.move(data_dir, moved_data_dir)

    moved_featurized_dataset = dc.data.DiskDataset(
        data_dir=moved_data_dir, reload=True)

    assert len(moved_featurized_dataset) == n_dataset
