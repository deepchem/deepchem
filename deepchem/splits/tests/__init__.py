"""
General API for testing splitter objects
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
import unittest
from deepchem.featurizers.featurize import DataFeaturizer
from deepchem.featurizers.fingerprints import CircularFingerprint

class TestSplitAPI(unittest.TestCase):
  """
  Test top-level API for Splitter objects.
  """

  def setUp(self):
    self.current_dir = os.path.dirname(os.path.abspath(__file__))
    self.test_data_dir = os.path.join(self.current_dir, "../../models/tests")
    self.smiles_field = "smiles"
    self.data_dir = tempfile.mkdtemp()
    self.train_dir = tempfile.mkdtemp()
    self.valid_dir = tempfile.mkdtemp()
    self.test_dir = tempfile.mkdtemp()

  def tearDown(self):
    shutil.rmtree(self.data_dir)
    shutil.rmtree(self.train_dir)
    shutil.rmtree(self.valid_dir)
    shutil.rmtree(self.test_dir)

  def load_solubility_data(self):
    """Loads solubility data from example.csv"""
    featurizers = [CircularFingerprint(size=1024)]
    tasks = ["log-solubility"]
    task_type = "regression"
    input_file = os.path.join(self.test_data_dir, "example.csv")
    featurizer = DataFeaturizer(
        tasks=tasks,
        smiles_field=self.smiles_field,
        featurizers=featurizers,
        verbosity="low")

    return featurizer.featurize(input_file, self.data_dir)

  def load_classification_data(self):
    """Loads classification data from example.csv"""
    featurizers = [CircularFingerprint(size=1024)]
    tasks = ["outcome"]
    task_type = "classification"
    input_file = os.path.join(self.test_data_dir, "example_classification.csv")
    featurizer = DataFeaturizer(
        tasks=tasks,
        smiles_field=self.smiles_field,
        featurizers=featurizers,
        verbosity="low")
    return featurizer.featurize(input_file, self.data_dir)

  def load_multitask_data(self):
    """Load example multitask data."""
    featurizers = [CircularFingerprint(size=1024)]
    tasks = ["task0", "task1", "task2", "task3", "task4", "task5", "task6",
             "task7", "task8", "task9", "task10", "task11", "task12",
             "task13", "task14", "task15", "task16"]
    input_file = os.path.join(self.test_data_dir, "multitask_example.csv")
    featurizer = DataFeaturizer(
        tasks=tasks,
        smiles_field=self.smiles_field,
        featurizers=featurizers,
        verbosity="low")
    return featurizer.featurize(input_file, self.data_dir)
