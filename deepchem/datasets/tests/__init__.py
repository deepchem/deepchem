"""
General API for testing dataset objects
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "GPL"

import unittest
import tempfile
import os
import shutil
import numpy as np
from deepchem.datasets import Dataset
from deepchem.featurizers.featurize import DataLoader
from deepchem.featurizers.fingerprints import CircularFingerprint
from deepchem.transformers import NormalizationTransformer
from deepchem.models.tests import TestAPI

class TestDatasetAPI(TestAPI):
  """
  Shared API for testing with dataset objects. 
  """

  def load_solubility_data(self):
    """Loads solubility data from example.csv"""
    if os.path.exists(self.data_dir):
      shutil.rmtree(self.data_dir)
    featurizer = CircularFingerprint(size=1024)
    tasks = ["log-solubility"]
    task_type = "regression"
    input_file = os.path.join(self.current_dir, "../../models/tests/example.csv")
    featurizer = DataLoader(
        tasks=tasks,
        smiles_field=self.smiles_field,
        featurizer=featurizer,
        verbosity="low")

    return featurizer.featurize(input_file, self.data_dir)

  def load_classification_data(self):
    """Loads classification data from example.csv"""
    if os.path.exists(self.data_dir):
      shutil.rmtree(self.data_dir)
    featurizer = CircularFingerprint(size=1024)
    tasks = ["outcome"]
    task_type = "classification"
    input_file = os.path.join(
        self.current_dir, "../../models/tests/example_classification.csv")
    loader = DataLoader(
        tasks=tasks,
        smiles_field=self.smiles_field,
        featurizer=featurizer,
        verbosity="low")
    return loader.featurize(input_file, self.data_dir)

  def load_multitask_data(self):
    """Load example multitask data."""
    if os.path.exists(self.data_dir):
      shutil.rmtree(self.data_dir)
    featurizer = CircularFingerprint(size=1024)
    tasks = ["task0", "task1", "task2", "task3", "task4", "task5", "task6",
             "task7", "task8", "task9", "task10", "task11", "task12",
             "task13", "task14", "task15", "task16"]
    input_file = os.path.join(
        self.current_dir, "../../models/tests/multitask_example.csv")
    loader = DataLoader(
        tasks=tasks,
        smiles_field=self.smiles_field,
        featurizer=featurizer,
        verbosity="low")
    return loader.featurize(input_file, self.data_dir)
