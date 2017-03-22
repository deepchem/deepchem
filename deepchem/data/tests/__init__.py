"""
General API for testing dataset objects
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "MIT"

import unittest
import tempfile
import os
import shutil
import numpy as np
import deepchem as dc


def load_solubility_data():
  """Loads solubility dataset"""
  current_dir = os.path.dirname(os.path.abspath(__file__))
  featurizer = dc.feat.CircularFingerprint(size=1024)
  tasks = ["log-solubility"]
  task_type = "regression"
  input_file = os.path.join(current_dir, "../../models/tests/example.csv")
  loader = dc.data.CSVLoader(
      tasks=tasks, smiles_field="smiles", featurizer=featurizer)

  return loader.featurize(input_file)


def load_butina_data():
  """Loads solubility dataset"""
  current_dir = os.path.dirname(os.path.abspath(__file__))
  featurizer = dc.feat.CircularFingerprint(size=1024)
  tasks = ["task"]
  # task_type = "regression"
  input_file = os.path.join(current_dir,
                            "../../models/tests/butina_example.csv")
  loader = dc.data.CSVLoader(
      tasks=tasks, smiles_field="smiles", featurizer=featurizer)

  return loader.featurize(input_file)


def load_multitask_data():
  """Load example multitask data."""
  current_dir = os.path.dirname(os.path.abspath(__file__))
  featurizer = dc.feat.CircularFingerprint(size=1024)
  tasks = [
      "task0", "task1", "task2", "task3", "task4", "task5", "task6", "task7",
      "task8", "task9", "task10", "task11", "task12", "task13", "task14",
      "task15", "task16"
  ]
  input_file = os.path.join(current_dir,
                            "../../models/tests/multitask_example.csv")
  loader = dc.data.CSVLoader(
      tasks=tasks, smiles_field="smiles", featurizer=featurizer)
  return loader.featurize(input_file)


def load_classification_data():
  """Loads classification data from example.csv"""
  current_dir = os.path.dirname(os.path.abspath(__file__))
  featurizer = dc.feat.CircularFingerprint(size=1024)
  tasks = ["outcome"]
  task_type = "classification"
  input_file = os.path.join(current_dir,
                            "../../models/tests/example_classification.csv")
  loader = dc.data.CSVLoader(
      tasks=tasks, smiles_field="smiles", featurizer=featurizer)
  return loader.featurize(input_file)


def load_sparse_multitask_dataset():
  """Load sparse tox multitask data, sample dataset."""
  current_dir = os.path.dirname(os.path.abspath(__file__))
  featurizer = dc.feat.CircularFingerprint(size=1024)
  tasks = [
      "task1", "task2", "task3", "task4", "task5", "task6", "task7", "task8",
      "task9"
  ]
  input_file = os.path.join(current_dir,
                            "../../models/tests/sparse_multitask_example.csv")
  loader = dc.data.CSVLoader(
      tasks=tasks, smiles_field="smiles", featurizer=featurizer)
  return loader.featurize(input_file)


def load_feat_multitask_data():
  """Load example with numerical features, tasks."""
  current_dir = os.path.dirname(os.path.abspath(__file__))
  features = ["feat0", "feat1", "feat2", "feat3", "feat4", "feat5"]
  featurizer = dc.feat.UserDefinedFeaturizer(features)
  tasks = ["task0", "task1", "task2", "task3", "task4", "task5"]
  input_file = os.path.join(current_dir,
                            "../../models/tests/feat_multitask_example.csv")
  loader = dc.data.UserCSVLoader(
      tasks=tasks, featurizer=featurizer, id_field="id")
  return loader.featurize(input_file)


def load_gaussian_cdf_data():
  """Load example with numbers sampled from Gaussian normal distribution.
     Each feature and task is a column of values that is sampled
     from a normal distribution of mean 0, stdev 1."""
  current_dir = os.path.dirname(os.path.abspath(__file__))
  features = ["feat0", "feat1"]
  featurizer = dc.feat.UserDefinedFeaturizer(features)
  tasks = ["task0", "task1"]
  input_file = os.path.join(current_dir,
                            "../../models/tests/gaussian_cdf_example.csv")
  loader = dc.data.UserCSVLoader(
      tasks=tasks, featurizer=featurizer, id_field="id")
  return loader.featurize(input_file)


def load_unlabelled_data():
  current_dir = os.path.dirname(os.path.abspath(__file__))
  featurizer = dc.feat.CircularFingerprint(size=1024)
  tasks = []
  input_file = os.path.join(current_dir, "../../data/tests/no_labels.csv")
  loader = dc.data.CSVLoader(
      tasks=tasks, smiles_field="smiles", featurizer=featurizer)
  return loader.featurize(input_file)
