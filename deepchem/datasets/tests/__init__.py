"""
General API for testing dataset objects
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "LGPL"

import unittest
import tempfile
import os
import shutil
import numpy as np
from deepchem.datasets import Dataset
from deepchem.featurizers.featurize import DataFeaturizer
from deepchem.featurizers.fingerprints import CircularFingerprint
from deepchem.transformers import NormalizationTransformer
from deepchem.splits.tests import TestSplitAPI

class TestDatasetAPI(TestSplitAPI):
  """
  Shared API for testing with dataset objects. 
  """
  # TODO(rbharath): There should be a more natural way to create a dataset
  # object, perhaps just starting from (Xs, ys, ws)
  def _create_dataset(self, compound_featurizers, complex_featurizers,
                      input_transformer_classes, output_transformer_classes,
                      input_file, tasks,
                      protein_pdb_field=None, ligand_pdb_field=None,
                      user_specified_features=None,
                      split_field=None,
                      shard_size=100):
    featurizers = compound_featurizers + complex_featurizers
    samples = self._gen_samples(
        compound_featurizers, complex_featurizers,
        input_transformer_classes, output_transformer_classes,
        input_file, tasks,
        protein_pdb_field=protein_pdb_field,
        ligand_pdb_field=ligand_pdb_field,
        user_specified_features=user_specified_features,
        split_field=split_field,
        shard_size=shard_size)
    use_user_specified_features = (user_specified_features is not None)
    dataset = Dataset(data_dir=self.data_dir, samples=samples, 
                      featurizers=featurizers, tasks=tasks,
                      use_user_specified_features=use_user_specified_features)
    return dataset

  def _load_solubility_data(self):
    """Loads solubility data from example.csv"""
    compound_featurizers = [CircularFingerprint(size=1024)]
    complex_featurizers = []
    input_transformer_classes = []
    output_transformer_classes = []
    tasks = ["log-solubility"]
    task_type = "regression"
    task_types = {task: task_type for task in tasks}
    input_file = "example.csv"
    return self._create_dataset(
        compound_featurizers, complex_featurizers,
        input_transformer_classes, output_transformer_classes,
        input_file, tasks)

  def _load_classification_data(self):
    """Loads classification data from example.csv"""
    compound_featurizers = [CircularFingerprint(size=1024)]
    complex_featurizers = []
    input_transformer_classes = []
    output_transformer_classes = []
    tasks = ["outcome"]
    task_type = "classification"
    task_types = {task: task_type for task in tasks}
    input_file = "example_classification.csv"
    return self._create_dataset(
        compound_featurizers, complex_featurizers,
        input_transformer_classes, output_transformer_classes,
        input_file, tasks)

  def _load_multitask_data(self):
    """Load example multitask data."""
    compound_featurizers = [CircularFingerprint(size=1024)]
    complex_featurizers = []
    output_transformer_classes = []
    input_transformer_classes = []
    tasks = ["task0", "task1", "task2", "task3", "task4", "task5", "task6",
             "task7", "task8", "task9", "task10", "task11", "task12",
             "task13", "task14", "task15", "task16"]
    task_types = {task: "classification" for task in tasks}
    input_file = "multitask_example.csv"
    return self._create_dataset(
        compound_featurizers, complex_featurizers,
        input_transformer_classes, output_transformer_classes,
        input_file, tasks)

