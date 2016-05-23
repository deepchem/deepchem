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
from deepchem.models.tests import TestAPI
from deepchem.splits import RandomSplitter
from deepchem.splits import ScaffoldSplitter
from deepchem.splits import SpecifiedSplitter
from deepchem.featurizers.featurize import DataFeaturizer
from deepchem.featurizers.fingerprints import CircularFingerprint
from deepchem.featurizers.featurize import FeaturizedSamples

class TestFeaturizedSamples(TestAPI):
  """
  Test Featurized Samples class.
  """
  def _featurize_train_valid_test_split(self, splittype, input_file, tasks,
                                        frac_train, frac_valid, frac_test):
    # Featurize input
    compound_featurizers = [CircularFingerprint(size=1024)]
    complex_featurizers = []
    featurizers = compound_featurizers + complex_featurizers

    input_file = os.path.join(self.current_dir, input_file)
    featurizer = DataFeaturizer(tasks=tasks,
                                smiles_field=self.smiles_field,
                                compound_featurizers=compound_featurizers,
                                complex_featurizers=complex_featurizers,
                                verbosity="low")

    #Featurizes samples and transforms them into NumPy arrays suitable for ML.
    #returns an instance of class FeaturizedSamples()

    samples = featurizer.featurize(input_file, self.feature_dir, self.samples_dir)

    # Splits featurized samples into train/test
    assert splittype in ["random", "specified", "scaffold"]
    if splittype == "random":
      splitter = RandomSplitter()
    elif splittype == "specified":
      splitter = SpecifiedSplitter()
    elif splittype == "scaffold":
      splitter = ScaffoldSplitter()
    if frac_valid > 0:
      train_samples, valid_samples, test_samples = splitter.train_valid_test_split(
          samples, train_dir=self.train_dir, valid_dir=self.valid_dir,
          test_dir=self.test_dir, frac_train=frac_train,
          frac_valid=frac_valid, frac_test=frac_test)

      return train_samples, valid_samples, test_samples
    else:
      train_samples, test_samples = splitter.train_test_split(
          samples, train_dir=self.train_dir, test_dir=self.test_dir,
          frac_train=frac_train)
      return train_samples, test_samples

  def scaffold_test_train_valid_test_split(self):
    """Test of singletask RF ECFP regression API."""
    splittype = "scaffold"
    input_transforms = []
    output_transforms = ["normalize"]
    model_params = {}
    tasks = ["log-solubility"]
    task_type = "regression"
    task_types = {task: task_type for task in tasks}
    input_file = "../../models/tests/example.csv"
    train_samples, valid_samples, test_samples = (
        self._featurize_train_valid_test_split(
            splittype, input_file, tasks, frac_train=.8,
            frac_valid=.1, frac_test=.1))
    assert len(train_samples) == 8
    assert len(valid_samples) == 1
    assert len(test_samples) == 1

  def scaffold_test_train_test_split(self):
    """Test of singletask RF ECFP regression API."""
    splittype = "scaffold"
    input_transforms = []
    output_transforms = ["normalize"]
    model_params = {}
    tasks = ["log-solubility"]
    task_type = "regression"
    task_types = {task: task_type for task in tasks}
    input_file = "../../models/tests/example.csv"
    train_samples, test_samples = (
        self._featurize_train_valid_test_split(
            splittype, input_file, tasks, frac_train=.8,
            frac_valid=0, frac_test=.2))
    assert len(train_samples) == 8
    assert len(test_samples) == 2

  def random_test_train_valid_test_split(self):
    """Test of singletask RF ECFP regression API."""
    splittype = "random"
    input_transforms = []
    output_transforms = ["normalize"]
    model_params = {}
    tasks = ["log-solubility"]
    task_type = "regression"
    task_types = {task: task_type for task in tasks}
    input_file = "../../models/tests/example.csv"
    train_samples, valid_samples, test_samples = (
        self._featurize_train_valid_test_split(
            splittype, input_file, tasks, frac_train=.8,
            frac_valid=.1, frac_test=.1))
    assert len(train_samples) == 8
    assert len(valid_samples) == 1
    assert len(test_samples) == 1

  def random_test_train_test_split(self):
    """Test of singletask RF ECFP regression API."""
    splittype = "random"
    input_transforms = []
    output_transforms = ["normalize"]
    model_params = {}
    tasks = ["log-solubility"]
    task_type = "regression"
    task_types = {task: task_type for task in tasks}
    input_file = "../../models/tests/example.csv"
    train_samples, test_samples = (
        self._featurize_train_valid_test_split(
            splittype, input_file, tasks, frac_train=.8,
            frac_valid=0, frac_test=.2))
    assert len(train_samples) == 8
    assert len(test_samples) == 2

  def test_samples_move(self):
    """Test that featurized samples can be moved and reloaded."""
    verbosity = "high"
    current_dir = os.path.dirname(os.path.realpath(__file__))
    feature_dir = os.path.join(self.base_dir, "features")
    moved_feature_dir = os.path.join(self.base_dir, "moved_features")
    samples_dir = os.path.join(self.base_dir, "samples")
    moved_samples_dir = os.path.join(self.base_dir, "moved_samples")
    dataset_file = os.path.join(
        current_dir, "../../models/tests/example.csv")

    featurizers = [CircularFingerprint(size=1024)]
    tasks = ["log-solubility"]
    featurizer = DataFeaturizer(tasks=tasks,
                                smiles_field="smiles",
                                compound_featurizers=featurizers,
                                verbosity=verbosity)
    featurized_samples = featurizer.featurize(
        dataset_file, feature_dir,
        samples_dir, reload=reload)
    n_samples = len(featurized_samples)
  
    # Now perform move
    shutil.move(feature_dir, moved_feature_dir)
    shutil.move(samples_dir, moved_samples_dir)

    moved_featurized_samples = FeaturizedSamples(
        samples_dir=moved_samples_dir, featurizers=featurizers,
        reload=True)

    assert len(moved_featurized_samples) == n_samples
        
