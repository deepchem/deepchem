"""
Tests for FeaturizedSamples class
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "LGPL"

import os
import unittest
import tempfile
import shutil
from deepchem.featurizers.featurize import DataFeaturizer
from deepchem.featurizers.fingerprints import CircularFingerprint

class TestFeaturizedSamples(unittest.TestCase):
  """
  Test Featurized Samples class.
  """
  def setUp(self):
    self.current_dir = os.path.dirname(os.path.abspath(__file__))
    self.smiles_field = "smiles"
    self.feature_dir = tempfile.mkdtemp()
    self.samples_dir = tempfile.mkdtemp()
    self.train_dir = tempfile.mkdtemp()
    self.valid_dir = tempfile.mkdtemp()
    self.test_dir = tempfile.mkdtemp()

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
                                verbose=True)

    #Featurizes samples and transforms them into NumPy arrays suitable for ML.
    #returns an instance of class FeaturizedSamples()

    samples = featurizer.featurize(input_file, self.feature_dir, self.samples_dir)

    # Splits featurized samples into train/test
    if frac_valid > 0:
      train_samples, valid_samples, test_samples = samples.train_valid_test_split(
          splittype, train_dir=self.train_dir, valid_dir=self.valid_dir,
          test_dir=self.test_dir, frac_train=frac_train,
          frac_valid=frac_valid, frac_test=frac_test)

      return train_samples, valid_samples, test_samples
    else:
      train_samples, test_samples = samples.train_test_split(
          splittype, train_dir=self.train_dir, test_dir=self.test_dir,
          frac_train=frac_train)
      return train_samples, test_samples

  def scaffold_test_train_valid_test_split(self):
    """Test of singletask RF ECFP regression API."""
    splittype = "scaffold"
    input_transforms = []
    output_transforms = ["normalize"]
    model_params = {}
    task_types = {"log-solubility": "regression"}
    input_file = "../../utils/test/example.csv"
    train_samples, valid_samples, test_samples = (
        self._featurize_train_valid_test_split(
            splittype, input_file, task_types.keys(), frac_train=.8,
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
    task_types = {"log-solubility": "regression"}
    input_file = "../../utils/test/example.csv"
    train_samples, test_samples = (
        self._featurize_train_valid_test_split(
            splittype, input_file, task_types.keys(), frac_train=.8,
            frac_valid=0, frac_test=.2))
    assert len(train_samples) == 8
    assert len(test_samples) == 2

  def random_test_train_valid_test_split(self):
    """Test of singletask RF ECFP regression API."""
    splittype = "random"
    input_transforms = []
    output_transforms = ["normalize"]
    model_params = {}
    task_types = {"log-solubility": "regression"}
    input_file = "../../utils/test/example.csv"
    train_samples, valid_samples, test_samples = (
        self._featurize_train_valid_test_split(
            splittype, input_file, task_types.keys(), frac_train=.8,
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
    task_types = {"log-solubility": "regression"}
    input_file = "../../utils/test/example.csv"
    train_samples, test_samples = (
        self._featurize_train_valid_test_split(
            splittype, input_file, task_types.keys(), frac_train=.8,
            frac_valid=0, frac_test=.2))
    assert len(train_samples) == 8
    assert len(test_samples) == 2
