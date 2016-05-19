"""
Tests for importing .sdf files
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Joseph Gomes"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "LGPL"

import os
import unittest
import tempfile
import shutil
from deepchem.splits import RandomSplitter
from deepchem.featurizers.featurize import DataFeaturizer
from deepchem.featurizers.coulomb_matrices import CoulombMatrixEig

class TestFeaturizedSamples(unittest.TestCase):
  """
  Test Featurized Samples class.
  """
  def setUp(self):
    self.current_dir = os.path.dirname(os.path.abspath(__file__))
    self.smiles_field = "smiles"
    self.mol_field = "mol"
    self.feature_dir = tempfile.mkdtemp()
    self.samples_dir = tempfile.mkdtemp()
    self.train_dir = tempfile.mkdtemp()
    self.valid_dir = tempfile.mkdtemp()
    self.test_dir = tempfile.mkdtemp()

  def _featurize_train_valid_test_split(self, splittype, input_file, tasks,
                                        frac_train, frac_valid, frac_test):
    # Featurize input
    compound_featurizers = [CoulombMatrixEig(6, remove_hydrogens=False)]
    complex_featurizers = []
    featurizers = compound_featurizers + complex_featurizers

    input_file = os.path.join(self.current_dir, input_file)
    featurizer = DataFeaturizer(tasks=tasks,
                                smiles_field=self.smiles_field,
                                mol_field=self.mol_field,
                                compound_featurizers=compound_featurizers,
                                complex_featurizers=complex_featurizers,
                                verbosity="low")

    #Featurizes samples and transforms them into NumPy arrays suitable for ML.
    #returns an instance of class FeaturizedSamples()

    samples = featurizer.featurize(input_file, self.feature_dir, self.samples_dir)

    # Splits featurized samples into train/test
    splitter = RandomSplitter()
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

  def random_test_train_valid_test_split_from_sdf(self):
    """Test of singletask RF ECFP regression API when reading from .sdf file."""
    splittype = "random"
    input_transforms = []
    output_transforms = ["normalize"]
    model_params = {}
    tasks = ["atomization_energy"]
    task_type = "regression"
    task_types = {task: task_type for task in tasks}
    input_file = "data/water.sdf"
    train_samples, valid_samples, test_samples = (
        self._featurize_train_valid_test_split(
            splittype, input_file, tasks, frac_train=.8,
            frac_valid=.1, frac_test=.1))
    assert len(train_samples) == 8
    assert len(valid_samples) == 1
    assert len(test_samples) == 1

