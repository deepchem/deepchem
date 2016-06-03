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
from deepchem.models.tests import TestAPI

class TestFeaturizedSamples(TestAPI):
  """
  Test Featurized Samples class.
  """

  def random_test_train_valid_test_split_from_sdf(self):
    """Test of singletask CoulombMatrixEig regression on .sdf file."""
    splittype = "random"
    input_transforms = []
    output_transforms = ["normalize"]
    model_params = {}
    tasks = ["atomization_energy"]
    task_type = "regression"
    task_types = {task: task_type for task in tasks}
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(current_dir, "data/water.sdf")

    featurizers = [CoulombMatrixEig(6, remove_hydrogens=False)]

    input_file = os.path.join(self.current_dir, input_file)
    featurizer = DataFeaturizer(tasks=tasks,
                                smiles_field=self.smiles_field,
                                mol_field="mol",
                                featurizers=featurizers,
                                verbosity="low")

    dataset = featurizer.featurize(input_file, self.data_dir, shard_size=None)

    # Splits featurized samples into train/test
    splitter = RandomSplitter()
    train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
        dataset, self.train_dir, self.valid_dir, self.test_dir)
    assert len(train_dataset) == 8
    assert len(valid_dataset) == 1
    assert len(test_dataset) == 1

