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
import deepchem as dc

class TestFeaturizedSamples(unittest.TestCase):
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

    featurizer = dc.feat.CoulombMatrixEig(6, remove_hydrogens=False)
    loader = dc.data.SDFLoader(
        tasks=tasks, smiles_field="smiles",
        mol_field="mol", featurizer=featurizer)

    dataset = loader.featurize(input_file)

    # Splits featurized samples into train/test
    splitter = dc.splits.RandomSplitter()
    train_dataset, valid_dataset, test_dataset = \
        splitter.train_valid_test_split(dataset)
    assert len(train_dataset) == 8
    assert len(valid_dataset) == 1
    assert len(test_dataset) == 1
