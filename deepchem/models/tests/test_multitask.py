"""
Integration tests for multitask datasets.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "MIT"

import os
import numpy as np
import tempfile
import shutil
import unittest
import deepchem as dc


class TestMultitask(unittest.TestCase):
  """
  Sanity tests for multitask data.
  """

  def setUp(self):
    super(TestMultitask, self).setUp()
    self.current_dir = os.path.dirname(os.path.abspath(__file__))

  def test_multitask_order(self):
    """Test that order of tasks in multitask datasets is preserved."""
    input_file = os.path.join(self.current_dir, "multitask_example.csv")
    tasks = [
        "task0", "task1", "task2", "task3", "task4", "task5", "task6", "task7",
        "task8", "task9", "task10", "task11", "task12", "task13", "task14",
        "task15", "task16"
    ]

    featurizer = dc.feat.CircularFingerprint(size=1024)

    loader = dc.data.CSVLoader(
        tasks=tasks, smiles_field="smiles", featurizer=featurizer)
    dataset = loader.featurize(input_file)

    splitter = dc.splits.ScaffoldSplitter()
    train_dataset, test_dataset = splitter.train_test_split(dataset)

    assert train_dataset.get_task_names() == tasks
    assert test_dataset.get_task_names() == tasks

  def test_multitask_data(self):
    """Test that data associated with a tasks stays associated with it."""
    tasks = ["task0", "task1"]
    n_samples = 100
    n_features = 3
    n_tasks = len(tasks)

    # Generate dummy dataset
    ids = np.array(["C"] * n_samples, dtype=object)
    X = np.random.rand(n_samples, n_features)
    y = np.random.randint(2, size=(n_samples, n_tasks))
    w = np.ones((n_samples, n_tasks))

    dataset = dc.data.DiskDataset.from_numpy(X, y, w, ids, tasks)
    np.testing.assert_allclose(X, dataset.X)
    np.testing.assert_allclose(y, dataset.y)
    np.testing.assert_allclose(w, dataset.w)
