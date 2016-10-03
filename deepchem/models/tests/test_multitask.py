"""
Integration tests for multitask datasets.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "GPL"

import os
import numpy as np
import tempfile
import shutil
from deepchem.featurizers.fingerprints import CircularFingerprint
from deepchem.featurizers.featurize import DataLoader
from deepchem.datasets import DiskDataset
from deepchem.models.tests import TestAPI
from deepchem.splits import ScaffoldSplitter

class TestMultitaskData(TestAPI):
  """
  Sanity tests for multitask data.
  """
  def test_multitask_order(self):
    """Test that order of tasks in multitask datasets is preserved."""
    input_file = os.path.join(self.current_dir, "multitask_example.csv")
    tasks = ["task0", "task1", "task2", "task3", "task4", "task5", "task6",
             "task7", "task8", "task9", "task10", "task11", "task12",
             "task13", "task14", "task15", "task16"]

    featurizer = CircularFingerprint(size=1024)

    loader = DataLoader(tasks=tasks,
                        smiles_field=self.smiles_field,
                        featurizer=featurizer,
                        verbosity="low")
    dataset = loader.featurize(input_file, self.data_dir)

    splitter = ScaffoldSplitter()
    train_dataset, test_dataset = splitter.train_test_split(
        dataset, self.train_dir, self.test_dir)
  
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
  
    dataset = DiskDataset.from_numpy(self.train_dir, X, y, w, ids, tasks)
    np.testing.assert_allclose(X, dataset.X)
    np.testing.assert_allclose(y, dataset.y)
    np.testing.assert_allclose(w, dataset.w)
