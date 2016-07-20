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
from deepchem.datasets import Dataset
from deepchem.models.tests import TestAPI
from deepchem.splits import ScaffoldSplitter

class TestMultitaskData(TestAPI):
  """
  Sanity tests for multitask data.
  """
  def test_multitask_order(self):
    """Test that order of tasks in multitask datasets is preserved."""
    from deepchem.models.keras_models.fcnet import MultiTaskDNN
    splittype = "scaffold"
    output_transformers = []
    input_transformers = []
    task_type = "classification"
    # TODO(rbharath): There should be some automatic check to ensure that all
    # required model_params are specified.
    model_params = {"nb_hidden": 10, "activation": "relu",
                    "dropout": .5, "learning_rate": .01,
                    "momentum": .9, "nesterov": False,
                    "decay": 1e-4, "batch_size": 5,
                    "nb_epoch": 2, "init": "glorot_uniform",
                    "nb_layers": 1, "batchnorm": False}

    input_file = os.path.join(self.current_dir, "multitask_example.csv")
    tasks = ["task0", "task1", "task2", "task3", "task4", "task5", "task6",
             "task7", "task8", "task9", "task10", "task11", "task12",
             "task13", "task14", "task15", "task16"]
    task_types = {task: task_type for task in tasks}

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
  
    dataset = Dataset.from_numpy(self.train_dir, X, y, w, ids, tasks)
    X_out, y_out, w_out, _ = dataset.to_numpy()
    np.testing.assert_allclose(X, X_out)
    np.testing.assert_allclose(y, y_out)
    np.testing.assert_allclose(w, w_out)
