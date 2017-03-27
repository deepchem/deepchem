"""
Testing singletask-to-multitask.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "MIT"

import tempfile
import shutil
import unittest
import numpy as np
import deepchem as dc
from sklearn.linear_model import LogisticRegression


class TestSingletasktoMultitask(unittest.TestCase):
  """
  Test top-level API for singletask_to_multitask ML models.
  """

  #def test_singletask_to_multitask_classification(self):
  #  n_features = 10
  #  n_tasks = 17
  #  tasks = range(n_tasks)
  #  # Define train dataset
  #  n_train = 100
  #  X_train = np.random.rand(n_train, n_features)
  #  y_train = np.random.randint(2, size=(n_train, n_tasks))
  #  w_train = np.ones_like(y_train)
  #  ids_train = ["C"] * n_train
  #  train_dataset = dc.data.DiskDataset.from_numpy(
  #      X_train, y_train, w_train, ids_train)

  #  # Define test dataset
  #  n_test = 10
  #  X_test = np.random.rand(n_test, n_features)
  #  y_test = np.random.randint(2, size=(n_test, n_tasks))
  #  w_test = np.ones_like(y_test)
  #  ids_test = ["C"] * n_test
  #  test_dataset = dc.data.DiskDataset.from_numpy(
  #      X_test, y_test, w_test, ids_test)

  #  classification_metrics = [dc.metrics.Metric(dc.metrics.roc_auc_score)]
  #  def model_builder(model_dir):
  #    sklearn_model = LogisticRegression()
  #    return dc.models.SklearnModel(sklearn_model, model_dir)
  #  multitask_model = dc.models.SingletaskToMultitask(
  #      tasks, model_builder)

  #  # Fit trained model
  #  multitask_model.fit(train_dataset)
  #  multitask_model.save()

  #  # Eval multitask_model on train/test
  #  _ = multitask_model.evaluate(train_dataset, classification_metrics)
  #  _ = multitask_model.evaluate(test_dataset, classification_metrics)

  def test_to_singletask(self):
    """Test that to_singletask works."""
    num_datapoints = 100
    num_features = 10
    num_tasks = 10
    # Generate data
    X = np.random.rand(num_datapoints, num_features)
    y = np.random.randint(2, size=(num_datapoints, num_tasks))
    w = np.random.randint(2, size=(num_datapoints, num_tasks))
    ids = np.array(["id"] * num_datapoints)

    dataset = dc.data.DiskDataset.from_numpy(X, y, w, ids)

    task_dirs = []
    try:
      for task in range(num_tasks):
        task_dirs.append(tempfile.mkdtemp())
      singletask_datasets = dc.models.SingletaskToMultitask._to_singletask(
          dataset, task_dirs)
      for task in range(num_tasks):
        singletask_dataset = singletask_datasets[task]
        X_task, y_task, w_task, ids_task = (
            singletask_dataset.X, singletask_dataset.y, singletask_dataset.w,
            singletask_dataset.ids)
        w_nonzero = w[:, task] != 0
        np.testing.assert_array_equal(X_task, X[w_nonzero != 0])
        np.testing.assert_array_equal(y_task.flatten(),
                                      y[:, task][w_nonzero != 0])
        np.testing.assert_array_equal(w_task.flatten(),
                                      w[:, task][w_nonzero != 0])
        np.testing.assert_array_equal(ids_task, ids[w_nonzero != 0])
    finally:
      # Cleanup
      for task_dir in task_dirs:
        shutil.rmtree(task_dir)
