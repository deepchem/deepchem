"""
Testing singletask/multitask data loading capabilities.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "MIT"

import os
import shutil
import unittest
import tempfile
import deepchem as dc
import numpy as np


class TestLoad(unittest.TestCase):
  """
  Test singletask/multitask data loading.
  """

  def test_move_load(self):
    """Test that datasets can be moved and loaded."""
    current_dir = os.path.dirname(os.path.realpath(__file__))
    base_dir = tempfile.mkdtemp()
    data_dir = os.path.join(base_dir, "data")
    moved_data_dir = os.path.join(base_dir, "moved_data")
    dataset_file = os.path.join(current_dir, "../../models/tests/example.csv")

    featurizer = dc.feat.CircularFingerprint(size=1024)
    tasks = ["log-solubility"]
    loader = dc.data.CSVLoader(
        tasks=tasks, smiles_field="smiles", featurizer=featurizer)
    dataset = loader.featurize(dataset_file, data_dir)

    X, y, w, ids = (dataset.X, dataset.y, dataset.w, dataset.ids)
    shutil.move(data_dir, moved_data_dir)

    moved_dataset = dc.data.DiskDataset(moved_data_dir)

    X_moved, y_moved, w_moved, ids_moved = (moved_dataset.X, moved_dataset.y,
                                            moved_dataset.w, moved_dataset.ids)

    np.testing.assert_allclose(X, X_moved)
    np.testing.assert_allclose(y, y_moved)
    np.testing.assert_allclose(w, w_moved)
    np.testing.assert_array_equal(ids, ids_moved)

  def test_multiload(self):
    """Check can re-use featurization for multiple task selections.
    """
    # Only for debug!
    np.random.seed(123)

    current_dir = os.path.dirname(os.path.realpath(__file__))
    ##Make directories to store the raw and featurized datasets.
    data_dir = tempfile.mkdtemp()

    # Load dataset
    print("About to load dataset.")
    dataset_file = os.path.join(current_dir,
                                "../../models/tests/multitask_example.csv")

    # Featurize tox21 dataset
    print("About to featurize dataset.")
    featurizer = dc.feat.CircularFingerprint(size=1024)
    all_tasks = ["task%d" % i for i in range(17)]

    ####### Do featurization
    loader = dc.data.CSVLoader(
        tasks=all_tasks, smiles_field="smiles", featurizer=featurizer)
    dataset = loader.featurize(dataset_file, data_dir)

    # Do train/valid split.
    X_multi, y_multi, w_multi, ids_multi = (dataset.X, dataset.y, dataset.w,
                                            dataset.ids)

    ####### Do singletask load
    y_tasks, w_tasks, = [], []
    dataset = dc.data.DiskDataset(data_dir)
    for ind, task in enumerate(all_tasks):
      print("Processing task %s" % task)

      X_task, y_task, w_task, ids_task = (dataset.X, dataset.y, dataset.w,
                                          dataset.ids)
      y_tasks.append(y_task[:, ind])
      w_tasks.append(w_task[:, ind])

    ################## Do comparison
    for ind, task in enumerate(all_tasks):
      y_multi_task = y_multi[:, ind]
      w_multi_task = w_multi[:, ind]

      y_task = y_tasks[ind]
      w_task = w_tasks[ind]

      np.testing.assert_allclose(y_multi_task.flatten(), y_task.flatten())
      np.testing.assert_allclose(w_multi_task.flatten(), w_task.flatten())

  def test_singletask_matches_multitask_load(self):
    """Check that singletask load and multitask load of dataset are same."""
    # Only for debug!
    np.random.seed(123)

    # Set some global variables up top
    reload = True

    current_dir = os.path.dirname(os.path.realpath(__file__))
    #Make directories to store the raw and featurized datasets.
    data_dir = tempfile.mkdtemp()

    # Load dataset
    print("About to load dataset.")
    dataset_file = os.path.join(current_dir,
                                "../../models/tests/multitask_example.csv")

    # Featurize tox21 dataset
    print("About to featurize dataset.")
    featurizer = dc.feat.CircularFingerprint(size=1024)
    all_tasks = ["task%d" % i for i in range(17)]
    # For debugging purposes
    n_tasks = 17
    tasks = all_tasks[0:n_tasks]

    ####### Do multitask load
    loader = dc.data.CSVLoader(
        tasks=tasks, smiles_field="smiles", featurizer=featurizer)
    dataset = loader.featurize(dataset_file, data_dir)

    # Do train/valid split.
    X_multi, y_multi, w_multi, ids_multi = (dataset.X, dataset.y, dataset.w,
                                            dataset.ids)

    ####### Do singletask load
    y_tasks, w_tasks, ids_tasks = [], [], []
    for task in tasks:
      print("Processing task %s" % task)
      if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
      loader = dc.data.CSVLoader(
          tasks=[task], smiles_field="smiles", featurizer=featurizer)
      dataset = loader.featurize(dataset_file, data_dir)

      X_task, y_task, w_task, ids_task = (dataset.X, dataset.y, dataset.w,
                                          dataset.ids)
      y_tasks.append(y_task)
      w_tasks.append(w_task)
      ids_tasks.append(ids_task)

    ################## Do comparison
    for ind, task in enumerate(tasks):
      y_multi_task = y_multi[:, ind]
      w_multi_task = w_multi[:, ind]

      y_task = y_tasks[ind]
      w_task = w_tasks[ind]
      ids_task = ids_tasks[ind]

      np.testing.assert_allclose(y_multi_task.flatten(), y_task.flatten())
      np.testing.assert_allclose(w_multi_task.flatten(), w_task.flatten())
