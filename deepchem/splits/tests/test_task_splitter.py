
"""
Tests for splitter objects.
"""
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

__author__ = "Bharath Ramsundar, Aneesh Pappu"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "GPL"

import tempfile
import numpy as np
from deepchem.splits.task_splitter import TaskSplitter
from deepchem.datasets import NumpyDataset
from deepchem.datasets.tests import TestDatasetAPI


class TestTaskSplitters(TestDatasetAPI):
  """
  Test some basic splitters.
  """

  def test_multitask_train_valid_test_split(self):
    """
    Test TaskSplitter train/valid/test split on multitask dataset.
    """
    n_samples = 100
    n_features = 10
    n_tasks = 10
    X = np.random.rand(n_samples, n_features)
    p = .05 # proportion actives
    y = np.random.binomial(1, p, size=(n_samples, n_tasks))
    dataset = NumpyDataset(X, y)
    ########################################### DEBUG
    print("dataset")
    print(dataset)
    ########################################### DEBUG

    task_splitter = TaskSplitter()
    train, valid, test = task_splitter.train_valid_test_split(
        dataset, frac_train=.4, frac_valid=.3, frac_test=.3)

    assert len(train.get_task_names()) == 4
    assert len(valid.get_task_names()) == 3
    assert len(test.get_task_names()) == 3
