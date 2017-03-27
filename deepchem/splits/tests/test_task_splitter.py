"""
Tests for splitter objects.
"""
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

__author__ = "Bharath Ramsundar, Aneesh Pappu"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "MIT"

import unittest
import tempfile
import numpy as np
import deepchem as dc


class TestTaskSplitters(unittest.TestCase):
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
    p = .05  # proportion actives
    y = np.random.binomial(1, p, size=(n_samples, n_tasks))
    dataset = dc.data.NumpyDataset(X, y)

    task_splitter = dc.splits.TaskSplitter()
    train, valid, test = task_splitter.train_valid_test_split(
        dataset, frac_train=.4, frac_valid=.3, frac_test=.3)

    assert len(train.get_task_names()) == 4
    assert len(valid.get_task_names()) == 3
    assert len(test.get_task_names()) == 3

  def test_multitask_K_fold_split(self):
    """
    Test TaskSplitter K-fold split on multitask dataset.
    """
    n_samples = 100
    n_features = 10
    n_tasks = 10
    X = np.random.rand(n_samples, n_features)
    p = .05  # proportion actives
    y = np.random.binomial(1, p, size=(n_samples, n_tasks))
    dataset = dc.data.NumpyDataset(X, y)
    K = 5

    task_splitter = dc.splits.TaskSplitter()
    fold_datasets = task_splitter.k_fold_split(dataset, K)

    for fold_dataset in fold_datasets:
      assert len(fold_dataset.get_task_names()) == 2

  def test_uneven_k_fold_split(self):
    """
    Test k-fold-split works when K does not divide n_tasks.
    """
    n_samples = 100
    n_features = 10
    n_tasks = 17
    X = np.random.rand(n_samples, n_features)
    p = .05  # proportion actives
    y = np.random.binomial(1, p, size=(n_samples, n_tasks))
    dataset = dc.data.NumpyDataset(X, y)
    K = 4
    task_splitter = dc.splits.TaskSplitter()
    fold_datasets = task_splitter.k_fold_split(dataset, K)

    for fold in range(K - 1):
      fold_dataset = fold_datasets[fold]
      assert len(fold_dataset.get_task_names()) == 4
    assert len(fold_datasets[-1].get_task_names()) == 5

  def test_uneven_train_valid_test_split(self):
    """
    Test train/valid/test split works when proportions don't divide n_tasks.
    """
    n_samples = 100
    n_features = 10
    n_tasks = 11
    X = np.random.rand(n_samples, n_features)
    p = .05  # proportion actives
    y = np.random.binomial(1, p, size=(n_samples, n_tasks))
    dataset = dc.data.NumpyDataset(X, y)

    task_splitter = dc.splits.TaskSplitter()
    train, valid, test = task_splitter.train_valid_test_split(
        dataset, frac_train=.4, frac_valid=.3, frac_test=.3)

    assert len(train.get_task_names()) == 4
    assert len(valid.get_task_names()) == 3
    # Note that the extra task goes to test
    assert len(test.get_task_names()) == 4

  def test_merge_fold_datasets(self):
    """
    Test that (K-1) folds can be merged into train dataset.
    """
    n_samples = 100
    n_features = 10
    n_tasks = 10
    X = np.random.rand(n_samples, n_features)
    p = .05  # proportion actives
    y = np.random.binomial(1, p, size=(n_samples, n_tasks))
    w = np.ones((n_samples, n_tasks))
    dataset = dc.data.NumpyDataset(X, y, w)
    K = 5

    task_splitter = dc.splits.TaskSplitter()
    fold_datasets = task_splitter.k_fold_split(dataset, K)
    # Number tasks per fold
    n_per_fold = 2

    for fold in range(K):
      train_inds = list(set(range(K)) - set([fold]))
      train_fold_datasets = [fold_datasets[ind] for ind in train_inds]
      train_dataset = dc.splits.merge_fold_datasets(train_fold_datasets)

      # Find the tasks that correspond to this test fold
      train_tasks = list(
          set(range(10)) - set(
              range(fold * n_per_fold, (fold + 1) * n_per_fold)))

      # Assert that all arrays look like they should
      np.testing.assert_array_equal(train_dataset.X, X)
      np.testing.assert_array_equal(train_dataset.y, y[:, train_tasks])
      np.testing.assert_array_equal(train_dataset.w, w[:, train_tasks])
      np.testing.assert_array_equal(train_dataset.X, X)
