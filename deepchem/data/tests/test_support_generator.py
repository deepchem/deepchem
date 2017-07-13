"""
Simple Tests for Support Generation 
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Han Altae-Tran and Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "MIT"

import numpy as np
import unittest
import tensorflow as tf
import deepchem as dc


class TestSupports(unittest.TestCase):
  """
  Test that support generation happens properly.
  """

  def test_remove_dead_examples(self):
    """Tests that examples with zero weight are removed."""
    n_samples = 100
    n_features = 3
    n_tasks = 1

    # Generate dummy dataset
    np.random.seed(123)
    p = .05
    ids = np.arange(n_samples)
    X = np.random.rand(n_samples, n_features)
    y = np.random.binomial(1, p, size=(n_samples, n_tasks))
    w = np.random.binomial(1, p, size=(n_samples, n_tasks))

    num_nonzero = np.count_nonzero(np.sum(w, axis=1))

    dataset = dc.data.NumpyDataset(X, y, w, ids)

    cleared_dataset = dc.data.remove_dead_examples(dataset)
    assert len(cleared_dataset) == num_nonzero

  def test_get_task_support_simple(self):
    """Tests that get_task_support samples correctly."""
    n_samples = 20
    n_features = 3
    n_tasks = 1
    n_trials = 10

    # Generate dummy dataset
    np.random.seed(123)
    ids = np.arange(n_samples)
    X = np.random.rand(n_samples, n_features)
    y = np.random.randint(2, size=(n_samples, n_tasks))
    w = np.ones((n_samples, n_tasks))
    dataset = dc.data.NumpyDataset(X, y, w, ids)

    n_episodes = 20
    n_pos = 1
    n_neg = 5
    supports = dc.data.get_task_support(
        dataset, n_episodes, n_pos, n_neg, task=0, log_every_n=10)
    assert len(supports) == n_episodes

    for support in supports:
      assert len(support) == n_pos + n_neg
      assert np.count_nonzero(support.y) == n_pos

  def test_get_task_support_missing(self):
    """Test that task support works in presence of missing data."""
    n_samples = 20
    n_features = 3
    n_tasks = 1
    n_trials = 10

    # Generate dummy dataset
    np.random.seed(123)
    ids = np.arange(n_samples)
    X = np.random.rand(n_samples, n_features)
    y = np.random.randint(2, size=(n_samples, n_tasks))
    w = np.ones((n_samples, n_tasks))
    # Set last n_samples/2 weights to 0
    w[n_samples // 2:] = 0
    dataset = dc.data.NumpyDataset(X, y, w, ids)

    n_episodes = 20
    n_pos = 1
    n_neg = 2
    supports = dc.data.get_task_support(
        dataset, n_episodes, n_pos, n_neg, task=0, log_every_n=10)
    assert len(supports) == n_episodes

    for support in supports:
      assert len(support) == n_pos + n_neg
      assert np.count_nonzero(support.y) == n_pos
      # Check that no support elements are sample from zero-weight samples
      for identifier in support.ids:
        assert identifier < n_samples / 2

  def test_get_task_test(self):
    """Tests that get_task_testsamples correctly."""
    n_samples = 20
    n_features = 3
    n_tasks = 1
    n_trials = 10

    # Generate dummy dataset
    np.random.seed(123)
    ids = np.arange(n_samples)
    X = np.random.rand(n_samples, n_features)
    y = np.random.randint(2, size=(n_samples, n_tasks))
    w = np.ones((n_samples, n_tasks))
    dataset = dc.data.NumpyDataset(X, y, w, ids)

    n_episodes = 20
    n_test = 10
    tests = dc.data.get_task_test(
        dataset, n_episodes, n_test, task=0, log_every_n=10)

    assert len(tests) == n_episodes
    for test in tests:
      assert len(test) == n_test

  def test_simple_support_generator(self):
    """Conducts simple test that support generator runs."""
    n_samples = 20
    n_features = 3
    n_tasks = 1
    n_pos = 1
    n_neg = 5
    n_trials = 10

    # Generate dummy dataset
    np.random.seed(123)
    ids = np.arange(n_samples)
    X = np.random.rand(n_samples, n_features)
    y = np.random.randint(2, size=(n_samples, n_tasks))
    w = np.ones((n_samples, n_tasks))
    dataset = dc.data.NumpyDataset(X, y, w, ids)

    # Create support generator
    supp_gen = dc.data.SupportGenerator(dataset, n_pos, n_neg, n_trials)

  def test_simple_episode_generator(self):
    """Conducts simple test that episode generator runs."""
    n_samples = 20
    n_features = 3
    n_tasks = 1
    n_pos = 1
    n_neg = 5
    n_test = 10
    n_episodes = 10

    # Generate dummy dataset
    np.random.seed(123)
    ids = np.arange(n_samples)
    X = np.random.rand(n_samples, n_features)
    y = np.random.randint(2, size=(n_samples, n_tasks))
    w = np.ones((n_samples, n_tasks))
    dataset = dc.data.NumpyDataset(X, y, w, ids)

    # Create support generator
    episode_gen = dc.data.EpisodeGenerator(dataset, n_pos, n_neg, n_test,
                                           n_episodes)

    n_episodes_found = 0
    for (task, support, test) in episode_gen:
      assert task >= 0
      assert task < n_tasks
      assert len(support) == n_pos + n_neg
      assert np.count_nonzero(support.y) == n_pos
      assert len(test) == n_test
      n_episodes_found += 1
    assert n_episodes_found == n_episodes

  def test_get_task_minus_support_simple(self):
    """Test that fixed index support can be removed from dataset."""
    n_samples = 20
    n_support = 5
    n_features = 3
    n_tasks = 1

    # Generate dummy dataset
    np.random.seed(123)
    ids = np.arange(n_samples)
    X = np.random.rand(n_samples, n_features)
    y = np.random.randint(2, size=(n_samples, n_tasks))
    w = np.ones((n_samples, n_tasks))
    dataset = dc.data.NumpyDataset(X, y, w, ids)

    support_dataset = dc.data.NumpyDataset(X[:n_support], y[:n_support],
                                           w[:n_support], ids[:n_support])

    task_dataset = dc.data.get_task_dataset_minus_support(
        dataset, support_dataset, task=0)

    # Assert all support elements have been removed
    assert len(task_dataset) == n_samples - n_support
    np.testing.assert_array_equal(task_dataset.X, X[n_support:])
    np.testing.assert_array_equal(task_dataset.y, y[n_support:])
    np.testing.assert_array_equal(task_dataset.w, w[n_support:])
    np.testing.assert_array_equal(task_dataset.ids, ids[n_support:])

  def test_dataset_difference_simple(self):
    """Test that fixed index can be removed from dataset."""
    n_samples = 20
    n_remove = 5
    n_features = 3
    n_tasks = 1

    # Generate dummy dataset
    np.random.seed(123)
    ids = np.arange(n_samples)
    X = np.random.rand(n_samples, n_features)
    y = np.random.randint(2, size=(n_samples, n_tasks))
    w = np.ones((n_samples, n_tasks))
    dataset = dc.data.NumpyDataset(X, y, w, ids)

    remove_dataset = dc.data.NumpyDataset(X[:n_remove], y[:n_remove],
                                          w[:n_remove], ids[:n_remove])

    out_dataset = dc.data.dataset_difference(dataset, remove_dataset)

    # Assert all remove elements have been removed
    assert len(out_dataset) == n_samples - n_remove
    np.testing.assert_array_equal(out_dataset.X, X[n_remove:])
    np.testing.assert_array_equal(out_dataset.y, y[n_remove:])
    np.testing.assert_array_equal(out_dataset.w, w[n_remove:])
    np.testing.assert_array_equal(out_dataset.ids, ids[n_remove:])

  def test_get_task_minus_support(self):
    """Test that random index support can be removed from dataset."""
    n_samples = 10
    n_support = 4
    n_features = 3
    n_tasks = 1

    # Generate dummy dataset
    np.random.seed(123)
    ids = np.arange(n_samples)
    X = np.random.rand(n_samples, n_features)
    y = np.random.randint(2, size=(n_samples, n_tasks))
    w = np.ones((n_samples, n_tasks))
    dataset = dc.data.NumpyDataset(X, y, w, ids)

    support_inds = sorted(
        np.random.choice(np.arange(n_samples), (n_support,), replace=False))
    support_dataset = dc.data.NumpyDataset(X[support_inds], y[support_inds],
                                           w[support_inds], ids[support_inds])

    task_dataset = dc.data.get_task_dataset_minus_support(
        dataset, support_dataset, task=0)

    # Assert all support elements have been removed
    data_inds = sorted(list(set(range(n_samples)) - set(support_inds)))
    assert len(task_dataset) == n_samples - n_support
    np.testing.assert_array_equal(task_dataset.X, X[data_inds])
    np.testing.assert_array_equal(task_dataset.y, y[data_inds])
    np.testing.assert_array_equal(task_dataset.w, w[data_inds])
    np.testing.assert_array_equal(task_dataset.ids, ids[data_inds])

  def test_dataset_difference(self):
    """Test that random index can be removed from dataset."""
    n_samples = 10
    n_remove = 4
    n_features = 3
    n_tasks = 1

    # Generate dummy dataset
    np.random.seed(123)
    ids = np.arange(n_samples)
    X = np.random.rand(n_samples, n_features)
    y = np.random.randint(2, size=(n_samples, n_tasks))
    w = np.ones((n_samples, n_tasks))
    dataset = dc.data.NumpyDataset(X, y, w, ids)

    remove_inds = sorted(
        np.random.choice(np.arange(n_samples), (n_remove,), replace=False))
    remove_dataset = dc.data.NumpyDataset(X[remove_inds], y[remove_inds],
                                          w[remove_inds], ids[remove_inds])

    out_dataset = dc.data.dataset_difference(dataset, remove_dataset)

    # Assert all remove elements have been removed
    data_inds = sorted(list(set(range(n_samples)) - set(remove_inds)))
    assert len(out_dataset) == n_samples - n_remove
    np.testing.assert_array_equal(out_dataset.X, X[data_inds])
    np.testing.assert_array_equal(out_dataset.y, y[data_inds])
    np.testing.assert_array_equal(out_dataset.w, w[data_inds])
    np.testing.assert_array_equal(out_dataset.ids, ids[data_inds])

  def test_get_task_minus_support_missing(self):
    """Test that support can be removed from dataset with missing data"""
    n_samples = 20
    n_support = 4
    n_features = 3
    n_tasks = 1

    # Generate dummy dataset
    np.random.seed(123)
    ids = np.arange(n_samples)
    X = np.random.rand(n_samples, n_features)
    y = np.random.randint(2, size=(n_samples, n_tasks))
    w = np.ones((n_samples, n_tasks))
    # Set last n_samples/2 weights to 0
    w[n_samples // 2:] = 0
    dataset = dc.data.NumpyDataset(X, y, w, ids)

    # Sample from first n_samples/2 elements for support
    support_inds = sorted(
        np.random.choice(
            np.arange(n_samples // 2), (n_support,), replace=False))
    support_dataset = dc.data.NumpyDataset(X[support_inds], y[support_inds],
                                           w[support_inds], ids[support_inds])

    task_dataset = dc.data.get_task_dataset_minus_support(
        dataset, support_dataset, task=0)

    # Should lie within first n_samples/2 samples only
    assert len(task_dataset) == n_samples / 2 - n_support
    for identifier in task_dataset.ids:
      assert identifier < n_samples / 2

  def test_support_generator_correct_samples(self):
    """Tests that samples from support generator have desired shape."""
    n_samples = 20
    n_features = 3
    n_tasks = 1
    n_pos = 1
    n_neg = 5
    n_trials = 10

    # Generate dummy dataset
    np.random.seed(123)
    ids = np.arange(n_samples)
    X = np.random.rand(n_samples, n_features)
    y = np.random.randint(2, size=(n_samples, n_tasks))
    w = np.ones((n_samples, n_tasks))
    dataset = dc.data.NumpyDataset(X, y, w, ids)

    # Create support generator
    supp_gen = dc.data.SupportGenerator(dataset, n_pos, n_neg, n_trials)
    num_supports = 0

    for (task, support) in supp_gen:
      assert support.X.shape == (n_pos + n_neg, n_features)
      num_supports += 1
      assert task == 0  # Only one task in this example
      n_supp_pos = np.count_nonzero(support.y)
      assert n_supp_pos == n_pos
    assert num_supports == n_trials

  def test_evaluation_strategy(self):
    """Tests that sampling supports for eval works properly."""
    n_samples = 2000
    n_features = 3
    n_tasks = 5
    n_pos = 1
    n_neg = 5
    n_trials = 10

    # Generate dummy dataset
    np.random.seed(123)
    ids = np.arange(n_samples)
    X = np.random.rand(n_samples, n_features)
    y = np.random.randint(2, size=(n_samples, n_tasks))
    w = np.random.randint(2, size=(n_samples, n_tasks))
    dataset = dc.data.NumpyDataset(X, y, w, ids)

    support_generator = dc.data.SupportGenerator(dataset, n_pos, n_neg,
                                                 n_trials)

    for ind, (task, support) in enumerate(support_generator):
      task_dataset = dc.data.get_task_dataset_minus_support(
          dataset, support, task)

      task_y = dataset.y[:, task]
      task_w = dataset.w[:, task]
      task_y = task_y[task_w != 0]
      assert len(task_y) == len(support) + len(task_dataset)
      print("Verifying that task_dataset doesn't overlap with support.")
      for task_id in task_dataset.ids:
        assert task_id not in set(support.ids)
