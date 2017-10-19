"""
Tests for dataset creation
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "MIT"

import unittest
import tempfile
import os
import shutil
import numpy as np
import deepchem as dc


class TestDatasets(unittest.TestCase):
  """
  Test basic top-level API for dataset objects.
  """

  def test_sparsify_and_densify(self):
    """Test that sparsify and densify work as inverses."""
    # Test on identity matrix
    num_samples = 10
    num_features = num_samples
    X = np.eye(num_samples)
    X_sparse = dc.data.sparsify_features(X)
    X_reconstructed = dc.data.densify_features(X_sparse, num_features)
    np.testing.assert_array_equal(X, X_reconstructed)

    # Generate random sparse features dataset
    np.random.seed(123)
    p = .05
    X = np.random.binomial(1, p, size=(num_samples, num_features))
    X_sparse = dc.data.sparsify_features(X)
    X_reconstructed = dc.data.densify_features(X_sparse, num_features)
    np.testing.assert_array_equal(X, X_reconstructed)

    # Test edge case with array of all zeros
    X = np.zeros((num_samples, num_features))
    X_sparse = dc.data.sparsify_features(X)
    X_reconstructed = dc.data.densify_features(X_sparse, num_features)
    np.testing.assert_array_equal(X, X_reconstructed)

  def test_pad_features(self):
    """Test that pad_features pads features correctly."""
    batch_size = 100
    num_features = 10
    num_tasks = 5

    # Test cases where n_samples < 2*n_samples < batch_size
    n_samples = 29
    X_b = np.zeros((n_samples, num_features))

    X_out = dc.data.pad_features(batch_size, X_b)
    assert len(X_out) == batch_size

    # Test cases where n_samples < batch_size
    n_samples = 79
    X_b = np.zeros((n_samples, num_features))
    X_out = dc.data.pad_features(batch_size, X_b)
    assert len(X_out) == batch_size

    # Test case where n_samples == batch_size
    n_samples = 100
    X_b = np.zeros((n_samples, num_features))
    X_out = dc.data.pad_features(batch_size, X_b)
    assert len(X_out) == batch_size

    # Test case for object featurization.
    n_samples = 2
    X_b = np.array([{"a": 1}, {"b": 2}])
    X_out = dc.data.pad_features(batch_size, X_b)
    assert len(X_out) == batch_size

    # Test case for more complicated object featurization
    n_samples = 2
    X_b = np.array([(1, {"a": 1}), (2, {"b": 2})])
    X_out = dc.data.pad_features(batch_size, X_b)
    assert len(X_out) == batch_size

    # Test case with multidimensional data
    n_samples = 50
    num_atoms = 15
    d = 3
    X_b = np.zeros((n_samples, num_atoms, d))
    X_out = dc.data.pad_features(batch_size, X_b)
    assert len(X_out) == batch_size

  def test_pad_batches(self):
    """Test that pad_batch pads batches correctly."""
    batch_size = 100
    num_features = 10
    num_tasks = 5

    # Test cases where n_samples < 2*n_samples < batch_size
    n_samples = 29
    X_b = np.zeros((n_samples, num_features))
    y_b = np.zeros((n_samples, num_tasks))
    w_b = np.zeros((n_samples, num_tasks))
    ids_b = np.zeros((n_samples,))

    X_out, y_out, w_out, ids_out = dc.data.pad_batch(batch_size, X_b, y_b, w_b,
                                                     ids_b)
    assert len(X_out) == len(y_out) == len(w_out) == len(ids_out) == batch_size

    # Test cases where n_samples < batch_size
    n_samples = 79
    X_b = np.zeros((n_samples, num_features))
    y_b = np.zeros((n_samples, num_tasks))
    w_b = np.zeros((n_samples, num_tasks))
    ids_b = np.zeros((n_samples,))

    X_out, y_out, w_out, ids_out = dc.data.pad_batch(batch_size, X_b, y_b, w_b,
                                                     ids_b)
    assert len(X_out) == len(y_out) == len(w_out) == len(ids_out) == batch_size

    # Test case where n_samples == batch_size
    n_samples = 100
    X_b = np.zeros((n_samples, num_features))
    y_b = np.zeros((n_samples, num_tasks))
    w_b = np.zeros((n_samples, num_tasks))
    ids_b = np.zeros((n_samples,))

    X_out, y_out, w_out, ids_out = dc.data.pad_batch(batch_size, X_b, y_b, w_b,
                                                     ids_b)
    assert len(X_out) == len(y_out) == len(w_out) == len(ids_out) == batch_size

    # Test case for object featurization.
    n_samples = 2
    X_b = np.array([{"a": 1}, {"b": 2}])
    y_b = np.zeros((n_samples, num_tasks))
    w_b = np.zeros((n_samples, num_tasks))
    ids_b = np.zeros((n_samples,))
    X_out, y_out, w_out, ids_out = dc.data.pad_batch(batch_size, X_b, y_b, w_b,
                                                     ids_b)
    assert len(X_out) == len(y_out) == len(w_out) == len(ids_out) == batch_size

    # Test case for more complicated object featurization
    n_samples = 2
    X_b = np.array([(1, {"a": 1}), (2, {"b": 2})])
    y_b = np.zeros((n_samples, num_tasks))
    w_b = np.zeros((n_samples, num_tasks))
    ids_b = np.zeros((n_samples,))
    X_out, y_out, w_out, ids_out = dc.data.pad_batch(batch_size, X_b, y_b, w_b,
                                                     ids_b)
    assert len(X_out) == len(y_out) == len(w_out) == len(ids_out) == batch_size

    # Test case with multidimensional data
    n_samples = 50
    num_atoms = 15
    d = 3
    X_b = np.zeros((n_samples, num_atoms, d))
    y_b = np.zeros((n_samples, num_tasks))
    w_b = np.zeros((n_samples, num_tasks))
    ids_b = np.zeros((n_samples,))

    X_out, y_out, w_out, ids_out = dc.data.pad_batch(batch_size, X_b, y_b, w_b,
                                                     ids_b)
    assert len(X_out) == len(y_out) == len(w_out) == len(ids_out) == batch_size

  def test_get_task_names(self):
    """Test that get_task_names returns correct task_names"""
    solubility_dataset = dc.data.tests.load_solubility_data()
    assert solubility_dataset.get_task_names() == ["log-solubility"]

    multitask_dataset = dc.data.tests.load_multitask_data()
    assert sorted(multitask_dataset.get_task_names()) == sorted([
        "task0", "task1", "task2", "task3", "task4", "task5", "task6", "task7",
        "task8", "task9", "task10", "task11", "task12", "task13", "task14",
        "task15", "task16"
    ])

  def test_get_data_shape(self):
    """Test that get_data_shape returns currect data shape"""
    solubility_dataset = dc.data.tests.load_solubility_data()
    assert solubility_dataset.get_data_shape() == (1024,)

    multitask_dataset = dc.data.tests.load_multitask_data()
    assert multitask_dataset.get_data_shape() == (1024,)

  def test_len(self):
    """Test that len(dataset) works."""
    solubility_dataset = dc.data.tests.load_solubility_data()
    assert len(solubility_dataset) == 10

  def test_reshard(self):
    """Test that resharding the dataset works."""
    solubility_dataset = dc.data.tests.load_solubility_data()
    X, y, w, ids = (solubility_dataset.X, solubility_dataset.y,
                    solubility_dataset.w, solubility_dataset.ids)
    assert solubility_dataset.get_number_shards() == 1
    solubility_dataset.reshard(shard_size=1)
    assert solubility_dataset.get_shard_size() == 1
    X_r, y_r, w_r, ids_r = (solubility_dataset.X, solubility_dataset.y,
                            solubility_dataset.w, solubility_dataset.ids)
    assert solubility_dataset.get_number_shards() == 10
    solubility_dataset.reshard(shard_size=10)
    assert solubility_dataset.get_shard_size() == 10
    X_rr, y_rr, w_rr, ids_rr = (solubility_dataset.X, solubility_dataset.y,
                                solubility_dataset.w, solubility_dataset.ids)

    # Test first resharding worked
    np.testing.assert_array_equal(X, X_r)
    np.testing.assert_array_equal(y, y_r)
    np.testing.assert_array_equal(w, w_r)
    np.testing.assert_array_equal(ids, ids_r)

    # Test second resharding worked
    np.testing.assert_array_equal(X, X_rr)
    np.testing.assert_array_equal(y, y_rr)
    np.testing.assert_array_equal(w, w_rr)
    np.testing.assert_array_equal(ids, ids_rr)

  def test_select(self):
    """Test that dataset select works."""
    num_datapoints = 10
    num_features = 10
    num_tasks = 1
    X = np.random.rand(num_datapoints, num_features)
    y = np.random.randint(2, size=(num_datapoints, num_tasks))
    w = np.ones((num_datapoints, num_tasks))
    ids = np.array(["id"] * num_datapoints)
    dataset = dc.data.DiskDataset.from_numpy(X, y, w, ids)

    indices = [0, 4, 5, 8]
    select_dataset = dataset.select(indices)
    X_sel, y_sel, w_sel, ids_sel = (select_dataset.X, select_dataset.y,
                                    select_dataset.w, select_dataset.ids)
    np.testing.assert_array_equal(X[indices], X_sel)
    np.testing.assert_array_equal(y[indices], y_sel)
    np.testing.assert_array_equal(w[indices], w_sel)
    np.testing.assert_array_equal(ids[indices], ids_sel)

  def test_get_shape(self):
    """Test that get_shape works."""
    num_datapoints = 100
    num_features = 10
    num_tasks = 10
    # Generate data
    X = np.random.rand(num_datapoints, num_features)
    y = np.random.randint(2, size=(num_datapoints, num_tasks))
    w = np.random.randint(2, size=(num_datapoints, num_tasks))
    ids = np.array(["id"] * num_datapoints)

    dataset = dc.data.NumpyDataset(X, y, w, ids)

    X_shape, y_shape, w_shape, ids_shape = dataset.get_shape()
    assert X_shape == X.shape
    assert y_shape == y.shape
    assert w_shape == w.shape
    assert ids_shape == ids.shape

  def test_iterbatches(self):
    """Test that iterating over batches of data works."""
    solubility_dataset = dc.data.tests.load_solubility_data()
    batch_size = 2
    data_shape = solubility_dataset.get_data_shape()
    tasks = solubility_dataset.get_task_names()
    for (X_b, y_b, w_b, ids_b) in solubility_dataset.iterbatches(batch_size):
      assert X_b.shape == (batch_size,) + data_shape
      assert y_b.shape == (batch_size,) + (len(tasks),)
      assert w_b.shape == (batch_size,) + (len(tasks),)
      assert ids_b.shape == (batch_size,)

  def test_itersamples_numpy(self):
    """Test that iterating over samples in a NumpyDataset works."""
    num_datapoints = 100
    num_features = 10
    num_tasks = 10
    # Generate data
    X = np.random.rand(num_datapoints, num_features)
    y = np.random.randint(2, size=(num_datapoints, num_tasks))
    w = np.random.randint(2, size=(num_datapoints, num_tasks))
    ids = np.array(["id"] * num_datapoints)
    dataset = dc.data.NumpyDataset(X, y, w, ids)
    for i, (sx, sy, sw, sid) in enumerate(dataset.itersamples()):
      np.testing.assert_array_equal(sx, X[i])
      np.testing.assert_array_equal(sy, y[i])
      np.testing.assert_array_equal(sw, w[i])
      np.testing.assert_array_equal(sid, ids[i])

  def test_itersamples_disk(self):
    """Test that iterating over samples in a DiskDataset works."""
    solubility_dataset = dc.data.tests.load_solubility_data()
    X = solubility_dataset.X
    y = solubility_dataset.y
    w = solubility_dataset.w
    ids = solubility_dataset.ids
    for i, (sx, sy, sw, sid) in enumerate(solubility_dataset.itersamples()):
      np.testing.assert_array_equal(sx, X[i])
      np.testing.assert_array_equal(sy, y[i])
      np.testing.assert_array_equal(sw, w[i])
      np.testing.assert_array_equal(sid, ids[i])

  def test_transform_numpy(self):
    """Test that the transform() method works for NumpyDatasets."""
    num_datapoints = 100
    num_features = 10
    num_tasks = 10

    # Generate data
    X = np.random.rand(num_datapoints, num_features)
    y = np.random.randint(2, size=(num_datapoints, num_tasks))
    w = np.random.randint(2, size=(num_datapoints, num_tasks))
    ids = np.array(["id"] * num_datapoints)
    dataset = dc.data.NumpyDataset(X, y, w, ids)

    # Transform it

    def fn(x, y, w):
      return (2 * x, 1.5 * y, w)

    transformed = dataset.transform(fn)
    np.testing.assert_array_equal(X, dataset.X)
    np.testing.assert_array_equal(y, dataset.y)
    np.testing.assert_array_equal(w, dataset.w)
    np.testing.assert_array_equal(ids, dataset.ids)
    np.testing.assert_array_equal(2 * X, transformed.X)
    np.testing.assert_array_equal(1.5 * y, transformed.y)
    np.testing.assert_array_equal(w, transformed.w)
    np.testing.assert_array_equal(ids, transformed.ids)

  def test_transform_disk(self):
    """Test that the transform() method works for DiskDatasets."""
    dataset = dc.data.tests.load_solubility_data()
    X = dataset.X
    y = dataset.y
    w = dataset.w
    ids = dataset.ids

    # Transform it
    def fn(x, y, w):
      return (2 * x, 1.5 * y, w)

    transformed = dataset.transform(fn)
    np.testing.assert_array_equal(X, dataset.X)
    np.testing.assert_array_equal(y, dataset.y)
    np.testing.assert_array_equal(w, dataset.w)
    np.testing.assert_array_equal(ids, dataset.ids)
    np.testing.assert_array_equal(2 * X, transformed.X)
    np.testing.assert_array_equal(1.5 * y, transformed.y)
    np.testing.assert_array_equal(w, transformed.w)
    np.testing.assert_array_equal(ids, transformed.ids)

  def test_to_numpy(self):
    """Test that transformation to numpy arrays is sensible."""
    solubility_dataset = dc.data.tests.load_solubility_data()
    data_shape = solubility_dataset.get_data_shape()
    tasks = solubility_dataset.get_task_names()
    X, y, w, ids = (solubility_dataset.X, solubility_dataset.y,
                    solubility_dataset.w, solubility_dataset.ids)
    N_samples = len(solubility_dataset)
    N_tasks = len(tasks)

    assert X.shape == (N_samples,) + data_shape
    assert y.shape == (N_samples, N_tasks)
    assert w.shape == (N_samples, N_tasks)
    assert ids.shape == (N_samples,)

  def test_consistent_ordering(self):
    """Test that ordering of labels is consistent over time."""
    solubility_dataset = dc.data.tests.load_solubility_data()

    ids1 = solubility_dataset.ids
    ids2 = solubility_dataset.ids

    assert np.array_equal(ids1, ids2)

  def test_get_statistics(self):
    """Test statistics computation of this dataset."""
    solubility_dataset = dc.data.tests.load_solubility_data()
    X, y, _, _ = (solubility_dataset.X, solubility_dataset.y,
                  solubility_dataset.w, solubility_dataset.ids)
    X_means, y_means = np.mean(X, axis=0), np.mean(y, axis=0)
    X_stds, y_stds = np.std(X, axis=0), np.std(y, axis=0)
    comp_X_means, comp_X_stds, comp_y_means, comp_y_stds = \
      solubility_dataset.get_statistics()
    np.testing.assert_allclose(comp_X_means, X_means)
    np.testing.assert_allclose(comp_y_means, y_means)
    np.testing.assert_allclose(comp_X_stds, X_stds)
    np.testing.assert_allclose(comp_y_stds, y_stds)

  def test_disk_iterate_batch_size(self):
    solubility_dataset = dc.data.tests.load_solubility_data()
    X, y, _, _ = (solubility_dataset.X, solubility_dataset.y,
                  solubility_dataset.w, solubility_dataset.ids)
    batch_sizes = []
    for X, y, _, _ in solubility_dataset.iterbatches(
        3, pad_batches=False, deterministic=True):
      batch_sizes.append(len(X))
    self.assertEqual([3, 3, 3, 1], batch_sizes)

  def test_numpy_iterate_batch_size(self):
    solubility_dataset = dc.data.tests.load_solubility_data()
    X, y, _, _ = (solubility_dataset.X, solubility_dataset.y,
                  solubility_dataset.w, solubility_dataset.ids)
    solubility_dataset = dc.data.NumpyDataset.from_DiskDataset(
        solubility_dataset)
    batch_sizes = []
    for X, y, _, _ in solubility_dataset.iterbatches(
        3, pad_batches=False, deterministic=True):
      batch_sizes.append(len(X))
    self.assertEqual([3, 3, 3, 1], batch_sizes)
