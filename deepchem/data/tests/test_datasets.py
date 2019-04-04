"""
Tests for dataset creation
"""
from __future__ import division
from __future__ import unicode_literals

__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "MIT"

import random
import math
import unittest
import tempfile
import os
import shutil
import numpy as np
import deepchem as dc
import tensorflow as tf
from tensorflow.python.framework import test_util


class TestDatasets(test_util.TensorFlowTestCase):
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

  def test_complete_shuffle(self):
    shard_sizes = [1, 2, 3, 4, 5]
    batch_size = 10

    all_Xs, all_ys, all_ws, all_ids = [], [], [], []

    def shard_generator():
      for sz in shard_sizes:
        X_b = np.random.rand(sz, 1)
        y_b = np.random.rand(sz, 1)
        w_b = np.random.rand(sz, 1)
        ids_b = np.random.rand(sz)

        all_Xs.append(X_b)
        all_ys.append(y_b)
        all_ws.append(w_b)
        all_ids.append(ids_b)

        yield X_b, y_b, w_b, ids_b

    dataset = dc.data.DiskDataset.create_dataset(shard_generator())

    res = dataset.complete_shuffle()

    # approx 1/15! chance of equality
    np.testing.assert_equal(np.any(np.not_equal(dataset.X, res.X)), True)
    np.testing.assert_equal(np.any(np.not_equal(dataset.y, res.w)), True)
    np.testing.assert_equal(np.any(np.not_equal(dataset.w, res.y)), True)
    np.testing.assert_equal(np.any(np.not_equal(dataset.ids, res.ids)), True)

    np.testing.assert_array_equal(
        np.sort(dataset.X, axis=0), np.sort(res.X, axis=0))
    np.testing.assert_array_equal(
        np.sort(dataset.y, axis=0), np.sort(res.y, axis=0))
    np.testing.assert_array_equal(
        np.sort(dataset.w, axis=0), np.sort(res.w, axis=0))
    np.testing.assert_array_equal(np.sort(dataset.ids), np.sort(res.ids))

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

  def test_disk_pad_batches(self):
    shard_sizes = [21, 11, 41, 21, 51]
    batch_size = 10

    all_Xs, all_ys, all_ws, all_ids = [], [], [], []

    def shard_generator():
      for sz in shard_sizes:
        X_b = np.random.rand(sz, 1)
        y_b = np.random.rand(sz, 1)
        w_b = np.random.rand(sz, 1)
        ids_b = np.random.rand(sz)

        all_Xs.append(X_b)
        all_ys.append(y_b)
        all_ws.append(w_b)
        all_ids.append(ids_b)

        yield X_b, y_b, w_b, ids_b

    dataset = dc.data.DiskDataset.create_dataset(shard_generator())

    all_Xs = np.concatenate(all_Xs, axis=0)
    all_ys = np.concatenate(all_ys, axis=0)
    all_ws = np.concatenate(all_ws, axis=0)
    all_ids = np.concatenate(all_ids, axis=0)

    test_Xs, test_ys, test_ws, test_ids = [], [], [], []
    for bidx, (a, b, c, d) in enumerate(
        dataset.iterbatches(
            batch_size=batch_size, pad_batches=True, deterministic=True)):

      test_Xs.append(a)
      test_ys.append(b)
      test_ws.append(c)
      test_ids.append(d)

    test_Xs = np.concatenate(test_Xs, axis=0)
    test_ys = np.concatenate(test_ys, axis=0)
    test_ws = np.concatenate(test_ws, axis=0)
    test_ids = np.concatenate(test_ids, axis=0)

    total_size = sum(shard_sizes)

    assert bidx == math.ceil(total_size / batch_size) - 1

    expected_batches = math.ceil(total_size / batch_size) * batch_size

    assert len(test_Xs) == expected_batches
    assert len(test_ys) == expected_batches
    assert len(test_ws) == expected_batches
    assert len(test_ids) == expected_batches

    np.testing.assert_array_equal(all_Xs, test_Xs[:total_size, :])
    np.testing.assert_array_equal(all_ys, test_ys[:total_size, :])
    np.testing.assert_array_equal(all_ws, test_ws[:total_size, :])
    np.testing.assert_array_equal(all_ids, test_ids[:total_size])

  def test_disk_iterate_y_w_None(self):
    shard_sizes = [21, 11, 41, 21, 51]
    batch_size = 10

    all_Xs, all_ys, all_ws, all_ids = [], [], [], []

    def shard_generator():
      for sz in shard_sizes:
        X_b = np.random.rand(sz, 1)
        ids_b = np.random.rand(sz)

        all_Xs.append(X_b)
        all_ids.append(ids_b)

        yield X_b, None, None, ids_b

    dataset = dc.data.DiskDataset.create_dataset(shard_generator())

    all_Xs = np.concatenate(all_Xs, axis=0)
    all_ids = np.concatenate(all_ids, axis=0)

    test_Xs, test_ids = [], []
    for bidx, (a, _, _, d) in enumerate(
        dataset.iterbatches(
            batch_size=batch_size, pad_batches=True, deterministic=True)):

      test_Xs.append(a)
      test_ids.append(d)

    test_Xs = np.concatenate(test_Xs, axis=0)
    test_ids = np.concatenate(test_ids, axis=0)

    total_size = sum(shard_sizes)

    assert bidx == math.ceil(total_size / batch_size) - 1

    expected_batches = math.ceil(total_size / batch_size) * batch_size

    assert len(test_Xs) == expected_batches
    assert len(test_ids) == expected_batches

    np.testing.assert_array_equal(all_Xs, test_Xs[:total_size, :])
    np.testing.assert_array_equal(all_ids, test_ids[:total_size])

  def test_disk_iterate_batch(self):

    all_batch_sizes = [None, 32, 17, 11]
    all_shard_sizes = [[7, 3, 12, 4, 5], [1, 1, 1, 1, 1], [31, 31, 31, 31, 31],
                       [21, 11, 41, 21, 51]]

    for idx in range(25):
      shard_length = random.randint(1, 32)
      shard_sizes = []
      for _ in range(shard_length):
        shard_sizes.append(random.randint(1, 128))
      all_shard_sizes.append(shard_sizes)
      if idx == 0:
        # special case to test
        all_batch_sizes.append(None)
      else:
        all_batch_sizes.append(random.randint(1, 256))

    for shard_sizes, batch_size in zip(all_shard_sizes, all_batch_sizes):

      all_Xs, all_ys, all_ws, all_ids = [], [], [], []

      def shard_generator():
        for sz in shard_sizes:
          X_b = np.random.rand(sz, 1)
          y_b = np.random.rand(sz, 1)
          w_b = np.random.rand(sz, 1)
          ids_b = np.random.rand(sz)

          all_Xs.append(X_b)
          all_ys.append(y_b)
          all_ws.append(w_b)
          all_ids.append(ids_b)

          yield X_b, y_b, w_b, ids_b

      dataset = dc.data.DiskDataset.create_dataset(shard_generator())

      all_Xs = np.concatenate(all_Xs, axis=0)
      all_ys = np.concatenate(all_ys, axis=0)
      all_ws = np.concatenate(all_ws, axis=0)
      all_ids = np.concatenate(all_ids, axis=0)

      total_size = sum(shard_sizes)

      assert dataset.X.shape[0] == total_size

      # deterministic
      test_Xs, test_ys, test_ws, test_ids = [], [], [], []
      for bidx, (a, b, c, d) in enumerate(
          dataset.iterbatches(
              batch_size=batch_size, pad_batches=False, deterministic=True)):

        test_Xs.append(a)
        test_ys.append(b)
        test_ws.append(c)
        test_ids.append(d)

      if batch_size is None:
        for idx, (tx, ty, tw, tids) in enumerate(
            zip(test_Xs, test_ys, test_ws, test_ids)):
          assert len(tx) == shard_sizes[idx]
          assert len(ty) == shard_sizes[idx]
          assert len(tw) == shard_sizes[idx]
          assert len(tids) == shard_sizes[idx]

      test_Xs = np.concatenate(test_Xs, axis=0)
      test_ys = np.concatenate(test_ys, axis=0)
      test_ws = np.concatenate(test_ws, axis=0)
      test_ids = np.concatenate(test_ids, axis=0)

      if batch_size is None:
        assert bidx == len(shard_sizes) - 1
      else:
        assert bidx == math.ceil(total_size / batch_size) - 1

      np.testing.assert_array_equal(all_Xs, test_Xs)
      np.testing.assert_array_equal(all_ys, test_ys)
      np.testing.assert_array_equal(all_ws, test_ws)
      np.testing.assert_array_equal(all_ids, test_ids)

      # non-deterministic
      test_Xs, test_ys, test_ws, test_ids = [], [], [], []

      for bidx, (a, b, c, d) in enumerate(
          dataset.iterbatches(
              batch_size=batch_size, pad_batches=False, deterministic=False)):

        test_Xs.append(a)
        test_ys.append(b)
        test_ws.append(c)
        test_ids.append(d)

      # we don't know the order in which the shards are iterated in.
      test_Xs = np.concatenate(test_Xs, axis=0)
      test_ys = np.concatenate(test_ys, axis=0)
      test_ws = np.concatenate(test_ws, axis=0)
      test_ids = np.concatenate(test_ids, axis=0)

      if batch_size is None:
        assert bidx == len(shard_sizes) - 1
      else:
        assert bidx == math.ceil(total_size / batch_size) - 1

      np.testing.assert_array_equal(
          np.sort(all_Xs, axis=0), np.sort(test_Xs, axis=0))
      np.testing.assert_array_equal(
          np.sort(all_ys, axis=0), np.sort(test_ys, axis=0))
      np.testing.assert_array_equal(
          np.sort(all_ws, axis=0), np.sort(test_ws, axis=0))
      np.testing.assert_array_equal(
          np.sort(all_ids, axis=0), np.sort(test_ids, axis=0))

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

  def test_merge(self):
    """Test that dataset merge works."""
    num_datapoints = 10
    num_features = 10
    num_tasks = 1
    num_datasets = 4
    datasets = []
    for i in range(num_datasets):
      Xi = np.random.rand(num_datapoints, num_features)
      yi = np.random.randint(2, size=(num_datapoints, num_tasks))
      wi = np.ones((num_datapoints, num_tasks))
      idsi = np.array(["id"] * num_datapoints)
      dataseti = dc.data.DiskDataset.from_numpy(Xi, yi, wi, idsi)
      datasets.append(dataseti)

    new_data = dc.data.datasets.DiskDataset.merge(datasets)

    # Check that we have all the data in
    assert new_data.X.shape == (num_datapoints * num_datasets, num_features)
    assert new_data.y.shape == (num_datapoints * num_datasets, num_tasks)
    assert len(new_data.tasks) == len(datasets[0].tasks)

  def test_make_iterator(self):
    """Test creating a Tensorflow Iterator from a Dataset."""
    X = np.random.random((100, 5))
    y = np.random.random((100, 1))
    dataset = dc.data.NumpyDataset(X, y)
    iterator = dataset.make_iterator(
        batch_size=10, epochs=2, deterministic=True)
    next_element = iterator.get_next()
    with self.session() as sess:
      for i in range(20):
        batch_X, batch_y, batch_w = sess.run(next_element)
        offset = (i % 10) * 10
        np.testing.assert_array_equal(X[offset:offset + 10, :], batch_X)
        np.testing.assert_array_equal(y[offset:offset + 10, :], batch_y)
        np.testing.assert_array_equal(np.ones((10, 1)), batch_w)
      finished = False
      try:
        sess.run(next_element)
      except tf.errors.OutOfRangeError:
        finished = True
    assert finished


if __name__ == "__main__":
  unittest.main()
