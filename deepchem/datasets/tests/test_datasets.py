"""
Tests for dataset creation
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "GPL"

import unittest
import tempfile
import os
import shutil
import numpy as np
from deepchem.datasets import sparsify_features
from deepchem.datasets import densify_features
from deepchem.datasets import pad_batch
from deepchem.datasets import pad_features
from deepchem.datasets import Dataset
from deepchem.featurizers.featurize import DataLoader
from deepchem.featurizers.fingerprints import CircularFingerprint
from deepchem.transformers import NormalizationTransformer
from deepchem.datasets.tests import TestDatasetAPI

class TestBasicDatasetAPI(TestDatasetAPI):
  """
  Test basic top-level API for dataset objects.
  """

  def test_sparsify_and_densify(self):
    """Test that sparsify and densify work as inverses."""
    # Test on identity matrix
    num_samples = 10
    num_features = num_samples
    X = np.eye(num_samples)
    X_sparse = sparsify_features(X)
    X_reconstructed = densify_features(X_sparse, num_features)
    np.testing.assert_array_equal(X, X_reconstructed)

    # Generate random sparse features dataset
    np.random.seed(123)
    p = .05
    X = np.random.binomial(1, p, size=(num_samples, num_features))
    X_sparse = sparsify_features(X)
    X_reconstructed = densify_features(X_sparse, num_features)
    np.testing.assert_array_equal(X, X_reconstructed)

    # Test edge case with array of all zeros
    X = np.zeros((num_samples, num_features))
    X_sparse = sparsify_features(X)
    X_reconstructed = densify_features(X_sparse, num_features)
    np.testing.assert_array_equal(X, X_reconstructed)

  def test_pad_features(self):
    """Test that pad_features pads features correctly."""
    batch_size = 100
    num_features = 10
    num_tasks = 5
  
    # Test cases where n_samples < 2*n_samples < batch_size
    n_samples = 29
    X_b = np.zeros((n_samples, num_features))
  
    X_out = pad_features(batch_size, X_b)
    assert len(X_out) == batch_size

    # Test cases where n_samples < batch_size
    n_samples = 79
    X_b = np.zeros((n_samples, num_features))
    X_out = pad_features(batch_size, X_b)
    assert len(X_out) == batch_size

    # Test case where n_samples == batch_size
    n_samples = 100 
    X_b = np.zeros((n_samples, num_features))
    X_out = pad_features(batch_size, X_b)
    assert len(X_out) == batch_size

    # Test case for object featurization.
    n_samples = 2
    X_b = np.array([{"a": 1}, {"b": 2}])
    X_out = pad_features(batch_size, X_b)
    assert len(X_out) == batch_size

    # Test case for more complicated object featurization
    n_samples = 2
    X_b = np.array([(1, {"a": 1}), (2, {"b": 2})])
    X_out = pad_features(batch_size, X_b)
    assert len(X_out) == batch_size

    # Test case with multidimensional data
    n_samples = 50
    num_atoms = 15
    d = 3
    X_b = np.zeros((n_samples, num_atoms, d))
    X_out = pad_features(batch_size, X_b)
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
  
    X_out, y_out, w_out, ids_out = pad_batch(
        batch_size, X_b, y_b, w_b, ids_b)
    assert len(X_out) == len(y_out) == len(w_out) == len(ids_out) == batch_size

    # Test cases where n_samples < batch_size
    n_samples = 79
    X_b = np.zeros((n_samples, num_features))
    y_b = np.zeros((n_samples, num_tasks))
    w_b = np.zeros((n_samples, num_tasks))
    ids_b = np.zeros((n_samples,))
  
    X_out, y_out, w_out, ids_out = pad_batch(
        batch_size, X_b, y_b, w_b, ids_b)
    assert len(X_out) == len(y_out) == len(w_out) == len(ids_out) == batch_size

    # Test case where n_samples == batch_size
    n_samples = 100 
    X_b = np.zeros((n_samples, num_features))
    y_b = np.zeros((n_samples, num_tasks))
    w_b = np.zeros((n_samples, num_tasks))
    ids_b = np.zeros((n_samples,))
  
    X_out, y_out, w_out, ids_out = pad_batch(
        batch_size, X_b, y_b, w_b, ids_b)
    assert len(X_out) == len(y_out) == len(w_out) == len(ids_out) == batch_size

    # Test case for object featurization.
    n_samples = 2
    X_b = np.array([{"a": 1}, {"b": 2}])
    y_b = np.zeros((n_samples, num_tasks))
    w_b = np.zeros((n_samples, num_tasks))
    ids_b = np.zeros((n_samples,))
    X_out, y_out, w_out, ids_out = pad_batch(
        batch_size, X_b, y_b, w_b, ids_b)
    assert len(X_out) == len(y_out) == len(w_out) == len(ids_out) == batch_size

    # Test case for more complicated object featurization
    n_samples = 2
    X_b = np.array([(1, {"a": 1}), (2, {"b": 2})])
    y_b = np.zeros((n_samples, num_tasks))
    w_b = np.zeros((n_samples, num_tasks))
    ids_b = np.zeros((n_samples,))
    X_out, y_out, w_out, ids_out = pad_batch(
        batch_size, X_b, y_b, w_b, ids_b)
    assert len(X_out) == len(y_out) == len(w_out) == len(ids_out) == batch_size

    # Test case with multidimensional data
    n_samples = 50
    num_atoms = 15
    d = 3
    X_b = np.zeros((n_samples, num_atoms, d))
    y_b = np.zeros((n_samples, num_tasks))
    w_b = np.zeros((n_samples, num_tasks))
    ids_b = np.zeros((n_samples,))
  
    X_out, y_out, w_out, ids_out = pad_batch(
        batch_size, X_b, y_b, w_b, ids_b)
    assert len(X_out) == len(y_out) == len(w_out) == len(ids_out) == batch_size
    
  def test_get_task_names(self):
    """Test that get_task_names returns correct task_names"""
    solubility_dataset = self.load_solubility_data()
    assert solubility_dataset.get_task_names() == ["log-solubility"]

    multitask_dataset = self.load_multitask_data()
    assert sorted(multitask_dataset.get_task_names()) == sorted(["task0",
        "task1", "task2", "task3", "task4", "task5", "task6", "task7", "task8",
        "task9", "task10", "task11", "task12", "task13", "task14", "task15",
        "task16"])

  def test_get_data_shape(self):
    """Test that get_data_shape returns currect data shape"""
    solubility_dataset = self.load_solubility_data()
    assert solubility_dataset.get_data_shape() == (1024,) 
    
    multitask_dataset = self.load_multitask_data()
    assert multitask_dataset.get_data_shape() == (1024,)

  def test_len(self):
    """Test that len(dataset) works."""
    solubility_dataset = self.load_solubility_data()
    assert len(solubility_dataset) == 10

  def test_reshard(self):
    """Test that resharding the dataset works."""
    solubility_dataset = self.load_solubility_data()
    X, y, w, ids = solubility_dataset.to_numpy()
    assert solubility_dataset.get_number_shards() == 1
    solubility_dataset.reshard(shard_size=1)
    assert solubility_dataset.get_shard_size() == 1
    X_r, y_r, w_r, ids_r = solubility_dataset.to_numpy()
    assert solubility_dataset.get_number_shards() == 10
    solubility_dataset.reshard(shard_size=10)
    assert solubility_dataset.get_shard_size() == 10
    X_rr, y_rr, w_rr, ids_rr = solubility_dataset.to_numpy()

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
    dataset = Dataset.from_numpy(self.data_dir, X, y, w, ids)

    select_dir = tempfile.mkdtemp()
    indices = [0, 4, 5, 8]
    select_dataset = dataset.select(select_dir, indices)
    X_sel, y_sel, w_sel, ids_sel = select_dataset.to_numpy()
    np.testing.assert_array_equal(X[indices], X_sel)
    np.testing.assert_array_equal(y[indices], y_sel)
    np.testing.assert_array_equal(w[indices], w_sel)
    np.testing.assert_array_equal(ids[indices], ids_sel)
    shutil.rmtree(select_dir)

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
    
    dataset = Dataset.from_numpy(self.data_dir, X, y, w, ids, verbosity="high")

    X_shape, y_shape, w_shape, ids_shape = dataset.get_shape()
    assert X_shape == X.shape
    assert y_shape == y.shape
    assert w_shape == w.shape
    assert ids_shape == ids.shape


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
    
    dataset = Dataset.from_numpy(self.data_dir, X, y, w, ids, verbosity="high")

    task_dirs = []
    try:
      for task in range(num_tasks):
        task_dirs.append(tempfile.mkdtemp())
      singletask_datasets = dataset.to_singletask(task_dirs)
      for task in range(num_tasks):
        singletask_dataset = singletask_datasets[task]
        X_task, y_task, w_task, ids_task = singletask_dataset.to_numpy()
        w_nonzero = w[:, task] != 0
        np.testing.assert_array_equal(X_task, X[w_nonzero != 0])
        np.testing.assert_array_equal(y_task.flatten(), y[:, task][w_nonzero != 0])
        np.testing.assert_array_equal(w_task.flatten(), w[:, task][w_nonzero != 0])
        np.testing.assert_array_equal(ids_task, ids[w_nonzero != 0])
    finally:
      # Cleanup
      for task_dir in task_dirs:
        shutil.rmtree(task_dir)
  
  def test_iterbatches(self):
    """Test that iterating over batches of data works."""
    solubility_dataset = self.load_solubility_data()
    batch_size = 2
    data_shape = solubility_dataset.get_data_shape()
    tasks = solubility_dataset.get_task_names()
    for (X_b, y_b, w_b, ids_b)  in solubility_dataset.iterbatches(batch_size):
      assert X_b.shape == (batch_size,) + data_shape
      assert y_b.shape == (batch_size,) + (len(tasks),)
      assert w_b.shape == (batch_size,) + (len(tasks),)
      assert ids_b.shape == (batch_size,)

  def test_to_numpy(self):
    """Test that transformation to numpy arrays is sensible."""
    solubility_dataset = self.load_solubility_data()
    data_shape = solubility_dataset.get_data_shape()
    tasks = solubility_dataset.get_task_names()
    X, y, w, ids = solubility_dataset.to_numpy()
    N_samples = len(solubility_dataset)
    N_tasks = len(tasks)
    
    assert X.shape == (N_samples,) + data_shape
    assert y.shape == (N_samples, N_tasks)
    assert w.shape == (N_samples, N_tasks)
    assert ids.shape == (N_samples,)

  def test_consistent_ordering(self):
    """Test that ordering of labels is consistent over time."""
    solubility_dataset = self.load_solubility_data()

    ids1 = solubility_dataset.ids
    ids2 = solubility_dataset.ids

    assert np.array_equal(ids1, ids2)

  def test_get_statistics(self):
    """Test statistics computation of this dataset."""
    solubility_dataset = self.load_solubility_data()
    X, y, _, _ = solubility_dataset.to_numpy()
    X_means, y_means = np.mean(X, axis=0), np.mean(y, axis=0)
    X_stds, y_stds = np.std(X, axis=0), np.std(y, axis=0)
    comp_X_means, comp_X_stds, comp_y_means, comp_y_stds = \
        solubility_dataset.get_statistics()
    np.testing.assert_allclose(comp_X_means, X_means)
    np.testing.assert_allclose(comp_y_means, y_means)
    np.testing.assert_allclose(comp_X_stds, X_stds)
    np.testing.assert_allclose(comp_y_stds, y_stds)
