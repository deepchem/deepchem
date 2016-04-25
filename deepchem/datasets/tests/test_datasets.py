"""
Tests for dataset creation
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "LGPL"

import unittest
import tempfile
import os
import shutil
import numpy as np
from deepchem.datasets import Dataset
from deepchem.featurizers.featurize import DataFeaturizer
from deepchem.featurizers.fingerprints import CircularFingerprint
from deepchem.transformers import NormalizationTransformer
from deepchem.datasets.tests import TestDatasetAPI

class TestBasicDatasetAPI(TestDatasetAPI):
  """
  Test basic top-level API for dataset objects.
  """
  def test_get_task_names(self):
    """Test that get_task_names returns correct task_names"""
    solubility_dataset = self._load_solubility_data()
    assert solubility_dataset.get_task_names() == ["log-solubility"]

    multitask_dataset = self._load_multitask_data()
    assert sorted(multitask_dataset.get_task_names()) == sorted(["task0",
        "task1", "task2", "task3", "task4", "task5", "task6", "task7", "task8",
        "task9", "task10", "task11", "task12", "task13", "task14", "task15",
        "task16"])
  

  def test_get_data_shape(self):
    """Test that get_data_shape returns currect data shape"""
    solubility_dataset = self._load_solubility_data()
    assert solubility_dataset.get_data_shape() == (1024,) 
    
    multitask_dataset = self._load_multitask_data()
    assert multitask_dataset.get_data_shape() == (1024,)

  def test_len(self):
    """Test that len(dataset) works."""
    solubility_dataset = self._load_solubility_data()
    assert len(solubility_dataset) == 10
  
  def test_iterbatches(self):
    """Test that iterating over batches of data works."""
    solubility_dataset = self._load_solubility_data()
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
    solubility_dataset = self._load_solubility_data()
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
    solubility_dataset = self._load_solubility_data()

    ids1 = solubility_dataset.get_ids()
    ids2 = solubility_dataset.get_ids()

    assert np.array_equal(ids1, ids2)

  def test_get_statistics(self):
    """Test statistics computation of this dataset."""
    solubility_dataset = self._load_solubility_data()
    X, y, _, _ = solubility_dataset.to_numpy()
    X_means, y_means = np.mean(X, axis=0), np.mean(y, axis=0)
    X_stds, y_stds = np.std(X, axis=0), np.std(y, axis=0)
    comp_X_means, comp_X_stds, comp_y_means, comp_y_stds = \
        solubility_dataset.get_statistics()
    np.testing.assert_allclose(comp_X_means, X_means)
    np.testing.assert_allclose(comp_y_means, y_means)
    np.testing.assert_allclose(comp_X_stds, X_stds)
    np.testing.assert_allclose(comp_y_stds, y_stds)
