"""
Tests for transformer objects. 
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "GPL"

import unittest
import numpy as np
from deepchem.transformers import LogTransformer
from deepchem.transformers import NormalizationTransformer
from deepchem.transformers import BalancingTransformer
from deepchem.datasets.tests import TestDatasetAPI

class TestTransformerAPI(TestDatasetAPI):
  """
  Test top-level API for transformer objects.
  """

  def test_y_log_transformer(self):
    """Tests logarithmic data transformer."""
    solubility_dataset = self.load_solubility_data()
    log_transformer = LogTransformer(
        transform_y=True, dataset=solubility_dataset)
    X, y, w, ids = solubility_dataset.to_numpy()
    log_transformer.transform(solubility_dataset)
    X_t, y_t, w_t, ids_t = solubility_dataset.to_numpy()
    
    # Check ids are unchanged.
    for id_elt, id_t_elt in zip(ids, ids_t):
      assert id_elt == id_t_elt
    # Check X is unchanged since this is a y transformer
    np.testing.assert_allclose(X, X_t)
    # Check w is unchanged since this is a y transformer
    np.testing.assert_allclose(w, w_t)
    # Check y is now a logarithmic version of itself
    np.testing.assert_allclose(y_t, np.log(y))

    # Check that untransform does the right thing.
    np.testing.assert_allclose(log_transformer.untransform(y_t), y)

  def test_X_log_transformer(self):
    """Tests logarithmic data transformer."""
    solubility_dataset = self.load_solubility_data()
    log_transformer = LogTransformer(
        transform_X=True, dataset=solubility_dataset)
    X, y, w, ids = solubility_dataset.to_numpy()
    log_transformer.transform(solubility_dataset)
    X_t, y_t, w_t, ids_t = solubility_dataset.to_numpy()
    
    # Check ids are unchanged.
    for id_elt, id_t_elt in zip(ids, ids_t):
      assert id_elt == id_t_elt
    # Check y is unchanged since this is a X transformer
    np.testing.assert_allclose(y, y_t)
    # Check w is unchanged since this is a y transformer
    np.testing.assert_allclose(w, w_t)
    # Check y is now a logarithmic version of itself
    np.testing.assert_allclose(X_t, np.log(X))

    # Check that untransform does the right thing.
    np.testing.assert_allclose(log_transformer.untransform(X_t), X)

  def test_y_normalization_transformer(self):
    """Tests normalization transformer."""
    solubility_dataset = self.load_solubility_data()
    normalization_transformer = NormalizationTransformer(
        transform_y=True, dataset=solubility_dataset)
    X, y, w, ids = solubility_dataset.to_numpy()
    normalization_transformer.transform(solubility_dataset)
    X_t, y_t, w_t, ids_t = solubility_dataset.to_numpy()
    # Check ids are unchanged.
    for id_elt, id_t_elt in zip(ids, ids_t):
      assert id_elt == id_t_elt
    # Check X is unchanged since this is a y transformer
    np.testing.assert_allclose(X, X_t)
    # Check w is unchanged since this is a y transformer
    np.testing.assert_allclose(w, w_t)
    # Check that y_t has zero mean, unit std.
    assert np.isclose(y_t.mean(), 0.)
    assert np.isclose(y_t.std(), 1.)

    # Check that untransform does the right thing.
    np.testing.assert_allclose(normalization_transformer.untransform(y_t), y)

  def test_X_normalization_transformer(self):
    """Tests normalization transformer."""
    solubility_dataset = self.load_solubility_data()
    normalization_transformer = NormalizationTransformer(
        transform_X=True, dataset=solubility_dataset)
    X, y, w, ids = solubility_dataset.to_numpy()
    normalization_transformer.transform(solubility_dataset)
    X_t, y_t, w_t, ids_t = solubility_dataset.to_numpy()
    # Check ids are unchanged.
    for id_elt, id_t_elt in zip(ids, ids_t):
      assert id_elt == id_t_elt
    # Check y is unchanged since this is a X transformer
    np.testing.assert_allclose(y, y_t)
    # Check w is unchanged since this is a y transformer
    np.testing.assert_allclose(w, w_t)
    # Check that X_t has zero mean, unit std.
    #np.set_printoptions(threshold='nan')
    mean = X_t.mean(axis=0)
    assert np.amax(np.abs(mean-np.zeros_like(mean))) < 1e-7
    orig_std_array = X.std(axis=0)
    std_array = X_t.std(axis=0)
    # Entries with zero std are not normalized
    for orig_std, std in zip(orig_std_array, std_array):
      if not np.isclose(orig_std, 0):
        assert np.isclose(std, 1)

    # TODO(rbharath): Untransform doesn't work properly for binary feature
    # vectors. Need to figure out what's wrong here. (low priority)
    ## Check that untransform does the right thing.
    #np.testing.assert_allclose(normalization_transformer.untransform(X_t), X)

  def test_singletask_balancing_transformer(self):
    """Test balancing transformer on single-task dataset."""

    classification_dataset = self.load_classification_data()
    balancing_transformer = BalancingTransformer(
      transform_w=True, dataset=classification_dataset)
    X, y, w, ids = classification_dataset.to_numpy()
    balancing_transformer.transform(classification_dataset)
    X_t, y_t, w_t, ids_t = classification_dataset.to_numpy()
    # Check ids are unchanged.
    for id_elt, id_t_elt in zip(ids, ids_t):
      assert id_elt == id_t_elt
    # Check X is unchanged since this is a w transformer
    np.testing.assert_allclose(X, X_t)
    # Check y is unchanged since this is a w transformer
    np.testing.assert_allclose(y, y_t)
    for ind, task in enumerate(classification_dataset.get_task_names()):
      y_task = y_t[:, ind]
      w_task = w_t[:, ind]
      w_orig_task = w[:, ind]
      # Assert that entries with zero weight retain zero weight
      np.testing.assert_allclose(
          w_task[w_orig_task == 0], np.zeros_like(w_task[w_orig_task == 0]))
      # Check that sum of 0s equals sum of 1s in transformed for each task
      assert np.isclose(np.sum(w_task[y_task == 0]),
                        np.sum(w_task[y_task == 1]))

  def test_multitask_balancing_transformer(self):
    """Test balancing transformer on multitask dataset."""
    multitask_dataset = self.load_multitask_data()
    balancing_transformer = BalancingTransformer(
      transform_w=True, dataset=multitask_dataset)
    X, y, w, ids = multitask_dataset.to_numpy()
    balancing_transformer.transform(multitask_dataset)
    X_t, y_t, w_t, ids_t = multitask_dataset.to_numpy()
    # Check ids are unchanged.
    for id_elt, id_t_elt in zip(ids, ids_t):
      assert id_elt == id_t_elt
    # Check X is unchanged since this is a w transformer
    np.testing.assert_allclose(X, X_t)
    # Check y is unchanged since this is a w transformer
    np.testing.assert_allclose(y, y_t)
    for ind, task in enumerate(multitask_dataset.get_task_names()):
      y_task = y_t[:, ind]
      w_task = w_t[:, ind]
      w_orig_task = w[:, ind]
      # Assert that entries with zero weight retain zero weight
      np.testing.assert_allclose(
          w_task[w_orig_task == 0], np.zeros_like(w_task[w_orig_task == 0]))
      # Check that sum of 0s equals sum of 1s in transformed for each task
      assert np.isclose(np.sum(w_task[y_task == 0]),
                        np.sum(w_task[y_task == 1]))
