"""
Tests for preprocessing code.
"""
__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2015, Stanford University"
__license__ = "LGPL"

import numpy as np
import unittest
from deep_chem.utils.preprocess import balance_positives
from deep_chem.utils.preprocess import undo_transform_outputs

def ensure_balanced(y, W):
  """Helper function that ensures postives and negatives are balanced."""
  n_samples, n_targets = np.shape(y)
  for target_ind in range(n_targets):
    pos_weight, neg_weight = 0, 0
    for sample_ind in range(n_samples):
      if y[sample_ind, target_ind] == 0:
        neg_weight += Wbal[sample_ind, target_ind]
      elif y[sample_ind, target_ind] == 1:
        pos_weight += Wbal[sample_ind, target_ind]
    assert np.isclose(pos_weight, neg_weight)

class TestPreprocess(unittest.TestCase):
  """
  Test Preprocessing code.
  """
  def test_balance_positives(self):
    n_samples, n_features, n_targets = 100, 10, 10
    y = np.random.randint(2, size=(n_samples, n_targets))
    W = np.ones((n_samples, n_targets))
    Wbal = balance_positives(y, W)
    for target_ind in range(n_targets):
      pos_weight, neg_weight = 0, 0
      for sample_ind in range(n_samples):
        if y[sample_ind, target_ind] == 0:
          neg_weight += Wbal[sample_ind, target_ind]
        elif y[sample_ind, target_ind] == 1:
          pos_weight += Wbal[sample_ind, target_ind]
      assert np.isclose(pos_weight, neg_weight)

  def test_undo_transform_outputs(self):
    # Test undo-log
    y_raw = np.ones(10)
    y_pred = np.log(y_raw)
    output_transforms = ["log"]
    assert np.array_equal(y_raw, undo_transform_outputs(y_raw, y_pred, output_transforms))

    # Test undo-normalization
    y_raw = np.random.randint(0, 10, size=(10,))
    mean = np.mean(y_raw)
    std = np.std(y_raw)
    y_pred = (y_raw-mean)/std
    output_transforms = ["normalize"]
    y_ret = undo_transform_outputs(y_raw, y_pred, output_transforms)
    assert np.allclose(y_raw, y_ret)
    
    # Test undo log-normalization
    y_raw = np.random.randint(1, 10, size=(10,))
    y_pred = np.log(y_raw)
    mean = np.mean(y_pred)
    std = np.std(y_pred)
    y_pred = (y_pred - mean)/std
    output_transforms = ["log", "normalize"]
    y_ret = undo_transform_outputs(y_raw, y_pred, output_transforms)
    assert np.allclose(y_raw, y_ret)
