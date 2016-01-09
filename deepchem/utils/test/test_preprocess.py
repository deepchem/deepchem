"""
Tests for preprocessing code.
"""
__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2015, Stanford University"
__license__ = "LGPL"

import numpy as np
import unittest

# TODO(rbharath): Can this just be removed?
def balance_positives(y, W):
  """Ensure that positive and negative examples have equal weight."""
  n_samples, n_targets = np.shape(y)
  for target_ind in range(n_targets):
    positive_inds, negative_inds = [], []
    to_next_target = False
    for sample_ind in range(n_samples):
      label = y[sample_ind, target_ind]
      if label == 1:
        positive_inds.append(sample_ind)
      elif label == 0:
        negative_inds.append(sample_ind)
      elif label == -1:  # Case of missing label
        continue
      else:
        warnings.warn("Labels must be 0/1 or -1 " +
                      "(missing data) for balance_positives target %d. " % target_ind +
                      "Continuing without balancing.")
        to_next_target = True
        break
    if to_next_target:
      continue
    n_positives, n_negatives = len(positive_inds), len(negative_inds)
    # TODO(rbharath): This results since the coarse train/test split doesn't
    # guarantee that the test set actually has any positives for targets. FIX
    # THIS BEFORE RELEASE!
    if n_positives == 0:
      pos_weight = 0
    else:
      pos_weight = float(n_negatives)/float(n_positives)
    W[positive_inds, target_ind] = pos_weight
    W[negative_inds, target_ind] = 1
  return W

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
