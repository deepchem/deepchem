"""
Testing singletask/multitask dataset shuffling 
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "MIT"

import os
import shutil
import tempfile
import unittest
import deepchem as dc
import numpy as np


class TestShuffle(unittest.TestCase):
  """
  Test singletask/multitask dataset shuffling.
  """

  #def test_shuffle(self):
  #  """Test that datasets can be merged."""
  #  current_dir = os.path.dirname(os.path.realpath(__file__))

  #  dataset_file = os.path.join(
  #      current_dir, "../../models/tests/example.csv")

  #  featurizer = dc.feat.CircularFingerprint(size=1024)
  #  tasks = ["log-solubility"]
  #  loader = dc.data.CSVLoader(
  #      tasks=tasks, smiles_field="smiles", featurizer=featurizer)
  #  dataset = loader.featurize(dataset_file, shard_size=2)

  #  X_orig, y_orig, w_orig, orig_ids = (dataset.X, dataset.y, dataset.w,
  #                                      dataset.ids)
  #  orig_len = len(dataset)

  #  dataset.shuffle(iterations=5)
  #  X_new, y_new, w_new, new_ids = (dataset.X, dataset.y, dataset.w,
  #                                  dataset.ids)
  #
  #  assert len(dataset) == orig_len
  #  # The shuffling should have switched up the ordering
  #  assert not np.array_equal(orig_ids, new_ids)
  #  # But all the same entries should still be present
  #  assert sorted(orig_ids) == sorted(new_ids)
  #  # All the data should have same shape
  #  assert X_orig.shape == X_new.shape
  #  assert y_orig.shape == y_new.shape
  #  assert w_orig.shape == w_new.shape

  def test_sparse_shuffle(self):
    """Test that sparse datasets can be shuffled quickly."""
    current_dir = os.path.dirname(os.path.realpath(__file__))

    dataset_file = os.path.join(current_dir, "../../models/tests/example.csv")

    featurizer = dc.feat.CircularFingerprint(size=1024)
    tasks = ["log-solubility"]
    loader = dc.data.CSVLoader(
        tasks=tasks, smiles_field="smiles", featurizer=featurizer)
    dataset = loader.featurize(dataset_file, shard_size=2)

    X_orig, y_orig, w_orig, orig_ids = (dataset.X, dataset.y, dataset.w,
                                        dataset.ids)
    orig_len = len(dataset)

    dataset.sparse_shuffle()
    X_new, y_new, w_new, new_ids = (dataset.X, dataset.y, dataset.w,
                                    dataset.ids)

    assert len(dataset) == orig_len
    # The shuffling should have switched up the ordering
    assert not np.array_equal(orig_ids, new_ids)
    # But all the same entries should still be present
    assert sorted(orig_ids) == sorted(new_ids)
    # All the data should have same shape
    assert X_orig.shape == X_new.shape
    assert y_orig.shape == y_new.shape
    assert w_orig.shape == w_new.shape

  def test_shuffle_each_shard(self):
    """Test that shuffle_each_shard works."""
    n_samples = 100
    n_tasks = 10
    n_features = 10

    X = np.random.rand(n_samples, n_features)
    y = np.random.randint(2, size=(n_samples, n_tasks))
    w = np.random.randint(2, size=(n_samples, n_tasks))
    ids = np.arange(n_samples)
    dataset = dc.data.DiskDataset.from_numpy(X, y, w, ids)
    dataset.reshard(shard_size=10)

    dataset.shuffle_each_shard()
    X_s, y_s, w_s, ids_s = (dataset.X, dataset.y, dataset.w, dataset.ids)
    assert X_s.shape == X.shape
    assert y_s.shape == y.shape
    assert ids_s.shape == ids.shape
    assert w_s.shape == w.shape

    # The ids should now store the performed permutation. Check that the
    # original dataset is recoverable.
    for i in range(n_samples):
      np.testing.assert_array_equal(X_s[i], X[ids_s[i]])
      np.testing.assert_array_equal(y_s[i], y[ids_s[i]])
      np.testing.assert_array_equal(w_s[i], w[ids_s[i]])
      np.testing.assert_array_equal(ids_s[i], ids[ids_s[i]])

  def test_shuffle_shards(self):
    """Test that shuffle_shards works."""
    n_samples = 100
    n_tasks = 10
    n_features = 10

    X = np.random.rand(n_samples, n_features)
    y = np.random.randint(2, size=(n_samples, n_tasks))
    w = np.random.randint(2, size=(n_samples, n_tasks))
    ids = np.arange(n_samples)
    dataset = dc.data.DiskDataset.from_numpy(X, y, w, ids)
    dataset.reshard(shard_size=10)
    dataset.shuffle_shards()

    X_s, y_s, w_s, ids_s = (dataset.X, dataset.y, dataset.w, dataset.ids)

    assert X_s.shape == X.shape
    assert y_s.shape == y.shape
    assert ids_s.shape == ids.shape
    assert w_s.shape == w.shape

    # The ids should now store the performed permutation. Check that the
    # original dataset is recoverable.
    for i in range(n_samples):
      np.testing.assert_array_equal(X_s[i], X[ids_s[i]])
      np.testing.assert_array_equal(y_s[i], y[ids_s[i]])
      np.testing.assert_array_equal(w_s[i], w[ids_s[i]])
      np.testing.assert_array_equal(ids_s[i], ids[ids_s[i]])
