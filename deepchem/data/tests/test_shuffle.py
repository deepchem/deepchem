"""
Testing singletask/multitask dataset shuffling
"""
import os
import deepchem as dc
import numpy as np


def test_complete_shuffle_one_shard():
  """Test that complete shuffle works with only one shard."""
  X = np.random.rand(10, 10)
  dataset = dc.data.DiskDataset.from_numpy(X)
  shuffled = dataset.complete_shuffle()
  assert len(shuffled) == len(dataset)
  assert not np.array_equal(shuffled.ids, dataset.ids)
  assert sorted(shuffled.ids) == sorted(dataset.ids)
  assert shuffled.X.shape == dataset.X.shape
  assert shuffled.y.shape == dataset.y.shape
  assert shuffled.w.shape == dataset.w.shape
  original_indices = dict((id, i) for i, id in enumerate(dataset.ids))
  shuffled_indices = dict((id, i) for i, id in enumerate(shuffled.ids))
  for id in dataset.ids:
    i = original_indices[id]
    j = shuffled_indices[id]
    assert np.array_equal(dataset.X[i], shuffled.X[j])
    assert np.array_equal(dataset.y[i], shuffled.y[j])
    assert np.array_equal(dataset.w[i], shuffled.w[j])


def test_complete_shuffle_multiple_shard():
  """Test that complete shuffle works with multiple shards."""
  X = np.random.rand(100, 10)
  dataset = dc.data.DiskDataset.from_numpy(X)
  dataset.reshard(shard_size=10)
  shuffled = dataset.complete_shuffle()
  assert len(shuffled) == len(dataset)
  assert not np.array_equal(shuffled.ids, dataset.ids)
  assert sorted(shuffled.ids) == sorted(dataset.ids)
  assert shuffled.X.shape == dataset.X.shape
  assert shuffled.y.shape == dataset.y.shape
  assert shuffled.w.shape == dataset.w.shape
  original_indices = dict((id, i) for i, id in enumerate(dataset.ids))
  shuffled_indices = dict((id, i) for i, id in enumerate(shuffled.ids))
  for id in dataset.ids:
    i = original_indices[id]
    j = shuffled_indices[id]
    assert np.array_equal(dataset.X[i], shuffled.X[j])
    assert np.array_equal(dataset.y[i], shuffled.y[j])
    assert np.array_equal(dataset.w[i], shuffled.w[j])


def test_complete_shuffle_multiple_shard_uneven():
  """Test that complete shuffle works with multiple shards and some shards not full size."""
  X = np.random.rand(57, 10)
  dataset = dc.data.DiskDataset.from_numpy(X)
  dataset.reshard(shard_size=10)
  shuffled = dataset.complete_shuffle()
  assert len(shuffled) == len(dataset)
  assert not np.array_equal(shuffled.ids, dataset.ids)
  assert sorted(shuffled.ids) == sorted(dataset.ids)
  assert shuffled.X.shape == dataset.X.shape
  assert shuffled.y.shape == dataset.y.shape
  assert shuffled.w.shape == dataset.w.shape
  original_indices = dict((id, i) for i, id in enumerate(dataset.ids))
  shuffled_indices = dict((id, i) for i, id in enumerate(shuffled.ids))
  for id in dataset.ids:
    i = original_indices[id]
    j = shuffled_indices[id]
    assert np.array_equal(dataset.X[i], shuffled.X[j])
    assert np.array_equal(dataset.y[i], shuffled.y[j])
    assert np.array_equal(dataset.w[i], shuffled.w[j])


def test_complete_shuffle():
  """Test that complete shuffle."""
  current_dir = os.path.dirname(os.path.realpath(__file__))

  dataset_file = os.path.join(current_dir, "../../models/tests/example.csv")

  featurizer = dc.feat.CircularFingerprint(size=1024)
  tasks = ["log-solubility"]
  loader = dc.data.CSVLoader(
      tasks=tasks, feature_field="smiles", featurizer=featurizer)
  dataset = loader.create_dataset(dataset_file, shard_size=2)

  X_orig, y_orig, w_orig, orig_ids = (dataset.X, dataset.y, dataset.w,
                                      dataset.ids)
  orig_len = len(dataset)

  shuffled = dataset.complete_shuffle()
  X_new, y_new, w_new, new_ids = (shuffled.X, shuffled.y, shuffled.w,
                                  shuffled.ids)

  assert len(shuffled) == orig_len
  # The shuffling should have switched up the ordering
  assert not np.array_equal(orig_ids, new_ids)
  # But all the same entries should still be present
  assert sorted(orig_ids) == sorted(new_ids)
  # All the data should have same shape
  assert X_orig.shape == X_new.shape
  assert y_orig.shape == y_new.shape
  assert w_orig.shape == w_new.shape
  original_indices = dict((id, i) for i, id in enumerate(dataset.ids))
  shuffled_indices = dict((id, i) for i, id in enumerate(shuffled.ids))
  for id in dataset.ids:
    i = original_indices[id]
    j = shuffled_indices[id]
    assert np.array_equal(dataset.X[i], shuffled.X[j])
    assert np.array_equal(dataset.y[i], shuffled.y[j])
    assert np.array_equal(dataset.w[i], shuffled.w[j])


def test_sparse_shuffle():
  """Test that sparse datasets can be shuffled quickly."""
  current_dir = os.path.dirname(os.path.realpath(__file__))

  dataset_file = os.path.join(current_dir, "../../models/tests/example.csv")

  featurizer = dc.feat.CircularFingerprint(size=1024)
  tasks = ["log-solubility"]
  loader = dc.data.CSVLoader(
      tasks=tasks, feature_field="smiles", featurizer=featurizer)
  dataset = loader.create_dataset(dataset_file, shard_size=2)

  X_orig, y_orig, w_orig, orig_ids = (dataset.X, dataset.y, dataset.w,
                                      dataset.ids)
  orig_len = len(dataset)

  dataset.sparse_shuffle()
  X_new, y_new, w_new, new_ids = (dataset.X, dataset.y, dataset.w, dataset.ids)

  assert len(dataset) == orig_len
  # The shuffling should have switched up the ordering
  assert not np.array_equal(orig_ids, new_ids)
  # But all the same entries should still be present
  assert sorted(orig_ids) == sorted(new_ids)
  # All the data should have same shape
  assert X_orig.shape == X_new.shape
  assert y_orig.shape == y_new.shape
  assert w_orig.shape == w_new.shape


def test_shuffle_each_shard():
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
  assert not (ids_s == ids).all()

  # The ids should now store the performed permutation. Check that the
  # original dataset is recoverable.
  for i in range(n_samples):
    np.testing.assert_array_equal(X_s[i], X[ids_s[i]])
    np.testing.assert_array_equal(y_s[i], y[ids_s[i]])
    np.testing.assert_array_equal(w_s[i], w[ids_s[i]])
    np.testing.assert_array_equal(ids_s[i], ids[ids_s[i]])


def test_shuffle_shards():
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
