import deepchem as dc
import numpy as np


def test_reshard_with_X():
  """Test resharding on a simple example"""
  X = np.random.rand(100, 10)
  dataset = dc.data.DiskDataset.from_numpy(X)
  assert dataset.get_number_shards() == 1
  dataset.reshard(shard_size=10)
  assert (dataset.X == X).all()
  assert dataset.get_number_shards() == 10


def test_reshard_with_X_y():
  """Test resharding on a simple example"""
  X = np.random.rand(100, 10)
  y = np.random.rand(100,)
  dataset = dc.data.DiskDataset.from_numpy(X, y)
  assert dataset.get_number_shards() == 1
  dataset.reshard(shard_size=10)
  assert (dataset.X == X).all()
  # This is necessary since from_numpy adds in shape information
  assert (dataset.y.flatten() == y).all()
  assert dataset.get_number_shards() == 10


def test_reshard_with_X_y_generative():
  """Test resharding for a hypothetical generative dataset."""
  X = np.random.rand(100, 10, 10)
  y = np.random.rand(100, 10, 10)
  dataset = dc.data.DiskDataset.from_numpy(X, y)
  assert (dataset.X == X).all()
  assert (dataset.y == y).all()
  assert dataset.get_number_shards() == 1
  dataset.reshard(shard_size=10)
  assert (dataset.X == X).all()
  assert (dataset.y == y).all()
  assert dataset.get_number_shards() == 10


def test_reshard_with_X_y_w():
  """Test resharding on a simple example"""
  X = np.random.rand(100, 10)
  y = np.random.rand(100,)
  w = np.ones_like(y)
  dataset = dc.data.DiskDataset.from_numpy(X, y, w)
  assert dataset.get_number_shards() == 1
  dataset.reshard(shard_size=10)
  assert (dataset.X == X).all()
  # This is necessary since from_numpy adds in shape information
  assert (dataset.y.flatten() == y).all()
  assert (dataset.w.flatten() == w).all()
  assert dataset.get_number_shards() == 10


def test_reshard_with_X_y_w_ids():
  """Test resharding on a simple example"""
  X = np.random.rand(100, 10)
  y = np.random.rand(100,)
  w = np.ones_like(y)
  ids = np.arange(100)
  dataset = dc.data.DiskDataset.from_numpy(X, y, w, ids)
  assert dataset.get_number_shards() == 1
  dataset.reshard(shard_size=10)
  assert (dataset.X == X).all()
  # This is necessary since from_numpy adds in shape information
  assert (dataset.y.flatten() == y).all()
  assert (dataset.w.flatten() == w).all()
  assert (dataset.ids == ids).all()
  assert dataset.get_number_shards() == 10
