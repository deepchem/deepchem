import deepchem as dc
import numpy as np


def test_numpy_dataset_get_shape():
  """Test that get_shape works for numpy datasets."""
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


def test_disk_dataset_get_shape_single_shard():
  """Test that get_shape works for disk dataset."""
  num_datapoints = 100
  num_features = 10
  num_tasks = 10
  # Generate data
  X = np.random.rand(num_datapoints, num_features)
  y = np.random.randint(2, size=(num_datapoints, num_tasks))
  w = np.random.randint(2, size=(num_datapoints, num_tasks))
  ids = np.array(["id"] * num_datapoints)

  dataset = dc.data.DiskDataset.from_numpy(X, y, w, ids)

  X_shape, y_shape, w_shape, ids_shape = dataset.get_shape()
  assert X_shape == X.shape
  assert y_shape == y.shape
  assert w_shape == w.shape
  assert ids_shape == ids.shape


def test_disk_dataset_get_shape_multishard():
  """Test that get_shape works for multisharded disk dataset."""
  num_datapoints = 100
  num_features = 10
  num_tasks = 10
  # Generate data
  X = np.random.rand(num_datapoints, num_features)
  y = np.random.randint(2, size=(num_datapoints, num_tasks))
  w = np.random.randint(2, size=(num_datapoints, num_tasks))
  ids = np.array(["id"] * num_datapoints)

  dataset = dc.data.DiskDataset.from_numpy(X, y, w, ids)
  # Should now have 10 shards
  dataset.reshard(shard_size=10)

  X_shape, y_shape, w_shape, ids_shape = dataset.get_shape()
  assert X_shape == X.shape
  assert y_shape == y.shape
  assert w_shape == w.shape
  assert ids_shape == ids.shape


def test_disk_dataset_get_legacy_shape_single_shard():
  """Test that get_shape works for legacy disk dataset."""
  num_datapoints = 100
  num_features = 10
  num_tasks = 10
  # Generate data
  X = np.random.rand(num_datapoints, num_features)
  y = np.random.randint(2, size=(num_datapoints, num_tasks))
  w = np.random.randint(2, size=(num_datapoints, num_tasks))
  ids = np.array(["id"] * num_datapoints)

  dataset = dc.data.DiskDataset.from_numpy(X, y, w, ids, legacy_metadata=True)

  X_shape, y_shape, w_shape, ids_shape = dataset.get_shape()
  assert X_shape == X.shape
  assert y_shape == y.shape
  assert w_shape == w.shape
  assert ids_shape == ids.shape


def test_disk_dataset_get_legacy_shape_multishard():
  """Test that get_shape works for multisharded legacy disk dataset."""
  num_datapoints = 100
  num_features = 10
  num_tasks = 10
  # Generate data
  X = np.random.rand(num_datapoints, num_features)
  y = np.random.randint(2, size=(num_datapoints, num_tasks))
  w = np.random.randint(2, size=(num_datapoints, num_tasks))
  ids = np.array(["id"] * num_datapoints)

  dataset = dc.data.DiskDataset.from_numpy(X, y, w, ids, legacy_metadata=True)
  # Should now have 10 shards
  dataset.reshard(shard_size=10)

  X_shape, y_shape, w_shape, ids_shape = dataset.get_shape()
  assert X_shape == X.shape
  assert y_shape == y.shape
  assert w_shape == w.shape
  assert ids_shape == ids.shape
