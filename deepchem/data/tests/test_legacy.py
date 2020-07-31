import deepchem as dc
import numpy as np


def test_make_legacy_dataset_from_numpy():
  """Test that legacy DiskDataset objects can be constructed."""
  num_datapoints = 100
  num_features = 10
  num_tasks = 10
  # Generate data
  X = np.random.rand(num_datapoints, num_features)
  y = np.random.randint(2, size=(num_datapoints, num_tasks))
  w = np.random.randint(2, size=(num_datapoints, num_tasks))
  ids = np.array(["id"] * num_datapoints)

  dataset = dc.data.DiskDataset.from_numpy(X, y, w, ids, legacy_metadata=True)
  assert dataset.legacy_metadata
  assert len(dataset.metadata_df.columns) == 4
  assert list(dataset.metadata_df.columns) == ['ids', 'X', 'y', 'w']

  # Test constructor reload works for legacy format
  dataset2 = dc.data.DiskDataset(dataset.data_dir)
  assert dataset2.legacy_metadata
  assert len(dataset2.metadata_df.columns) == 4
  assert list(dataset2.metadata_df.columns) == ['ids', 'X', 'y', 'w']


def test_reshard():
  """Test that resharding updates legacy datasets."""
  num_datapoints = 100
  num_features = 10
  num_tasks = 10
  # Generate data
  X = np.random.rand(num_datapoints, num_features)
  y = np.random.randint(2, size=(num_datapoints, num_tasks))
  w = np.random.randint(2, size=(num_datapoints, num_tasks))
  ids = np.array(["id"] * num_datapoints)

  dataset = dc.data.DiskDataset.from_numpy(X, y, w, ids, legacy_metadata=True)
  assert dataset.legacy_metadata
  assert len(dataset.metadata_df.columns) == 4
  assert list(dataset.metadata_df.columns) == ['ids', 'X', 'y', 'w']

  # Reshard this dataset
  dataset.reshard(shard_size=10)
  assert dataset.get_number_shards() == 10
  # Check metadata has been updated
  assert not dataset.legacy_metadata
  assert len(dataset.metadata_df.columns) == 8
  assert list(dataset.metadata_df.columns) == [
      'ids', 'X', 'y', 'w', 'ids_shape', 'X_shape', 'y_shape', 'w_shape'
  ]
