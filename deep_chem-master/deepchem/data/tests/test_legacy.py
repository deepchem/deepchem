import os
import deepchem as dc
import numpy as np
import tempfile


def test_make_legacy_dataset_from_numpy():
  """Test that legacy DiskDataset objects can be constructed."""
  current_dir = os.path.dirname(os.path.abspath(__file__))
  # legacy_dataset is a dataset in the legacy format kept around for testing purposes.
  data_dir = os.path.join(current_dir, "legacy_dataset")
  dataset = dc.data.DiskDataset(data_dir)
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
  # legacy_dataset_reshard is a sharded dataset in the legacy format kept
  # around for testing resharding.
  current_dir = os.path.dirname(os.path.abspath(__file__))
  data_dir = os.path.join(current_dir, "legacy_dataset_reshard")
  dataset = dc.data.DiskDataset(data_dir)
  assert dataset.legacy_metadata
  assert len(dataset.metadata_df.columns) == 4
  assert list(dataset.metadata_df.columns) == ['ids', 'X', 'y', 'w']

  with tempfile.TemporaryDirectory() as tmpdirname:
    copy = dataset.copy(tmpdirname)
    assert np.all(copy.X == dataset.X)
    assert np.all(copy.y == dataset.y)
    assert np.all(copy.w == dataset.w)
    assert np.all(copy.ids == dataset.ids)

    # Reshard copy
    copy.reshard(shard_size=10)
    assert copy.get_number_shards() == 10
    # Check metadata has been updated
    assert not copy.legacy_metadata
    assert len(copy.metadata_df.columns) == 8
    assert list(copy.metadata_df.columns) == [
        'ids', 'X', 'y', 'w', 'ids_shape', 'X_shape', 'y_shape', 'w_shape'
    ]
