import deepchem as dc
import tempfile
import numpy as np
import os


def test_copy():
    """Test that copy works correctly."""
    num_datapoints = 100
    num_features = 10
    num_tasks = 10
    # Generate data
    X = np.random.rand(num_datapoints, num_features)
    y = np.random.randint(2, size=(num_datapoints, num_tasks))
    w = np.random.randint(2, size=(num_datapoints, num_tasks))
    ids = np.array(["id"] * num_datapoints)

    # legacy_dataset_reshard is a shared dataset in the legacy format kept
    # around for testing resharding.
    dataset = dc.data.DiskDataset.from_numpy(X, y, w, ids)
    # Set cache to 0 size to avoid cache hiding errors
    dataset.memory_cache_size = 0

    with tempfile.TemporaryDirectory() as tmpdirname:
        copy = dataset.copy(tmpdirname)
        assert np.all(copy.X == dataset.X)
        assert np.all(copy.y == dataset.y)
        assert np.all(copy.w == dataset.w)
        assert np.all(copy.ids == dataset.ids)


def test_move():
    """Test that move works correctly."""
    num_datapoints = 100
    num_features = 10
    num_tasks = 10
    # Generate data
    X = np.random.rand(num_datapoints, num_features)
    y = np.random.randint(2, size=(num_datapoints, num_tasks))
    w = np.random.randint(2, size=(num_datapoints, num_tasks))
    ids = np.array(["id"] * num_datapoints)

    # legacy_dataset_reshard is a shared dataset in the legacy format kept
    # around for testing resharding.
    dataset = dc.data.DiskDataset.from_numpy(X, y, w, ids)
    # Set cache to 0 size to avoid cache hiding errors
    dataset.memory_cache_size = 0
    data_dir = dataset.data_dir

    with tempfile.TemporaryDirectory() as tmpdirname:
        dataset.move(tmpdirname, delete_if_exists=False)
        assert np.all(X == dataset.X)
        assert np.all(y == dataset.y)
        assert np.all(w == dataset.w)
        assert np.all(ids == dataset.ids)
        assert dataset.data_dir == os.path.join(tmpdirname,
                                                os.path.basename(data_dir))
