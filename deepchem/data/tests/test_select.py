import deepchem as dc
import numpy as np
import os


def test_select():
    """Test that dataset select works."""
    num_datapoints = 10
    num_features = 10
    num_tasks = 1
    X = np.random.rand(num_datapoints, num_features)
    y = np.random.randint(2, size=(num_datapoints, num_tasks))
    w = np.ones((num_datapoints, num_tasks))
    ids = np.array(["id"] * num_datapoints)
    dataset = dc.data.DiskDataset.from_numpy(X, y, w, ids)

    indices = [0, 4, 5, 8]
    select_dataset = dataset.select(indices)
    assert isinstance(select_dataset, dc.data.DiskDataset)
    X_sel, y_sel, w_sel, ids_sel = (select_dataset.X, select_dataset.y,
                                    select_dataset.w, select_dataset.ids)
    np.testing.assert_array_equal(X[indices], X_sel)
    np.testing.assert_array_equal(y[indices], y_sel)
    np.testing.assert_array_equal(w[indices], w_sel)
    np.testing.assert_array_equal(ids[indices], ids_sel)


def test_image_dataset_select():
    """Test that select works on image datasets."""
    path = os.path.join(os.path.dirname(__file__), 'images')
    files = [os.path.join(path, f) for f in os.listdir(path)]
    dataset = dc.data.ImageDataset(files, np.random.random(10))
    indices = [0, 4, 5, 8, 2]
    select_dataset = dataset.select(indices)
    assert isinstance(select_dataset, dc.data.ImageDataset)
    X_sel, y_sel, w_sel, ids_sel = (select_dataset.X, select_dataset.y,
                                    select_dataset.w, select_dataset.ids)
    np.testing.assert_array_equal(dataset.X[indices], X_sel)
    np.testing.assert_array_equal(dataset.y[indices], y_sel)
    np.testing.assert_array_equal(dataset.w[indices], w_sel)
    np.testing.assert_array_equal(dataset.ids[indices], ids_sel)


def test_numpy_dataset_select():
    """Test that dataset select works with numpy dataset."""
    num_datapoints = 10
    num_features = 10
    num_tasks = 1
    X = np.random.rand(num_datapoints, num_features)
    y = np.random.randint(2, size=(num_datapoints, num_tasks))
    w = np.ones((num_datapoints, num_tasks))
    ids = np.array(["id"] * num_datapoints)
    dataset = dc.data.NumpyDataset(X, y, w, ids)

    indices = [0, 4, 5, 8, 2]
    select_dataset = dataset.select(indices)
    assert isinstance(select_dataset, dc.data.NumpyDataset)
    X_sel, y_sel, w_sel, ids_sel = (select_dataset.X, select_dataset.y,
                                    select_dataset.w, select_dataset.ids)
    np.testing.assert_array_equal(X[indices], X_sel)
    np.testing.assert_array_equal(y[indices], y_sel)
    np.testing.assert_array_equal(w[indices], w_sel)
    np.testing.assert_array_equal(ids[indices], ids_sel)


def test_select_multishard():
    """Test that dataset select works with multiple shards."""
    num_datapoints = 100
    num_features = 10
    num_tasks = 1
    X = np.random.rand(num_datapoints, num_features)
    y = np.random.randint(2, size=(num_datapoints, num_tasks))
    w = np.ones((num_datapoints, num_tasks))
    ids = np.array(["id"] * num_datapoints)
    dataset = dc.data.DiskDataset.from_numpy(X, y, w, ids)
    dataset.reshard(shard_size=10)

    indices = [10, 42, 51, 82, 2, 4, 6]
    select_dataset = dataset.select(indices)
    assert isinstance(select_dataset, dc.data.DiskDataset)
    X_sel, y_sel, w_sel, ids_sel = (select_dataset.X, select_dataset.y,
                                    select_dataset.w, select_dataset.ids)
    np.testing.assert_array_equal(X[indices], X_sel)
    np.testing.assert_array_equal(y[indices], y_sel)
    np.testing.assert_array_equal(w[indices], w_sel)
    np.testing.assert_array_equal(ids[indices], ids_sel)


def test_select_not_sorted():
    """Test that dataset select with ids not in sorted order."""
    num_datapoints = 10
    num_features = 10
    num_tasks = 1
    X = np.random.rand(num_datapoints, num_features)
    y = np.random.randint(2, size=(num_datapoints, num_tasks))
    w = np.ones((num_datapoints, num_tasks))
    ids = np.array(["id"] * num_datapoints)
    dataset = dc.data.DiskDataset.from_numpy(X, y, w, ids)

    indices = [4, 2, 8, 5, 0]
    select_dataset = dataset.select(indices)
    assert isinstance(select_dataset, dc.data.DiskDataset)
    X_sel, y_sel, w_sel, ids_sel = (select_dataset.X, select_dataset.y,
                                    select_dataset.w, select_dataset.ids)
    np.testing.assert_array_equal(X[indices], X_sel)
    np.testing.assert_array_equal(y[indices], y_sel)
    np.testing.assert_array_equal(w[indices], w_sel)
    np.testing.assert_array_equal(ids[indices], ids_sel)


def test_select_to_numpy():
    """Test that dataset select works."""
    num_datapoints = 10
    num_features = 10
    num_tasks = 1
    X = np.random.rand(num_datapoints, num_features)
    y = np.random.randint(2, size=(num_datapoints, num_tasks))
    w = np.ones((num_datapoints, num_tasks))
    ids = np.array(["id"] * num_datapoints)
    dataset = dc.data.DiskDataset.from_numpy(X, y, w, ids)

    indices = [0, 4, 5, 8]
    select_dataset = dataset.select(indices, output_numpy_dataset=True)
    assert isinstance(select_dataset, dc.data.NumpyDataset)
    X_sel, y_sel, w_sel, ids_sel = (select_dataset.X, select_dataset.y,
                                    select_dataset.w, select_dataset.ids)
    np.testing.assert_array_equal(X[indices], X_sel)
    np.testing.assert_array_equal(y[indices], y_sel)
    np.testing.assert_array_equal(w[indices], w_sel)
    np.testing.assert_array_equal(ids[indices], ids_sel)
