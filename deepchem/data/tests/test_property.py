import numpy as np
import deepchem as dc


def test_y_property():
    """Test that dataset.y works."""
    num_datapoints = 10
    num_features = 10
    num_tasks = 1
    X = np.random.rand(num_datapoints, num_features)
    y = np.random.randint(2, size=(num_datapoints, num_tasks))
    w = np.ones((num_datapoints, num_tasks))
    ids = np.array(["id"] * num_datapoints)
    dataset = dc.data.DiskDataset.from_numpy(X, y, w, ids)
    y_out = dataset.y
    np.testing.assert_array_equal(y, y_out)


def test_w_property():
    """Test that dataset.y works."""
    num_datapoints = 10
    num_features = 10
    num_tasks = 1
    X = np.random.rand(num_datapoints, num_features)
    y = np.random.randint(2, size=(num_datapoints, num_tasks))
    w = np.ones((num_datapoints, num_tasks))
    ids = np.array(["id"] * num_datapoints)
    dataset = dc.data.DiskDataset.from_numpy(X, y, w, ids)
    w_out = dataset.w
    np.testing.assert_array_equal(w, w_out)
