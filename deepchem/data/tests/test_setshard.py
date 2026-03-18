import deepchem as dc
import numpy as np


def test_setshard_with_X_y():
    """Test setsharding on a simple example"""
    X = np.random.rand(10, 3)
    y = np.random.rand(10,)
    dataset = dc.data.DiskDataset.from_numpy(X, y)
    X_shape, y_shape, _, _ = dataset.get_shape()
    assert X_shape[0] == 10
    assert y_shape[0] == 10
    for i, (X, y, w, ids) in enumerate(dataset.itershards()):
        X = X[1:]
        y = y[1:]
        w = w[1:]
        ids = ids[1:]
        dataset.set_shard(i, X, y, w, ids)
    X_shape, y_shape, _, _ = dataset.get_shape()
    assert X_shape[0] == 9
    assert y_shape[0] == 9
