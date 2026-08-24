import numpy as np
import deepchem as dc


def test_numpy_dataset_shuffle_changes_order():

    X = np.arange(10).reshape(10,1)
    y = np.arange(10)

    dataset = dc.data.NumpyDataset(X, y)

    shuffled = dataset.shuffle(seed=42)

    assert not np.array_equal(dataset.X, shuffled.X)


def test_numpy_dataset_shuffle_alignment():

    X = np.arange(10).reshape(10,1)
    y = np.arange(10)

    dataset = dc.data.NumpyDataset(X, y)

    shuffled = dataset.shuffle(seed=42)

    for i in range(len(shuffled.X)):
        assert shuffled.X[i][0] == shuffled.y[i]