import numpy as np
import deepchem as dc


def test_train_test_split_basic():

    X = np.random.rand(10, 3)
    y = np.random.rand(10, 1)

    dataset = dc.data.NumpyDataset(X, y)

    train, test = dataset.train_test_split(test_size=0.3)

    assert len(train) == 7
    assert len(test) == 3


def test_train_test_split_alignment():

    X = np.arange(10).reshape(10, 1)
    y = np.arange(10)

    dataset = dc.data.NumpyDataset(X, y)

    train, test = dataset.train_test_split(seed=42)

    for i in range(len(train)):
        assert train.X[i][0] == train.y[i]