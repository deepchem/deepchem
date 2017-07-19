"""
Tox21 dataset loader.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import numpy as np
import shutil
import deepchem as dc


def load_sluice():
    sluice_tasks = ['func1', 'func2']
    X = np.random.randint(0, high=10, size=[1000, 1])
    X = X.astype(np.float64)
    y1 = np.copy(X)
    y2 = np.copy(y1)

    y1 += 5
    y2 += 10

    y = np.concatenate((y1, y2), axis=1)

    print(X[:20])
    temp_X = np.zeros((10000, 10))
    for row, value in enumerate(X):
        temp_X[row, X[row, 0]] = 1

    X = temp_X
    X_train = X[:800]
    X_valid = X[800:900]
    X_test = X[900:1000]

    y_train = y[:800]
    y_valid = y[800:900]
    y_test = y[900:1000]

    train = dc.data.NumpyDataset(X=X_train, y=y_train, n_tasks=2)
    valid = dc.data.NumpyDataset(X=X_valid, y=y_valid, n_tasks=2)
    test = dc.data.NumpyDataset(X=X_test, y=y_test, n_tasks=2)

    print('training data shape')
    print(train.get_shape())

    transformers = []
    return sluice_tasks, (train, valid, test), transformers
