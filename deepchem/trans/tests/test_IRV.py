import deepchem as dc
import numpy as np


def test_IRV_transformer():
    n_features = 128
    n_samples = 20
    test_samples = 5
    n_tasks = 2
    X = np.random.randint(2, size=(n_samples, n_features))
    y = np.zeros((n_samples, n_tasks))
    w = np.ones((n_samples, n_tasks))
    dataset = dc.data.NumpyDataset(X, y, w, ids=None)
    X_test = np.random.randint(2, size=(test_samples, n_features))
    y_test = np.zeros((test_samples, n_tasks))
    w_test = np.ones((test_samples, n_tasks))
    test_dataset = dc.data.NumpyDataset(X_test, y_test, w_test, ids=None)
    sims = np.sum(X_test[0, :] * X, axis=1, dtype=float) / np.sum(
        np.sign(X_test[0, :] + X), axis=1, dtype=float)
    sims = sorted(sims, reverse=True)
    IRV_transformer = dc.trans.IRVTransformer(10, n_tasks, dataset)
    test_dataset_trans = IRV_transformer.transform(test_dataset)
    dataset_trans = IRV_transformer.transform(dataset)
    assert test_dataset_trans.X.shape == (test_samples, 20 * n_tasks)
    assert np.allclose(test_dataset_trans.X[0, :10], sims[:10])
    assert np.allclose(test_dataset_trans.X[0, 10:20], [0] * 10)
    assert not np.isclose(dataset_trans.X[0, 0], 1.)
