import numpy as np

import deepchem as dc


def test_coulomb_fit_transformer():
    """Test coulomb fit transformer on singletask dataset."""
    n_samples = 10
    n_features = 3
    n_tasks = 1
    ids = np.arange(n_samples)
    X = np.random.rand(n_samples, n_features, n_features)
    y = np.zeros((n_samples, n_tasks))
    w = np.ones((n_samples, n_tasks))
    dataset = dc.data.NumpyDataset(X, y, w, ids)
    fit_transformer = dc.trans.CoulombFitTransformer(dataset)
    X_t = fit_transformer.X_transform(dataset.X)
    assert len(X_t.shape) == 2
