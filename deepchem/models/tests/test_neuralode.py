import deepchem as dc
import numpy as np
import pytest
from deepchem.models import NeuralODEModel

try:
    import torch

    has_torch = True
except:
    has_torch = False


@pytest.mark.torch
def test_neuralodemodel():
    """Test that a 1D neural ode can overfit simple regression datasets."""
    n_samples = 10
    n_features = 1
    n_tasks = 1

    np.random.seed(123)
    X = np.random.rand(n_samples, 10, n_features)
    y = np.random.randint(2, size=(n_samples, n_tasks)).astype(np.float32)
    dataset = dc.data.NumpyDataset(X, y)
    model = NeuralODEModel(n_features=n_features)

    regression_metric = dc.metrics.Metric(dc.metrics.mean_squared_error)

    # Fit trained model
    model.fit(dataset, nb_epoch=200)

    # Eval model on train
    scores = model.evaluate(dataset, [regression_metric])
    assert scores[regression_metric.name] < 0.1
