import os
import numpy as np
import pytest
from scipy import io as scipy_io

from deepchem.data import NumpyDataset

from deepchem.models.torch_models import DTNNModel

@pytest.mark.torch
def test_dtnn():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(current_dir, "assets/example_DTNN.mat")
    dataset = scipy_io.loadmat(input_file)
    X = dataset['X']
    y = dataset['T']
    w = np.ones_like(y)
    dataset = NumpyDataset(X, y, w, ids=None)
    n_tasks = y.shape[1]
    model = DTNNModel(n_tasks,
                      n_embedding=20,
                      n_distance=100,
                      learning_rate=1.0,
                      mode="regression")

    print(model)
    # Fit trained model
    model.fit(dataset, nb_epoch=250)
    # Eval model on train
    pred = model.predict(dataset)
    mean_rel_error = np.mean(np.abs(1 - pred / y))
    assert mean_rel_error < 0.1

test_dtnn()