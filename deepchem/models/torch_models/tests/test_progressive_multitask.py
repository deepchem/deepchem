import numpy as np
import deepchem as dc

import pytest

try:
    import torch
    import torch.nn as nn

    has_torch = True
except ModuleNotFoundError:
    has_torch = False
    pass


@pytest.mark.torch
def test_construction():
    """
    Test that PrgressiveMultiTask Model can be constructed without crash.
    """

    model = dc.models.torch_models.ProgressiveMultitask(
        n_tasks=1,
        n_features=100,
        layer_sizes=[128, 256],
        alpha_init_stddevs=[0.08],
        weight_init_stddevs=0.02,
        bias_init_consts=1.0,
        activation_fns=nn.ReLU,
        dropouts=[0.2],
        n_outputs=2,
    )

    assert model is not None


@pytest.mark.torch
def test_compare():
    """
    Test that the PyTorch and TensorFlow versions of ProgressiveMultiTask
    give the same results.
    """

    n_tasks = 2
    n_samples = 20
    n_features = 12
    np.random.seed(123)

    ids = np.arange(n_samples)
    X = np.random.rand(n_samples, n_features)
    y = np.zeros((n_samples, n_tasks))
    w = np.ones((n_samples, n_tasks))
    dataset = dc.data.NumpyDataset(X, y, w, ids)

    # torch_out = torch_model
    tf_model = dc.models.ProgressiveMultitaskRegressor(
        n_tasks=n_tasks,
        n_features=n_features,
        layer_sizes=[128, 256],
        alpha_init_stddevs=0.02,
        weight_init_stddevs=0.02,
        dropouts=0,
    )

    tf_model.fit(dataset, nb_epoch=5)

    torch_model = dc.models.torch_models.ProgressiveMultitask(
        n_tasks=n_tasks,
        n_features=n_features,
        layer_sizes=[128, 256],
        alpha_init_stddevs=0.02,
        weight_init_stddevs=0.02,
        dropouts=0,
    )

    def move_param(tf_param, transpose=False):
        if transpose:
            return nn.Parameter(torch.from_numpy(tf_param.numpy()).T)
        else:
            return nn.Parameter(torch.from_numpy(tf_param.numpy()))

    # Porting the weights from TF to PyTorch
    # task 0 layer 0
    torch_model.layers[0][0][0].weight = move_param(
        tf_model._task_layers[0][0].weights[0], transpose=True)
    torch_model.layers[0][0][0].bias = move_param(
        tf_model._task_layers[0][0].weights[1])

    # task 0 layer 1
    torch_model.layers[0][1][0].weight = move_param(
        tf_model._task_layers[0][1].weights[0], transpose=True)
    torch_model.layers[0][1][0].bias = move_param(
        tf_model._task_layers[0][1].weights[1])

    # task 0 output layer
    torch_model.layers[0][2][0].weight = move_param(
        tf_model._task_layers[0][2].weights[0], transpose=True)
    torch_model.layers[0][2][0].bias = move_param(
        tf_model._task_layers[0][2].weights[1])

    # task 1 layer 0
    torch_model.layers[1][0][0].weight = move_param(
        tf_model._task_layers[1][0].weights[0], transpose=True)
    torch_model.layers[1][0][0].bias = move_param(
        tf_model._task_layers[1][0].weights[1])

    # task 1 layer 1
    torch_model.layers[1][1][0].weight = move_param(
        tf_model._task_layers[1][4].weights[0], transpose=True)
    torch_model.layers[1][1][0].bias = move_param(
        tf_model._task_layers[1][4].weights[1])

    # task 1 output layer
    torch_model.layers[1][2][0].weight = move_param(
        tf_model._task_layers[1][5].weights[0], transpose=True)
    torch_model.layers[1][2][0].bias = move_param(
        tf_model._task_layers[1][5].weights[1])

    # task 1 adapter 0
    torch_model.alphas[0][0] = move_param(
        tf_model._task_layers[1][1].weights[0])
    torch_model.adapters[0][0][0].weight = move_param(
        tf_model._task_layers[1][2].weights[0], transpose=True)
    torch_model.adapters[0][0][0].bias = move_param(
        tf_model._task_layers[1][2].weights[1])
    torch_model.adapters[0][0][2].weight = move_param(
        tf_model._task_layers[1][3].weights[0], transpose=True)

    # task 1 adapter 1
    torch_model.alphas[0][1] = move_param(
        tf_model._task_layers[1][6].weights[0])
    torch_model.adapters[0][1][0].weight = move_param(
        tf_model._task_layers[1][7].weights[0], transpose=True)
    torch_model.adapters[0][1][0].bias = move_param(
        tf_model._task_layers[1][7].weights[1])
    torch_model.adapters[0][1][2].weight = move_param(
        tf_model._task_layers[1][8].weights[0], transpose=True)

    tf_out = tf_model.predict(dataset)
    torch_out = torch_model(torch.from_numpy(X).float())

    assert np.allclose(tf_out, torch_out.cpu().detach().numpy(), atol=1e-4)
