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
def test_forward():
    """
    Test that the forward pass of ProgressiveMultiTask Model can be executed without crash
    and that the output has the correct value.
    """

    n_tasks = 2
    n_features = 12

    torch_model = dc.models.torch_models.ProgressiveMultitask(
        n_tasks=n_tasks,
        n_features=n_features,
        layer_sizes=[128, 256],
        alpha_init_stddevs=0.02,
        weight_init_stddevs=0.02,
        dropouts=0,
    )

    def to_torch_param(weights):
        """Convert numpy weights to torch parameters to be used as model weights"""
        return nn.Parameter(torch.from_numpy(weights))

    weights = np.load(
        "deepchem/models/torch_models/tests/assets/progressive-multitask-sample-weights.npz"
    )
    torch_weights = {
        k: to_torch_param(v) for k, v in weights.items() if k != "output"
    }
    torch_weights["output"] = weights["output"]

    # Porting the weights from TF to PyTorch
    # task 0 layer 0
    torch_model.layers[0][0].weight = torch_weights["layer-0-0-w"]
    torch_model.layers[0][0].bias = torch_weights["layer-0-0-b"]

    # task 0 layer 1
    torch_model.layers[0][1].weight = torch_weights["layer-0-1-w"]
    torch_model.layers[0][1].bias = torch_weights["layer-0-1-b"]

    # task 0 output layer
    torch_model.layers[0][2].weight = torch_weights["layer-0-2-w"]
    torch_model.layers[0][2].bias = torch_weights["layer-0-2-b"]

    # task 1 layer 0
    torch_model.layers[1][0].weight = torch_weights["layer-1-0-w"]
    torch_model.layers[1][0].bias = torch_weights["layer-1-0-b"]

    # task 1 layer 1
    torch_model.layers[1][1].weight = torch_weights["layer-1-1-w"]
    torch_model.layers[1][1].bias = torch_weights["layer-1-1-b"]

    # task 1 output layer
    torch_model.layers[1][2].weight = torch_weights["layer-1-2-w"]
    torch_model.layers[1][2].bias = torch_weights["layer-1-2-b"]

    # task 1 adapter 0
    torch_model.alphas[0][0] = torch_weights["alpha-0-0"]
    torch_model.adapters[0][0][0].weight = torch_weights["adapter-0-0-0-w"]
    torch_model.adapters[0][0][0].bias = torch_weights["adapter-0-0-0-b"]
    torch_model.adapters[0][0][1].weight = torch_weights["adapter-0-0-1-w"]

    # task 1 adapter 1
    torch_model.alphas[0][1] = torch_weights["alpha-0-1"]
    torch_model.adapters[0][1][0].weight = torch_weights["adapter-0-1-0-w"]
    torch_model.adapters[0][1][0].bias = torch_weights["adapter-0-1-0-b"]
    torch_model.adapters[0][1][1].weight = torch_weights["adapter-0-1-1-w"]

    input_x = weights["input"]
    output = torch_weights["output"]
    torch_out = torch_model(
        torch.from_numpy(input_x).float()).cpu().detach().numpy()

    assert np.allclose(output, torch_out,
                       atol=1e-4), "Predictions are not close"
