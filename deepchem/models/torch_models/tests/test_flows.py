"""File to test the various flows implemented in deepchem
"""

import pytest
import numpy as np

import unittest

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import MultivariateNormal
    from deepchem.models.torch_models.flows import Affine, MaskedAffineFlow, ActNorm
    has_torch = True
except:
    has_torch = False


@unittest.skipIf(not has_torch, 'torch is not installed')
@pytest.mark.torch
def test_Affine():
    """
    This test should evaluate if the transformation its being applied
    correctly. When computing the logarithm of the determinant jacobian matrix
    the result must be zero for any distribution when performing the first forward
    and inverse pass (initialized). This is the expected
    behavior since nothing is being learned yet.

    input shape: (samples, dim)
    output shape: (samples, dim)

    """

    dim = 2
    samples = 96
    data = MultivariateNormal(torch.zeros(dim), torch.eye(dim))
    tensor = data.sample(torch.Size((samples, dim)))
    _, log_det_jacobian = Affine(dim).forward(tensor)
    _, inverse_log_det_jacobian = Affine(dim).inverse(tensor)

    # The first pass of the transformation should be 0
    log_det_jacobian = log_det_jacobian.detach().numpy()
    inverse_log_det_jacobian = inverse_log_det_jacobian.detach().numpy()
    zeros = np.zeros((samples,))

    assert np.array_equal(log_det_jacobian, zeros)
    assert np.array_equal(inverse_log_det_jacobian, zeros)


@unittest.skipIf(not has_torch, 'torch is not installed')
@pytest.mark.torch
def test_Masked_Affine_flow():
    """
    This test should evaluate MaskedAffineFlow.
    """
    dim = 2
    samples = 96
    data = MultivariateNormal(torch.zeros(dim), torch.eye(dim))
    tensor = data.sample(torch.Size((samples, dim)))

    layers = 4
    hidden_size = 16
    masks = F.one_hot(torch.tensor([i % 2 for i in range(layers)])).float()
    s_func = nn.Sequential(
        nn.Linear(in_features=dim, out_features=hidden_size), nn.LeakyReLU(),
        nn.Linear(in_features=hidden_size, out_features=hidden_size),
        nn.LeakyReLU(), nn.Linear(in_features=hidden_size, out_features=dim))
    t_func = nn.Sequential(
        nn.Linear(in_features=dim, out_features=hidden_size), nn.LeakyReLU(),
        nn.Linear(in_features=hidden_size, out_features=hidden_size),
        nn.LeakyReLU(), nn.Linear(in_features=hidden_size, out_features=dim))
    layers = nn.ModuleList(
        [MaskedAffineFlow(mask, s_func, t_func) for mask in masks])
    for layer in layers:
        _, inverse_log_det_jacobian = layer.inverse(tensor)
        inverse_log_det_jacobian = inverse_log_det_jacobian.detach().numpy()
        assert np.any(inverse_log_det_jacobian)


@unittest.skipIf(not has_torch, 'torch is not installed')
@pytest.mark.torch
def test_actnorm():
    """
    This test evaluates ActNorm layer.
    """
    dim = 2
    samples = 96
    data = MultivariateNormal(torch.zeros(dim), torch.eye(dim))
    tensor = data.sample(torch.Size((samples, dim)))

    actnorm = ActNorm(dim)
    _, log_det_jacobian = actnorm.forward(tensor)
    _, inverse_log_det_jacobian = actnorm.inverse(tensor)

    log_det_jacobian = log_det_jacobian.detach().numpy()
    inverse_log_det_jacobian = inverse_log_det_jacobian.detach().numpy()
    ones = np.ones((samples,))
    value = ones * log_det_jacobian[
        0]  # the first pass should have all the values equal

    assert np.array_equal(log_det_jacobian, value)
    assert np.any(inverse_log_det_jacobian)
