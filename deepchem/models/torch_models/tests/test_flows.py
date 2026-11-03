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
    from deepchem.models.torch_models.layers import RealNVPLayer
    from deepchem.models.torch_models.flows import Affine, MaskedAffineFlow, ActNorm, ClampExp, ConstScaleLayer, MLPFlow, NormalizingFlowModel, NormalizingFlow
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


@unittest.skipIf(not has_torch, 'torch is not installed')
@pytest.mark.torch
def test_clampexp():
    """
    This test evaluates the clampexp function.
    """
    lambda_param_list = [0.1, 0.5, 1, 2, 5, 10]

    tensor = torch.tensor([-1, 0.5, 0.6, 0.7])
    outputs = {
        0.1: [0.9048, 1.0000, 1.0000, 1.0000],
        0.5: [0.6065, 1.0000, 1.0000, 1.0000],
        1: [0.3679, 1.0000, 1.0000, 1.0000],
        2: [0.1353, 1.0000, 1.0000, 1.0000],
        5: [0.0067, 1.0000, 1.0000, 1.0000],
        10: [0., 1.0000, 1.0000, 1.0000]
    }
    for lambda_param in lambda_param_list:
        clamp_exp = ClampExp(lambda_param)
        tensor_out = clamp_exp(tensor)
        assert torch.allclose(tensor_out,
                              torch.Tensor(outputs[lambda_param]),
                              atol=1e-4)


@unittest.skipIf(not has_torch, 'torch is not installed')
@pytest.mark.torch
def test_constscalelayer():
    """
    This test evaluates the ConstScaleLayer.
    """
    scale = 2
    const_scale_layer = ConstScaleLayer(scale)
    tensor = torch.tensor([1, 2, 3, 4])
    tensor_out = const_scale_layer(tensor)
    assert torch.allclose(tensor_out, tensor * scale)


@unittest.skipIf(not has_torch, 'torch is not installed')
@pytest.mark.torch
def test_mlp_flow():
    """
    This test evaluates the MLP_flow.
    """
    seed = 42
    layers = [2, 4, 4, 2]
    mlp_flow = MLPFlow(layers)
    torch.manual_seed(seed)
    np.random.seed(seed)
    input_tensor = torch.randn(1, 2)
    output_tensor = mlp_flow(input_tensor)
    assert output_tensor.shape == torch.Size([1, 2])


@unittest.skipIf(not has_torch, 'torch is not installed')
@pytest.mark.torch
def test_normalizing_flow_model():
    """
    This test aims to evaluate if the normalizingFlow model is working correctly.
    """
    nfmodel = NormalizingFlowModel(dim=4, num_layers=2)
    onehots = [[0, 1, 1, 0], [0, 1, 0, 1]]
    input_tensor = torch.tensor(onehots)
    noise_tensor = torch.rand(input_tensor.shape)
    data = torch.add(input_tensor, noise_tensor)
    nfmodel.fit(data, epochs=10, learning_rate=0.001, weight_decay=0.0001)
    gen_mols, _ = nfmodel.nfm.sample(10)

    assert gen_mols.shape == torch.Size([10, 4])


@unittest.skipIf(not has_torch, 'torch is not installed')
@pytest.mark.torch
def test_normalizing_flow_pytorch():
    """
    Test that NormalizingFlow sampling reports the target density.

    Single and composed affine transforms are checked against an independent
    analytic MultivariateNormal oracle, both for densities returned by
    ``sample`` and for ``log_prob`` evaluated at those samples. The gradient
    of the reported mean density with respect to each transform's scale is
    also checked.
    """

    dim = 2
    n_samples = 64

    cases = [
        ([torch.tensor([0.5, -0.3])], [torch.tensor([0.2, -0.1])]),
        ([torch.tensor([0.4, -0.2]),
          torch.tensor([-0.1, 0.25])],
         [torch.tensor([0.1, 0.3]),
          torch.tensor([-0.15, 0.05])]),
    ]

    for case_index, (scale_params, shifts) in enumerate(cases):
        transforms = []
        for scale_param, shift in zip(scale_params, shifts):
            layer = Affine(dim)
            with torch.no_grad():
                layer.scale.copy_(scale_param)
                layer.shift.copy_(shift)
            transforms.append(layer)

        base_distribution = MultivariateNormal(torch.zeros(dim), torch.eye(dim))
        model = NormalizingFlow(transforms, base_distribution, dim)

        # Independent analytic oracle for the composed affine transform.
        mean = torch.zeros(dim)
        std = torch.ones(dim)
        for scale_param, shift in zip(scale_params, shifts):
            scale = torch.exp(scale_param)
            mean = scale * mean + shift
            std = scale * std
        oracle = MultivariateNormal(mean, torch.diag(std * std))

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1000 + case_index)
            samples, log_prob_sample = model.sample(n_samples)

        assert torch.allclose(log_prob_sample,
                              oracle.log_prob(samples),
                              atol=1e-5)
        log_prob_inputs = model.log_prob(samples)
        assert torch.allclose(log_prob_inputs,
                              oracle.log_prob(samples),
                              atol=1e-5)

        log_prob_sample.mean().backward()
        for layer in transforms:
            assert torch.allclose(layer.scale.grad,
                                  torch.full((dim,), -1.0),
                                  atol=1e-5)


@unittest.skipIf(not has_torch, 'torch is not installed')
@pytest.mark.torch
def test_RealNVPLayer():
    """
    Test a concrete 2D RealNVPLayer against an independent dense-Jacobian
    oracle.

    Forward and inverse passes must round-trip, their log-determinants must be
    negatives of one another, and each reported log-determinant must match the
    log absolute determinant of the dense autograd Jacobian computed at the
    same inputs. Fixed nonzero parameters keep the Jacobian from collapsing to
    the identity.
    """

    dim = 2
    n_samples = 4
    hidden_size = 8

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        mask = torch.tensor([1.0, 0.0])
        layer = RealNVPLayer(mask, hidden_size)
        with torch.no_grad():
            layer.scale.fill_(0.7)
        torch.manual_seed(11)
        x = torch.randn(n_samples, dim)

    y, log_det = layer.forward(x)
    x_recovered, inverse_log_det = layer.inverse(y)

    assert torch.allclose(x_recovered, x, atol=1e-5)
    assert torch.allclose(log_det + inverse_log_det,
                          torch.zeros(n_samples),
                          atol=1e-5)

    def forward_single(inp):
        out, _ = layer.forward(inp.unsqueeze(0))
        return out.squeeze(0)

    for i in range(n_samples):
        oracle_log_det = torch.log(
            torch.abs(
                torch.det(
                    torch.autograd.functional.jacobian(forward_single, x[i]))))
        assert torch.allclose(oracle_log_det, log_det[i], atol=1e-5)
