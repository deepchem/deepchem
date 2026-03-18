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
  This test aims to evaluate if the normalizingFlow model is being applied
  correctly. That is if the sampling, and its log_prob, are being computed
  after performing the transformation layers. Also, if log_prob of an input
  tensor have consistency with the NormalizingFlow model.

  NormalizingFlow:
    sample:
      input shape: (samples)
      output shape: ((samples, dim), (samples))

    log_prob: Method used to learn parameter (optimizing loop)
      input shape: (samples)
      output shape: (samples)

  """

    dim = 2
    samples = 96
    base_distribution = MultivariateNormal(torch.zeros(dim), torch.eye(dim))
    tensor = base_distribution.sample(torch.Size((samples, dim)))
    transformation = [Affine(dim)]
    model = NormalizingFlow(transformation, base_distribution, dim)

    # Test sampling method
    sampling, log_prob_ = model.sample(samples)

    # Test log_prob method (this method is used when inverse pass)
    # Output must be a Nth zero array since nothing is being learned yet
    log_prob = model.log_prob(tensor)

    # Featurize to assert for tests
    log_prob_ = log_prob_.detach().numpy()
    log_prob = log_prob.detach().numpy()
    zeros = np.zeros((samples,))

    # Assert errors for sample method
    assert log_prob_.any()

    # Assert errors for log_prob method
    assert np.array_equal(log_prob, zeros)


@unittest.skipIf(not has_torch, 'torch is not installed')
@pytest.mark.torch
def test_RealNVPLayer():
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

    layers = 4
    hidden_size = 16
    masks = F.one_hot(torch.tensor([i % 2 for i in range(layers)])).float()
    layers = nn.ModuleList([RealNVPLayer(mask, hidden_size) for mask in masks])

    for layer in layers:
        _, inverse_log_det_jacobian = layer.inverse(tensor)
        inverse_log_det_jacobian = inverse_log_det_jacobian.detach().numpy()
        assert np.any(inverse_log_det_jacobian)
