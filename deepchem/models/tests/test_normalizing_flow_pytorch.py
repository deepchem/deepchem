"""
Test for Pytorch Normalizing Flow  model and its transformations
"""
import pytest
import numpy as np

import unittest

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import MultivariateNormal
    from deepchem.models.torch_models.layers import Affine, RealNVPLayer
    from deepchem.models.torch_models.normalizing_flows_pytorch import NormalizingFlow
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
