"""
Test for Pytorch Normalizing Flow  model and its transformations
"""
import pytest
import numpy as np

import unittest

try:
  import torch
  from torch.distributions import MultivariateNormal
  from deepchem.models.torch_models.normalizing_flows_pytorch import Affine
  has_torch = True
except:
  has_torch = False


@unittest.skipIf(not has_torch, 'torch is not installed')
@pytest.mark.torch
def test_Affine():
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
