"""
Normalizing flows for transforming probability distributions using PyTorch.
"""

import logging

import torch
import torch.nn as nn


logger = logging.getLogger(__name__)


class Affine(nn.Module):
  """Class which performs the Affine transformation.

  This transformation is based on the affinity of the base distribution with
  the target distribution. A geometric transformation is applied where
  the parameters performs changes on the scale and shift of a function
  (inputs).

  Normalizing Flow transformations must be bijective in order to compute
  the logartihm of jacobian's determinant. For this reason, transformations
  must perform a forward and inverse pass.

  """

  def __init__(self, dim: int) -> None:
      """Create a Affine transform layer.

      Parameters
      ----------
      dim: int
        Value of the Nth dimension of the dataset.

      """

      super().__init__()
      self.dim = dim
      self.scale = nn.Parameter(torch.zeros(self.dim))
      self.shift = nn.Parameter(torch.zeros(self.dim))

  def forward(self, x):
      y = torch.exp(self.scale) * x + self.shift
      det_jacobian = torch.exp(self.scale.sum())
      log_det_jacobian = torch.ones(y.shape[0]) * torch.log(det_jacobian)

      return y, log_det_jacobian

  def inverse(self, y):
      x = (y - self.shift) / torch.exp(self.scale)
      det_jacobian = 1 / torch.exp(self.scale.sum())
      inverse_log_det_jacobian = torch.ones(y.shape[0]) * torch.log(det_jacobian)

      return x, inverse_log_det_jacobian
