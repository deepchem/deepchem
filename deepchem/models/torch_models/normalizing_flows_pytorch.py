"""
Normalizing flows for transforming probability distributions using PyTorch.
"""

import logging
from typing import Sequence, Tuple

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
  the logarithm of jacobian's determinant. For this reason, transformations
  must perform a forward and inverse pass.

  Example
  --------
  >>>
  >> import deepchem as dc
  >> from deepchem.models.torch_models import Affine
  >> import torch
  >> from torch.distributions import MultivariateNormal
  >> # initialize the transformation layer's parameters
  >> dim = 2
  >> transforms = Affine(2)
  >> # formward pass based on a distribution
  >> distribution = MultivariateNormal(torch.zeros(dim), torch.eye(dim))
  >> transforms.forward(distribution)
  >> # inverse pass based on a distribution
  >> transforms.inverse(distribution)

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

  def forward(self, x: Sequence) -> Tuple[float]:
    """
    Performs a transformation between two different distributions. This
    particular transformation represents the following function:
    y = x*exp(a) + b, where a is scale parameter and b performs a shift.
    This class also returns the logarithm of the jacobians determinant
    which is useful when invert a transformation and compute the
    probability of the transformation.
    """

    y = torch.exp(self.scale) * x + self.shift
    det_jacobian = torch.exp(self.scale.sum())
    log_det_jacobian = torch.ones(y.shape[0]) * torch.log(det_jacobian)

    return y, log_det_jacobian

  def inverse(self, y: Sequence) -> Tuple[float]:
    """
    Performs a transformation between two different distributions.
    This transformation represents the bacward pass of the function
    mention before. Its mathematical representation is x = (y -b)/ exp(a)
    , where "a" is scale parameter and "b" performs a shift. This class
    also returns the logarithm of the jacobians determinant which is
    useful when invert a transformation and compute the probability of
    the transformation.
    """

    x = (y - self.shift) / torch.exp(self.scale)
    det_jacobian = 1 / torch.exp(self.scale.sum())
    inverse_log_det_jacobian = torch.ones(y.shape[0]) * torch.log(det_jacobian)

    return x, inverse_log_det_jacobian
