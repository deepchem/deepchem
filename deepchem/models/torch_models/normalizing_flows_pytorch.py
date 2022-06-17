"""Normalizing flows for transforming probability distributions using PyTorch.
"""
import torch
from torch import nn
from typing import Sequence, Tuple


class NormalizingFlow(nn.Module):
  """Normalizing flows are widley used to perform generative models.
  Normalizing flow models gives advantages over variational autoencoders
  (VAE) because of ease in sampling by applying invertible transformations
  (Frey, Gadepally, & Ramsundar, 2022)."""

  def __init__(self, transform: Sequence, base_distribution: torch.Tensor, dim: int) -> None:
    """This class considers a transformation, or a composition of transformations
    functions (layers), between a base distribuiton and a target distribution.

    Parameters
    ----------
    transform: Sequence
      Bijective transformation/transformations which are considered the layers
      of a Normalizinf Flow model.
    base_distribution: torch.Tensor
      Probability distribution to initialize the algorithm. The Multivariate Normal
      distribution is mainly used at this parameter.
    dim: int
      Value of the Nth dimension of the dataset.

    """
    super().__init__()
    self.dim = dim
    self.transforms = nn.ModuleList(transform)
    self.base_distribution = base_distribution

  def log_prob(self, inputs: Sequence) -> torch.Tensor:
    """This method computes the probabilty of the inputs when
    transformation/transformations are applied.

    inputs shape: (samples, dim)
    log_prob shape: (samples, dim)
    """
    log_prob = torch.zeros(inputs.shape[0])
    for biject in reversed(self.transforms):
        inputs, inverse_log_det_jacobian = biject.inverse(inputs)
        log_prob += inverse_log_det_jacobian

    return log_prob

  def sample(self, n_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """This method performs a sampling from the transformed distribution.
    Besides the outputs (sampling), this method returns the logarithm of
    probability to obtain the outputs at the base distribution.

    n_samples shape: (samples)
    output shape: (n_samples, dim)
    log_prob shape: (n_samples)
    """
    outputs = self.base_distribution.sample((n_samples, ))
    log_prob = self.base_distribution.log_prob(outputs)

    for biject in self.transforms:
        outputs, log_det_jacobian = biject.forward(outputs)
        log_prob += log_det_jacobian

    return outputs, log_prob
