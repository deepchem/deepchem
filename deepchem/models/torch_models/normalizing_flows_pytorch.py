"""Normalizing flows for transforming probability distributions using PyTorch.
"""
import torch
from torch import nn
from typing import Sequence, Tuple


class NormalizingFlow(nn.Module):
    """Normalizing flows are widley used to perform generative models.
    This algorithm gives advantages over variational autoencoders (VAE) because
    of ease in sampling by applying invertible transformations
    (Frey, Gadepally, & Ramsundar, 2022).

    Example
    --------
    >>> import deepchem as dc
    >>> from deepchem.models.torch_models.layers import Affine
    >>> from deepchem.models.torch_models.normalizing_flows_pytorch import NormalizingFlow
    >>> import torch
    >>> from torch.distributions import MultivariateNormal
    >>> # initialize the transformation layer's parameters
    >>> dim = 2
    >>> samples = 96
    >>> transforms = [Affine(dim)]
    >>> distribution = MultivariateNormal(torch.zeros(dim), torch.eye(dim))
    >>> # initialize normalizing flow model
    >>> model = NormalizingFlow(transforms, distribution, dim)
    >>> # evaluate the log_prob when applying the transformation layers
    >>> input = distribution.sample(torch.Size((samples, dim)))
    >>> len(model.log_prob(input))
    96
    >>> # evaluates the the sampling method and its log_prob
    >>> len(model.sample(samples))
    2

    """

    def __init__(self, transform: Sequence, base_distribution,
                 dim: int) -> None:
        """This class considers a transformation, or a composition of transformations
        functions (layers), between a base distribution and a target distribution.

        Parameters
        ----------
        transform: Sequence
            Bijective transformation/transformations which are considered the layers
            of a Normalizing Flow model.
        base_distribution: torch.Tensor
            Probability distribution to initialize the algorithm. The Multivariate Normal
            distribution is mainly used for this parameter.
        dim: int
            Value of the Nth dimension of the dataset.

        """
        super().__init__()
        self.dim = dim
        self.transforms = nn.ModuleList(transform)
        self.base_distribution = base_distribution

    def log_prob(self, inputs: torch.Tensor) -> torch.Tensor:
        """This method computes the probability of the inputs when
        transformation/transformations are applied.

        Parameters
        ----------
        inputs: torch.Tensor
            Tensor used to evaluate the log_prob computation of the learned
            distribution.
            shape: (samples, dim)

        Returns
        -------
        log_prob: torch.Tensor
            This tensor contains the value of the log probability computed.
            shape: (samples)

        """
        log_prob = torch.zeros(inputs.shape[0])
        for biject in reversed(self.transforms):
            inputs, inverse_log_det_jacobian = biject.inverse(inputs)
            log_prob += inverse_log_det_jacobian

        return log_prob

    def sample(self, n_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs a sampling from the transformed distribution.
        Besides the outputs (sampling), this method returns the logarithm of
        probability to obtain the outputs at the base distribution.

        Parameters
        ----------
        n_samples: int
            Number of samples to select from the transformed distribution

        Returns
        -------
        sample: tuple
            This tuple contains a two torch.Tensor objects. The first represents
            a sampling of the learned distribution when transformations had been
            applied. The secong torc.Tensor is the computation of log probabilities
            of the transformed distribution.
            shape: ((samples, dim), (samples))

        """
        outputs = self.base_distribution.sample((n_samples,))
        log_prob = self.base_distribution.log_prob(outputs)

        for biject in self.transforms:
            outputs, log_det_jacobian = biject.forward(outputs)
            log_prob += log_det_jacobian

        return outputs, log_prob
