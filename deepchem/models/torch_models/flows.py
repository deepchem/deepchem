""" This module contains the implementation of the various flow layers and models"""
import torch
import torch.nn as nn
from typing import Sequence, Tuple


class Flow(nn.Module):
    """
    Generic class for flow functions
    """

    def __init__(self):
        """Initializes the flow function
        """
        super().__init__()

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the flow

        Parameters
        ----------
        z: torch.Tensor
            Input tensor

        Returns
        -------
        z_: torch.Tensor
            Transformed tensor
        log_det: torch.Tensor
            Logarithm of the determinant of the Jacobian of the transformation
        """
        raise NotImplementedError("Forward pass has not been implemented.")

    def inverse(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inverse pass of the flow

        Parameters
        ----------
        z: torch.Tensor
            Input tensor

        Returns
        -------
        z_: torch.Tensor
            Transformed tensor
        log_det: torch.Tensor
            Logarithm of the determinant of the Jacobian of the transformation
        """
        raise NotImplementedError("This flow has no algebraic inverse.")


# Adding duplicate class here, the layers `Affine` class to be deprecated
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
    >>> import deepchem as dc
    >>> from deepchem.models.torch_models.layers import Affine
    >>> import torch
    >>> from torch.distributions import MultivariateNormal
    >>> # initialize the transformation layer's parameters
    >>> dim = 2
    >>> samples = 96
    >>> transforms = Affine(dim)
    >>> # forward pass based on a given distribution
    >>> distribution = MultivariateNormal(torch.zeros(dim), torch.eye(dim))
    >>> input = distribution.sample(torch.Size((samples, dim)))
    >>> len(transforms.forward(input))
    2
    >>> # inverse pass based on a distribution
    >>> len(transforms.inverse(input))
    2

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

    def forward(self, x: Sequence) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs a transformation between two different distributions. This
        particular transformation represents the following function:
        y = x * exp(a) + b, where a is scale parameter and b performs a shift.
        This class also returns the logarithm of the jacobians determinant
        which is useful when invert a transformation and compute the
        probability of the transformation.

        Parameters
        ----------
        x : Sequence
            Tensor sample with the initial distribution data which will pass into
            the normalizing flow algorithm.

        Returns
        -------
        y : torch.Tensor
            Transformed tensor according to Affine layer with the shape of 'x'.
        log_det_jacobian : torch.Tensor
            Tensor which represents the info about the deviation of the initial
            and target distribution.

        """

        y = torch.exp(self.scale) * x + self.shift
        det_jacobian = torch.exp(self.scale.sum())
        log_det_jacobian = torch.ones(y.shape[0]) * torch.log(det_jacobian)

        return y, log_det_jacobian

    def inverse(self, y: Sequence) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs a transformation between two different distributions.
        This transformation represents the bacward pass of the function
        mention before. Its mathematical representation is x = (y - b) / exp(a)
        , where "a" is scale parameter and "b" performs a shift. This class
        also returns the logarithm of the jacobians determinant which is
        useful when invert a transformation and compute the probability of
        the transformation.

        Parameters
        ----------
        y : Sequence
            Tensor sample with transformed distribution data which will be used in
            the normalizing algorithm inverse pass.

        Returns
        -------
        x : torch.Tensor
            Transformed tensor according to Affine layer with the shape of 'y'.
        inverse_log_det_jacobian : torch.Tensor
            Tensor which represents the information of the deviation of the initial
            and target distribution.

        """

        x = (y - self.shift) / torch.exp(self.scale)
        det_jacobian = 1 / torch.exp(self.scale.sum())
        inverse_log_det_jacobian = torch.ones(
            x.shape[0]) * torch.log(det_jacobian)

        return x, inverse_log_det_jacobian
