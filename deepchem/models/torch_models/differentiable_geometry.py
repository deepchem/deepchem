"""Differentiable molecular geometry operations for PyTorch."""

import torch
import torch.nn as nn


class DifferentiablePairwiseDistanceLayer(nn.Module):
    """Compute pairwise Euclidean distances from atomic coordinates.

    This layer computes the pairwise distance matrix between all atoms
    in a molecule in a fully differentiable manner, enabling gradient
    flow for molecular geometry optimization tasks.

    Examples
    --------
    >>> import torch
    >>> layer = DifferentiablePairwiseDistanceLayer()
    >>> coords = torch.randn(2, 10, 3, requires_grad=True)
    >>> distances = layer(coords)
    >>> distances.shape
    torch.Size([2, 10, 10])
    """

    def __init__(self):
        super(DifferentiablePairwiseDistanceLayer, self).__init__()

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Compute pairwise distances.

        Parameters
        ----------
        coords : torch.Tensor
            Atomic coordinates of shape (batch, atoms, 3)

        Returns
        -------
        torch.Tensor
            Pairwise distance matrix of shape (batch, atoms, atoms)
        """
        return torch.cdist(coords, coords, p=2)
