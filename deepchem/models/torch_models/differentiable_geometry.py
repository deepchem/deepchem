"""Differentiable molecular geometry operations for PyTorch."""

from typing import Optional
import torch
import torch.nn as nn


class DifferentiablePairwiseDistanceLayer(nn.Module):
    """Batched pairwise distance computation for molecular geometries.

    This layer provides a reusable component for computing pairwise distance 
    matrices in molecular modeling workflows. Unlike a raw torch.cdist call, 
    this layer ensures consistent batch handling, validates molecular geometry 
    inputs, and supports configurable distance metrics commonly used in 
    computational chemistry (L1, L2, Linf).

    The layer is designed for use in graph neural networks, molecular dynamics,
    and geometry optimization pipelines where distances must be computed 
    repeatedly with gradient tracking.

    Parameters
    ----------
    p : float, optional (default=2.0)
        Distance norm to compute. Common values:
        - 1.0: Manhattan distance
        - 2.0: Euclidean distance (default)
        - float('inf'): Chebyshev distance

    Examples
    --------
    >>> import torch
    >>> layer = DifferentiablePairwiseDistanceLayer()
    >>> coords = torch.randn(2, 10, 3, requires_grad=True)
    >>> distances = layer(coords)
    >>> distances.shape
    torch.Size([2, 10, 10])
    >>> 
    >>> # Manhattan distance
    >>> layer_l1 = DifferentiablePairwiseDistanceLayer(p=1.0)
    >>> distances_l1 = layer_l1(coords)
    """

    def __init__(self, p: float = 2.0):
        super(DifferentiablePairwiseDistanceLayer, self).__init__()
        self.p = p

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Compute pairwise distances for batched molecular coordinates.

        Parameters
        ----------
        coords : torch.Tensor
            Atomic coordinates of shape (batch_size, num_atoms, coordinate_dims)
            Typically coordinate_dims=3 for 3D Cartesian coordinates.

        Returns
        -------
        torch.Tensor
            Pairwise distance matrix of shape (batch_size, num_atoms, num_atoms)
            where output[b, i, j] is the distance between atoms i and j in batch b.

        Raises
        ------
        ValueError
            If input tensor does not have exactly 3 dimensions or if batch_size < 1.
        """
        if coords.dim() != 3:
            raise ValueError(
                f"Expected 3D input tensor (batch_size, num_atoms, coordinate_dims), "
                f"got {coords.dim()}D tensor with shape {coords.shape}")

        if coords.size(0) < 1:
            raise ValueError(
                f"Batch size must be at least 1, got {coords.size(0)}")

        return torch.cdist(coords, coords, p=self.p)
