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


class DifferentiableBondAngleLayer(nn.Module):
    """Compute bond angles from atomic coordinates using 3-point geometry.

    This layer computes bond angles for all atom triplets (i-j-k) where
    the angle is centered at atom j. The computation is fully vectorized
    and differentiable, supporting gradient-based optimization.

    Examples
    --------
    >>> import torch
    >>> layer = DifferentiableBondAngleLayer()
    >>> coords = torch.randn(2, 5, 3, requires_grad=True)
    >>> angles = layer(coords)
    >>> angles.shape
    torch.Size([2, 5, 5, 5])
    """

    def __init__(self, eps: float = 1e-7):
        super(DifferentiableBondAngleLayer, self).__init__()
        self.eps = eps

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Compute bond angles for all atom triplets.

        Parameters
        ----------
        coords : torch.Tensor
            Atomic coordinates of shape (batch, atoms, 3)

        Returns
        -------
        torch.Tensor
            Bond angles in radians of shape (batch, atoms, atoms, atoms)
            where angles[b, i, j, k] is the angle i-j-k centered at j
        """
        coords_i = coords.unsqueeze(2).unsqueeze(3)
        coords_j = coords.unsqueeze(1).unsqueeze(3)
        coords_k = coords.unsqueeze(1).unsqueeze(2)

        vec_ji = coords_i - coords_j
        vec_jk = coords_k - coords_j

        norm_ji = torch.norm(vec_ji, dim=-1, keepdim=True).clamp(min=self.eps)
        norm_jk = torch.norm(vec_jk, dim=-1, keepdim=True).clamp(min=self.eps)

        cos_angle = (vec_ji * vec_jk).sum(dim=-1) / (
            norm_ji.squeeze(-1) * norm_jk.squeeze(-1))

        cos_angle = torch.clamp(cos_angle, -1.0 + self.eps, 1.0 - self.eps)

        return torch.acos(cos_angle)
