# This file is part of DeepChem.
#
# Author: SIDDHANT JAIN
# License: MIT

import torch
import torch.nn as nn


def polynomial_envelope(length: torch.Tensor,
                        max_length: float,
                        p: int = 5) -> torch.Tensor:
    """Computes the polynomial envelope function.

    This function ensures that atomic interactions decay smoothly to zero at the
    cutoff distance (max_length). It is based on Equation 8 from the MACE paper
    (Klicpera et al., ICLR 2020).

    Parameters
    ----------
    length : torch.Tensor
        Input distances tensor.
    max_length : float
        The cutoff distance (r_cut).
    p : int, optional
        Power parameter controlling the steepness of the envelope (default 5).

    Returns
    -------
    torch.Tensor
        The envelope values with the same shape as `length`.
    """
    if not isinstance(length, torch.Tensor):
        length = torch.tensor(length)

    # Normalized distance (r / r_cut)
    r_normalized = length / max_length

    # Polynomial terms
    c1 = (p + 1.0) * (p + 2.0) / 2.0
    c2 = p * (p + 2.0)
    c3 = p * (p + 1.0) / 2.0

    term1 = c1 * torch.pow(r_normalized, p)
    term2 = c2 * torch.pow(r_normalized, p + 1)
    term3 = c3 * torch.pow(r_normalized, p + 2)

    envelope = 1.0 - term1 + term2 - term3

    # Apply mask to ensure values beyond cutoff are exactly 0.0
    mask = (length < max_length).float()
    return envelope * mask


class BesselBasis(nn.Module):
    """Bessel Basis function for radial expansion.

    Expands interatomic distances into a set of orthogonal sine waves (Bessel functions).
    This serves as the initial feature representation for atomic bonds.

    Parameters
    ----------
    r_max : float
        The cutoff distance.
    num_basis : int
        The number of basis functions (frequencies) to use.
    """

    def __init__(self, r_max: float, num_basis: int = 8):
        super(BesselBasis, self).__init__()
        self.r_max = r_max
        self.num_basis = num_basis

        # Precompute frequencies (n * pi)
        # Register as buffer to ensure it moves to GPU with the model
        n = torch.arange(1, num_basis + 1).float()
        self.register_buffer('n_pi', n * torch.pi)

        # Precompute normalization factor
        self.prefactor = (2.0 / r_max)**0.5

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """Compute the Bessel Basis expansion.

        Parameters
        ----------
        r : torch.Tensor
            Input distances tensor of shape [..., N].

        Returns
        -------
        torch.Tensor
            Basis expansion of shape [..., N, num_basis].
        """
        # Expand dims for broadcasting: [..., 1]
        r_expanded = r.unsqueeze(-1)

        # Compute (n * pi * r / r_max)
        arg = self.n_pi * (r_expanded / self.r_max)

        # Compute sin(x)/x, handling division by zero with epsilon
        eps = 1e-6
        return self.prefactor * torch.sin(arg) / (r_expanded + eps)


class RadialEmbedding(nn.Module):
    """Radial Embedding Block for MLIPs.

    Combines the Bessel Basis expansion with the Polynomial Envelope to create
    smooth, cutoff-aware radial features.

    Parameters
    ----------
    r_max : float
        The cutoff distance.
    num_basis : int
        Number of basis functions.
    p : int
        Envelope power parameter.
    """

    def __init__(self, r_max: float, num_basis: int = 8, p: int = 5):
        super(RadialEmbedding, self).__init__()
        self.bessel_fn = BesselBasis(r_max=r_max, num_basis=num_basis)
        self.r_max = r_max
        self.p = p

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """Compute radial embeddings.

        Parameters
        ----------
        r : torch.Tensor
            Input distances.

        Returns
        -------
        torch.Tensor
            Radial features of shape [..., num_basis].
        """
        basis = self.bessel_fn(r)
        env = polynomial_envelope(r, self.r_max, self.p)

        # Reshape envelope to [..., 1] to broadcast over basis dimension
        env = env.unsqueeze(-1)

        return basis * env
