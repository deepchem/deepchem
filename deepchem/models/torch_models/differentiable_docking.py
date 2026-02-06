"""
Differentiable Docking Module for DeepChem.

This module provides a differentiable implementation of molecular docking
scoring using the Lennard-Jones potential. The differentiability enables
gradient-based optimization of ligand poses.

References
----------
.. [1] Jones, J. E. "On the determination of molecular fields."
    Proceedings of the Royal Society of London A 106.738 (1924): 463-477.
"""
import torch
import torch.nn as nn

__all__ = ['DifferentiableDocking']


def pairwise_distances(coords1: torch.Tensor,
                       coords2: torch.Tensor) -> torch.Tensor:
    """Compute pairwise Euclidean distances between two sets of coordinates.

    This function computes the Euclidean distance between every pair of
    points from two coordinate sets in a fully differentiable manner.

    Parameters
    ----------
    coords1 : torch.Tensor
        First set of coordinates with shape `(N, 3)`.
    coords2 : torch.Tensor
        Second set of coordinates with shape `(M, 3)`.

    Returns
    -------
    torch.Tensor
        A tensor of shape `(N, M)` containing pairwise distances.

    Examples
    --------
    >>> import torch
    >>> coords1 = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    >>> coords2 = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    >>> dists = pairwise_distances(coords1, coords2)
    >>> dists.shape
    torch.Size([2, 2])
    """
    # coords1: (N, 3) -> (N, 1, 3)
    # coords2: (M, 3) -> (1, M, 3)
    # diff: (N, M, 3)
    diff = coords1.unsqueeze(1) - coords2.unsqueeze(0)
    # distances: (N, M)
    distances = torch.sqrt(torch.sum(diff**2, dim=-1) + 1e-8)
    return distances


class DifferentiableDocking(nn.Module):
    """Differentiable docking scoring using Lennard-Jones potential.

    This module computes the interaction energy between a ligand and protein
    using the Lennard-Jones potential. The implementation is fully
    differentiable, enabling gradient-based optimization of ligand poses.

    The Lennard-Jones potential is given by:

    .. math::

        E(r) = \\epsilon \\left[ \\left(\\frac{\\sigma}{r}\\right)^{12}
               - 2\\left(\\frac{\\sigma}{r}\\right)^{6} \\right]

    where:
    - r is the distance between atoms
    - ε (epsilon) is the depth of the potential well
    - σ (sigma) is the distance at which the potential is zero

    Parameters
    ----------
    epsilon : float, optional (default=1.0)
        Depth of the potential well (energy scale). Controls the
        strength of the interaction.
    sigma : float, optional (default=2.0)
        Distance at which the potential is zero (in Angstroms).
        Typical values range from 1.5 to 4.0 for atomic interactions.
    cutoff : float, optional (default=8.0)
        Cutoff distance in Angstroms. Interactions beyond this distance
        are ignored for computational efficiency.
    soft_cutoff : bool, optional (default=False)
        If True, use a smooth sigmoid-based cutoff instead of a hard
        cutoff. This provides smoother gradients at the boundary.
    learnable : bool, optional (default=False)
        If True, epsilon and sigma become learnable parameters that
        can be optimized during training.

    Examples
    --------
    >>> import torch
    >>> from deepchem.models.torch_models import DifferentiableDocking
    >>> # Create module
    >>> docking = DifferentiableDocking(epsilon=1.0, sigma=2.0)
    >>> # Random ligand (10 atoms) and protein (50 atoms)
    >>> ligand_coords = torch.randn(10, 3, requires_grad=True)
    >>> protein_coords = torch.randn(50, 3)
    >>> # Compute energy
    >>> energy = docking(ligand_coords, protein_coords)
    >>> # Verify differentiability
    >>> energy.backward()
    >>> assert ligand_coords.grad is not None

    Notes
    -----
    The Lennard-Jones potential models van der Waals interactions:
    - The r^-12 term represents Pauli repulsion at short distances
    - The r^-6 term represents attractive dispersion forces

    For docking applications, the ligand coordinates are typically
    optimized while keeping the protein coordinates fixed.

    References
    ----------
    .. [1] Jones, J. E. "On the determination of molecular fields."
        Proceedings of the Royal Society of London A 106.738 (1924): 463-477.
    """

    def __init__(self,
                 epsilon: float = 1.0,
                 sigma: float = 2.0,
                 cutoff: float = 8.0,
                 soft_cutoff: bool = False,
                 learnable: bool = False):
        super(DifferentiableDocking, self).__init__()

        self.cutoff = cutoff
        self.soft_cutoff = soft_cutoff
        self.learnable = learnable

        if learnable:
            self.epsilon = nn.Parameter(
                torch.tensor(epsilon, dtype=torch.float32))
            self.sigma = nn.Parameter(torch.tensor(sigma, dtype=torch.float32))
        else:
            self.register_buffer('epsilon',
                                 torch.tensor(epsilon, dtype=torch.float32))
            self.register_buffer('sigma',
                                 torch.tensor(sigma, dtype=torch.float32))

    def forward(self, ligand_coords: torch.Tensor,
                protein_coords: torch.Tensor) -> torch.Tensor:
        """Compute the interaction energy between ligand and protein.

        Parameters
        ----------
        ligand_coords : torch.Tensor
            Ligand atom coordinates with shape `(N, 3)`, where N is
            the number of ligand atoms.
        protein_coords : torch.Tensor
            Protein atom coordinates with shape `(M, 3)`, where M is
            the number of protein atoms.

        Returns
        -------
        torch.Tensor
            Scalar tensor containing the total interaction energy.

        Raises
        ------
        ValueError
            If input coordinates do not have shape (N, 3) or (M, 3).
        """
        # Input validation
        if ligand_coords.ndim != 2 or ligand_coords.shape[1] != 3:
            raise ValueError("ligand_coords must have shape (N, 3)")
        if protein_coords.ndim != 2 or protein_coords.shape[1] != 3:
            raise ValueError("protein_coords must have shape (M, 3)")

        # Compute pairwise distances: (N, M)
        distances = pairwise_distances(ligand_coords, protein_coords)

        # Clamp distances to prevent division by zero and NaN gradients
        distances = torch.clamp(distances, min=1e-6)

        # Apply cutoff mask for efficiency
        if self.soft_cutoff:
            # Smooth sigmoid-based cutoff for nicer gradients
            mask = torch.sigmoid(10.0 * (self.cutoff - distances))
        else:
            # Hard cutoff (faster, but discontinuous at boundary)
            mask = (distances < self.cutoff)

        # Compute Lennard-Jones potential
        # E(r) = epsilon * [(sigma/r)^12 - 2*(sigma/r)^6]
        sigma_over_r = self.sigma / distances
        sigma_over_r_6 = sigma_over_r**6
        sigma_over_r_12 = sigma_over_r_6**2

        lj_energy = self.epsilon * (sigma_over_r_12 - 2 * sigma_over_r_6)

        # Apply mask: only count interactions within cutoff
        masked_energy = lj_energy * mask

        # Sum all pairwise interactions
        total_energy = torch.sum(masked_energy)

        return total_energy

    def optimize_pose(self,
                      ligand_coords: torch.Tensor,
                      protein_coords: torch.Tensor,
                      n_steps: int = 100,
                      learning_rate: float = 0.01) -> torch.Tensor:
        """Optimize ligand pose to minimize interaction energy.

        This method performs gradient descent on the ligand coordinates
        to find a lower energy configuration.

        Parameters
        ----------
        ligand_coords : torch.Tensor
            Initial ligand atom coordinates with shape `(N, 3)`.
        protein_coords : torch.Tensor
            Protein atom coordinates with shape `(M, 3)`.
        n_steps : int, optional (default=100)
            Number of optimization steps.
        learning_rate : float, optional (default=0.01)
            Step size for gradient descent.

        Returns
        -------
        torch.Tensor
            Optimized ligand coordinates with shape `(N, 3)`.

        Examples
        --------
        >>> import torch
        >>> docking = DifferentiableDocking()
        >>> ligand = torch.randn(10, 3)
        >>> protein = torch.randn(50, 3)
        >>> optimized_ligand = docking.optimize_pose(ligand, protein, n_steps=50)
        >>> optimized_ligand.shape
        torch.Size([10, 3])
        """
        # Clone and enable gradients
        coords = ligand_coords.clone().detach().requires_grad_(True)
        protein_coords = protein_coords.detach()

        optimizer = torch.optim.SGD([coords], lr=learning_rate)

        for _ in range(n_steps):
            optimizer.zero_grad()
            energy = self.forward(coords, protein_coords)
            energy.backward()
            optimizer.step()

        return coords.detach()
