"""All-atom loss functions for RFDiffusion All-Atom.

These are the loss terms RFdiffusion All-Atom-style training adds on top
of the backbone-only diffusion objective once side-chain and/or ligand
atoms are involved [Krishna2024]_, [Watson2023]_:

- :func:`frame_aligned_point_error` -- the FAPE loss from AlphaFold2
  [Jumper2021]_ (Algorithm 21): compares predicted and true atom
  positions after expressing both in every local residue/ligand frame,
  so the loss is invariant to any global rigid motion.
- :func:`dihedral_angle` / :func:`chi_angle_loss` -- a generic 4-atom
  dihedral angle and a masked loss between predicted and true dihedrals,
  usable for side-chain chi angles once 4 atom positions defining that
  angle are available.
- :func:`ligand_clash_loss` -- a soft steric-clash penalty between atom
  pairs that are closer than the sum of their van der Waals radii.
- :func:`masked_all_atom_l2_loss` -- a plain masked coordinate loss for
  supervising individual atom positions directly.

These functions take frames/coordinates as plain tensors rather than
assuming a specific atom14-style per-residue-type atom ordering. RFdiffusion
All-Atom's own reference implementation hardcodes chi-angle atom indices
per amino acid against its internal atom ordering; DeepChem's RFdiffusion
stack does not yet have a full-atom (side-chain) protein representation to
match that ordering against; see [Krishna2024]_ for how that mapping
would need to be sourced once one exists. Every function here is
otherwise the same closed-form math as the reference.

References
----------
.. [Jumper2021] Jumper, J., et al. "Highly accurate protein structure
   prediction with AlphaFold." Nature 596 (2021) 583-589.
.. [Krishna2024] Krishna, R., et al. "Generalized biomolecular modeling
   and design with RoseTTAFold All-Atom." Science 384 (2024) eadl2528.
.. [Watson2023] Watson, J. L., et al. "De novo design of protein
   structure and function with RFdiffusion." Nature 620 (2023) 1089-1100.

Notes
-----
This module requires PyTorch to be installed.
"""

from typing import Dict, Optional

try:
    import torch
except ModuleNotFoundError:
    raise ImportError('rfdiffusion_losses requires PyTorch to be installed.')

__all__ = [
    'DEFAULT_VDW_RADII',
    'chi_angle_loss',
    'dihedral_angle',
    'frame_aligned_point_error',
    'ligand_clash_loss',
    'masked_all_atom_l2_loss',
    'vdw_radii_from_symbols',
]

# Bondi van der Waals radii (Angstrom) for elements common in proteins
# and drug-like ligands [Bondi1964]_. Anything not listed falls back to
# the carbon radius, a reasonable default for an unlisted heavy atom.
#
# .. [Bondi1964] Bondi, A. "van der Waals Volumes and Radii." J. Phys.
#    Chem. 68 (1964) 441-451.
DEFAULT_VDW_RADII: Dict[str, float] = {
    'H': 1.20,
    'C': 1.70,
    'N': 1.55,
    'O': 1.52,
    'F': 1.47,
    'P': 1.80,
    'S': 1.80,
    'Cl': 1.75,
    'Br': 1.85,
    'I': 1.98,
}
_DEFAULT_FALLBACK_RADIUS = DEFAULT_VDW_RADII['C']


def vdw_radii_from_symbols(symbols) -> torch.Tensor:
    """Look up Bondi van der Waals radii for a sequence of element symbols.

    Parameters
    ----------
    symbols : sequence of str
        Element symbols, e.g. ``['C', 'C', 'O', 'N']``.

    Returns
    -------
    torch.Tensor
        Float tensor of shape ``(len(symbols),)`` with each atom's van
        der Waals radius in Angstrom. Unrecognized symbols fall back to
        the carbon radius.

    Examples
    --------
    >>> from deepchem.models.torch_models.rfdiffusion_losses import (
    ...     vdw_radii_from_symbols)
    >>> [round(r, 2) for r in vdw_radii_from_symbols(['C', 'O', 'H']).tolist()]
    [1.7, 1.52, 1.2]
    """
    return torch.tensor(
        [DEFAULT_VDW_RADII.get(s, _DEFAULT_FALLBACK_RADIUS) for s in symbols],
        dtype=torch.float32)


def frame_aligned_point_error(pred_rotations: torch.Tensor,
                              pred_translations: torch.Tensor,
                              true_rotations: torch.Tensor,
                              true_translations: torch.Tensor,
                              pred_positions: torch.Tensor,
                              true_positions: torch.Tensor,
                              frame_mask: Optional[torch.Tensor] = None,
                              position_mask: Optional[torch.Tensor] = None,
                              length_scale: float = 10.0,
                              clamp_distance: Optional[float] = 10.0,
                              eps: float = 1e-8) -> torch.Tensor:
    """Frame Aligned Point Error (FAPE) [Jumper2021]_ Algorithm 21.

    For every (frame, point) pair, both the predicted and the true point
    are expressed in that frame's local coordinates -- ``R^T (x - t)``,
    the same convention as
    :func:`~deepchem.models.torch_models.rfdiffusion_frames.apply_inverse_rigid`
    -- and the Euclidean distance between the two local positions is
    averaged over every unmasked (frame, point) pair. Expressing points
    locally before comparing them makes the loss invariant to any global
    rigid motion applied equally to the predicted and true structures.

    Parameters
    ----------
    pred_rotations : torch.Tensor
        Predicted frame rotations, shape ``(..., F, 3, 3)``.
    pred_translations : torch.Tensor
        Predicted frame origins, shape ``(..., F, 3)``.
    true_rotations : torch.Tensor
        Ground-truth frame rotations, shape ``(..., F, 3, 3)``.
    true_translations : torch.Tensor
        Ground-truth frame origins, shape ``(..., F, 3)``.
    pred_positions : torch.Tensor
        Predicted atom positions, shape ``(..., P, 3)``.
    true_positions : torch.Tensor
        Ground-truth atom positions, shape ``(..., P, 3)``.
    frame_mask : torch.Tensor, optional
        Validity mask over frames, shape ``(..., F)``.
    position_mask : torch.Tensor, optional
        Validity mask over points, shape ``(..., P)``.
    length_scale : float, default 10.0
        Distances are divided by this value (Angstrom) before averaging,
        matching AlphaFold2's ``Z = 10`` normalization.
    clamp_distance : float, optional, default 10.0
        If given, per-pair distances are clamped to this value
        (Angstrom) before normalization, so a handful of very wrong
        points cannot dominate the loss. Pass ``None`` to disable
        clamping.
    eps : float, default 1e-8
        Numerical stability constant for the mask normalization.

    Returns
    -------
    torch.Tensor
        FAPE loss, shape equal to the leading batch shape of the inputs
        (scalar if there is no batch dimension).

    Examples
    --------
    >>> import torch
    >>> from deepchem.models.torch_models.rfdiffusion_frames import (
    ...     make_identity_rigid)
    >>> from deepchem.models.torch_models.rfdiffusion_losses import (
    ...     frame_aligned_point_error)
    >>> R, t = make_identity_rigid((2,))
    >>> points = torch.randn(5, 3)
    >>> loss = frame_aligned_point_error(R, t, R, t, points, points)
    >>> round(float(loss), 6)
    0.0
    """
    pred_rel = pred_positions.unsqueeze(-3) - pred_translations.unsqueeze(-2)
    pred_local = torch.matmul(pred_rel.unsqueeze(-2),
                              pred_rotations.unsqueeze(-3)).squeeze(-2)
    true_rel = true_positions.unsqueeze(-3) - true_translations.unsqueeze(-2)
    true_local = torch.matmul(true_rel.unsqueeze(-2),
                              true_rotations.unsqueeze(-3)).squeeze(-2)

    distances = (pred_local - true_local).norm(dim=-1)  # (..., F, P)
    if clamp_distance is not None:
        distances = distances.clamp(max=clamp_distance)
    distances = distances / length_scale

    mask = torch.ones_like(distances)
    if frame_mask is not None:
        mask = mask * frame_mask.unsqueeze(-1)
    if position_mask is not None:
        mask = mask * position_mask.unsqueeze(-2)

    denom = mask.sum(dim=(-2, -1)).clamp(min=eps)
    return (distances * mask).sum(dim=(-2, -1)) / denom


def dihedral_angle(p0: torch.Tensor,
                   p1: torch.Tensor,
                   p2: torch.Tensor,
                   p3: torch.Tensor,
                   eps: float = 1e-8) -> torch.Tensor:
    """Signed dihedral angle defined by four points.

    Parameters
    ----------
    p0, p1, p2, p3 : torch.Tensor
        Points of shape ``(..., 3)``, in order along the dihedral bond
        p1-p2 (p0 and p3 are the outer substituents).
    eps : float, default 1e-8
        Numerical stability constant.

    Returns
    -------
    torch.Tensor
        Signed angle in radians, shape ``(...,)``.

    Examples
    --------
    >>> import torch
    >>> from deepchem.models.torch_models.rfdiffusion_losses import (
    ...     dihedral_angle)
    >>> p0 = torch.tensor([1.0, 0.0, 0.0])
    >>> p1 = torch.tensor([0.0, 0.0, 0.0])
    >>> p2 = torch.tensor([0.0, 1.0, 0.0])
    >>> p3 = torch.tensor([1.0, 1.0, 0.0])
    >>> round(float(dihedral_angle(p0, p1, p2, p3)), 4)
    0.0
    """
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    b1n = b1 / b1.norm(dim=-1, keepdim=True).clamp(min=eps)
    v = b0 - (b0 * b1n).sum(dim=-1, keepdim=True) * b1n
    w = b2 - (b2 * b1n).sum(dim=-1, keepdim=True) * b1n
    x = (v * w).sum(dim=-1)
    y = (torch.linalg.cross(b1n, v, dim=-1) * w).sum(dim=-1)
    return torch.atan2(y, x)


def chi_angle_loss(pred_atoms: torch.Tensor,
                   true_atoms: torch.Tensor,
                   mask: Optional[torch.Tensor] = None,
                   eps: float = 1e-8) -> torch.Tensor:
    """Masked loss between predicted and true dihedral angles.

    Uses ``1 - cos(delta)`` rather than a squared angle difference so the
    loss is smooth across the +-pi wraparound: two angles that differ by
    a full turn contribute zero loss either way, without the usual "which
    branch of the angle" bookkeeping.

    Parameters
    ----------
    pred_atoms : torch.Tensor
        Four atom positions defining the predicted dihedral, shape
        ``(..., 4, 3)``.
    true_atoms : torch.Tensor
        Four atom positions defining the true dihedral, shape
        ``(..., 4, 3)``.
    mask : torch.Tensor, optional
        Validity mask, shape matching the leading (batch) dims of
        ``pred_atoms`` / ``true_atoms`` (everything but the last two
        axes).
    eps : float, default 1e-8
        Numerical stability constant for the mask normalization.

    Returns
    -------
    torch.Tensor
        Scalar masked-mean loss (or plain mean if ``mask`` is ``None``).

    Examples
    --------
    >>> import torch
    >>> from deepchem.models.torch_models.rfdiffusion_losses import (
    ...     chi_angle_loss)
    >>> atoms = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0],
    ...                       [0.0, 1.0, 0.0], [-1.0, 1.0, 0.0]])
    >>> loss = chi_angle_loss(atoms.unsqueeze(0), atoms.unsqueeze(0))
    >>> round(float(loss), 6)
    0.0
    """
    pred_angle = dihedral_angle(pred_atoms[..., 0, :], pred_atoms[..., 1, :],
                                pred_atoms[..., 2, :], pred_atoms[...,
                                                                  3, :], eps)
    true_angle = dihedral_angle(true_atoms[..., 0, :], true_atoms[..., 1, :],
                                true_atoms[..., 2, :], true_atoms[...,
                                                                  3, :], eps)
    loss = 1.0 - torch.cos(pred_angle - true_angle)
    if mask is not None:
        denom = mask.sum().clamp(min=eps)
        return (loss * mask).sum() / denom
    return loss.mean()


def ligand_clash_loss(coords: torch.Tensor,
                      radii: torch.Tensor,
                      mask: Optional[torch.Tensor] = None,
                      tolerance: float = 0.4,
                      eps: float = 1e-8) -> torch.Tensor:
    """Soft steric-clash penalty between atom pairs.

    Penalizes atom pairs whose distance is smaller than the sum of their
    van der Waals radii minus a small tolerance, with a squared penalty
    on the amount of overlap. This is a simplified, directly
    differentiable stand-in for the full Lennard-Jones potential
    RFdiffusion All-Atom uses internally [Krishna2024]_ -- it captures
    the same "penalize atoms that are too close" idea without the
    attractive well or custom backward pass of a full LJ term.

    Parameters
    ----------
    coords : torch.Tensor
        Atom coordinates, shape ``(..., N, 3)``.
    radii : torch.Tensor
        Per-atom van der Waals radii in the same units as ``coords``
        (Angstrom), shape ``(..., N)`` or broadcastable to it. See
        :func:`vdw_radii_from_symbols` for a convenience lookup.
    mask : torch.Tensor, optional
        Atom validity mask, shape ``(..., N)``.
    tolerance : float, default 0.4
        Overlap (Angstrom) allowed before a pair starts being penalized.
    eps : float, default 1e-8
        Numerical stability constant.

    Returns
    -------
    torch.Tensor
        Mean squared clash penalty over unmasked, non-self atom pairs,
        shape equal to the leading batch shape of ``coords``.

    Examples
    --------
    >>> import torch
    >>> from deepchem.models.torch_models.rfdiffusion_losses import (
    ...     ligand_clash_loss)
    >>> coords = torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    >>> radii = torch.tensor([1.7, 1.7])
    >>> loss = ligand_clash_loss(coords, radii)
    >>> round(float(loss), 6)
    0.0
    """
    n = coords.shape[-2]
    diff = coords.unsqueeze(-2) - coords.unsqueeze(-3)  # (..., N, N, 3)
    dist = diff.norm(dim=-1).clamp(min=eps)

    radii_sum = radii.unsqueeze(-1) + radii.unsqueeze(-2)
    violation = (radii_sum - tolerance - dist).clamp(min=0.0)

    not_self = ~torch.eye(n, dtype=torch.bool, device=coords.device)
    pair_mask = not_self.to(dist.dtype).expand_as(dist).clone()
    if mask is not None:
        pair_mask = pair_mask * mask.unsqueeze(-1) * mask.unsqueeze(-2)

    denom = pair_mask.sum(dim=(-2, -1)).clamp(min=eps)
    return ((violation**2) * pair_mask).sum(dim=(-2, -1)) / denom


def masked_all_atom_l2_loss(pred_coords: torch.Tensor,
                            true_coords: torch.Tensor,
                            mask: Optional[torch.Tensor] = None,
                            eps: float = 1e-8) -> torch.Tensor:
    """Masked mean squared distance between predicted and true atoms.

    Parameters
    ----------
    pred_coords : torch.Tensor
        Predicted atom coordinates, shape ``(..., N, 3)``.
    true_coords : torch.Tensor
        True atom coordinates, shape ``(..., N, 3)``.
    mask : torch.Tensor, optional
        Atom validity mask, shape ``(..., N)``.
    eps : float, default 1e-8
        Numerical stability constant for the mask normalization.

    Returns
    -------
    torch.Tensor
        Masked-mean squared error, shape equal to the leading batch
        shape of the inputs.

    Examples
    --------
    >>> import torch
    >>> from deepchem.models.torch_models.rfdiffusion_losses import (
    ...     masked_all_atom_l2_loss)
    >>> pred = torch.zeros(4, 3)
    >>> true = torch.ones(4, 3)
    >>> mask = torch.tensor([1.0, 1.0, 0.0, 0.0])
    >>> float(masked_all_atom_l2_loss(pred, true, mask))
    3.0
    """
    squared_error = ((pred_coords - true_coords)**2).sum(dim=-1)  # (..., N)
    if mask is not None:
        denom = mask.sum(dim=-1).clamp(min=eps)
        return (squared_error * mask).sum(dim=-1) / denom
    return squared_error.mean(dim=-1)
