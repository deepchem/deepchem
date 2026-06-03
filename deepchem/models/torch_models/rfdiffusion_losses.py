"""Losses for RFDiffusion training and evaluation.

This module ships three losses used by the RFDiffusion pipeline
[Watson2023]_:

* :func:`backbone_fape_loss` — Frame-Aligned Point Error on the
  backbone N / CA / C atoms following AlphaFold2 [Jumper2021]_, with the
  standard 10 Å clamp.
* :func:`chi_angle_loss` — periodic ℓ² loss on the side-chain χ-angles
  computed via ``1 − cos(Δχ)`` so that the loss is differentiable and
  periodic.
* :func:`ligand_contact_loss` — Huber-style attraction term encouraging
  specified protein residues to make close contact with a ligand point
  cloud.

All losses accept boolean masks and zero out gradient contributions of
masked elements exactly (i.e. masked-atom gradient must be identically
zero).

Developer-facing summary
------------------------
``backbone_fape_loss`` is the main geometry-alignment loss for frame and
backbone supervision. ``chi_angle_loss`` and ``all_atom_l2_loss`` are
for finer-grained atom / torsion supervision, and
``ligand_contact_loss`` is the contact-style auxiliary term for
ligand-conditioned generation.

References
----------
.. [Watson2023] Watson et al. "De novo design of protein structure and
   function with RFdiffusion." Nature 620 (2023) 1089-1100.
.. [Jumper2021] Jumper et al. "Highly accurate protein structure
   prediction with AlphaFold." Nature 596 (2021) 583-589.
"""

import math
from typing import Optional

try:
    import torch
except ModuleNotFoundError:
    raise ImportError('rfdiffusion_losses requires PyTorch to be installed.')

__all__ = [
    'backbone_fape_loss',
    'chi_angle_loss',
    'ligand_contact_loss',
    'all_atom_l2_loss',
]


def _apply_inverse_rigid(points: torch.Tensor,
                         rotations: torch.Tensor,
                         translations: torch.Tensor) -> torch.Tensor:
    """Apply the inverse rigid transform ``Rᵀ (x − t)`` to a point cloud.

    Parameters
    ----------
    points : torch.Tensor
        Point cloud of shape ``(..., N, 3)``.
    rotations : torch.Tensor
        Rotation matrices of shape ``(..., 3, 3)``.
    translations : torch.Tensor
        Translation vectors of shape ``(..., 3)``.

    Returns
    -------
    torch.Tensor
        Points expressed in the local frame, shape ``(..., N, 3)``.
    """
    centred = points - translations.unsqueeze(-2)
    # x_local = (Rᵀ)·(x − t) — written as a batched matmul.
    return torch.einsum('...ij,...nj->...ni', rotations.transpose(-1, -2),
                        centred)


def backbone_fape_loss(pred_rotations: torch.Tensor,
                       pred_translations: torch.Tensor,
                       true_rotations: torch.Tensor,
                       true_translations: torch.Tensor,
                       pred_atoms: torch.Tensor,
                       true_atoms: torch.Tensor,
                       mask: Optional[torch.Tensor] = None,
                       clamp_distance: float = 10.0,
                       length_scale: float = 10.0,
                       eps: float = 1e-4) -> torch.Tensor:
    """Backbone Frame-Aligned Point Error (FAPE).

    For every frame ``T_i = (R_i, t_i)`` we compute the predicted and
    true positions of every backbone atom expressed in the *local*
    frame of ``T_i``, take the L₂ distance between them, clamp it to
    ``clamp_distance`` (default 10 Å) and average over the i × j pairs.

    Mathematically the per-pair contribution is

    .. math::

        \\ell_{ij}
        = \\min\\!\\big(d_{ij},\\, d_{\\mathrm{clamp}}\\big),\\quad
        d_{ij} = \\| R_i^{\\top}(\\hat x_j - \\hat t_i)
                    - R_i^{*\\top}(x_j^{*} - t_i^{*})\\|_{2}.

    The final loss is the mean of ``ℓ_{ij}`` over unmasked pairs,
    divided by ``length_scale`` so the value is roughly dimensionless.

    Parameters
    ----------
    pred_rotations : torch.Tensor
        Predicted rotation matrices of shape ``(B, L, 3, 3)``.
    pred_translations : torch.Tensor
        Predicted translations of shape ``(B, L, 3)``.
    true_rotations : torch.Tensor
        Ground-truth rotation matrices of the same shape.
    true_translations : torch.Tensor
        Ground-truth translations of the same shape.
    pred_atoms : torch.Tensor
        Predicted backbone atom coordinates of shape ``(B, L, A, 3)``.
    true_atoms : torch.Tensor
        Ground-truth atom coordinates of the same shape.
    mask : torch.Tensor, optional
        Boolean mask of shape ``(B, L)`` flagging valid residues.
    clamp_distance : float, default 10.0
        Distance clamp d_clamp.
    length_scale : float, default 10.0
        Divisor used to scale the averaged distance.
    eps : float, default 1e-4
        Small constant inside the square root for numerical stability.

    Returns
    -------
    torch.Tensor
        Scalar FAPE loss.
    """
    if pred_rotations.shape != true_rotations.shape:
        raise ValueError('Rotation shape mismatch.')
    if pred_translations.shape != true_translations.shape:
        raise ValueError('Translation shape mismatch.')
    if pred_atoms.shape != true_atoms.shape:
        raise ValueError('Atom shape mismatch.')
    if pred_atoms.ndim != 4:
        raise ValueError('pred_atoms must have shape (B, L, A, 3).')
    batch, length, num_atoms, _ = pred_atoms.shape
    # Flatten atoms to a single point cloud per residue (per frame).
    pred_flat = pred_atoms.reshape(batch, length * num_atoms, 3)
    true_flat = true_atoms.reshape(batch, length * num_atoms, 3)
    # Compute local-frame coordinates for every (i, j) pair.
    pred_local = _apply_inverse_rigid(
        pred_flat.unsqueeze(1).expand(-1, length, -1, -1),
        pred_rotations, pred_translations)
    true_local = _apply_inverse_rigid(
        true_flat.unsqueeze(1).expand(-1, length, -1, -1),
        true_rotations, true_translations)
    diff = pred_local - true_local
    dist = torch.sqrt((diff * diff).sum(-1) + eps)  # (B, L, L*A)
    dist = torch.clamp(dist, max=clamp_distance)
    if mask is not None:
        # Per-frame mask broadcast against (L, A) atom mask.
        frame_mask = mask.unsqueeze(-1)  # (B, L, 1)
        atom_mask = mask.unsqueeze(-1).expand(-1, -1, num_atoms).reshape(
            batch, length * num_atoms).unsqueeze(1)  # (B, 1, L*A)
        pair_mask = frame_mask * atom_mask
        dist = dist * pair_mask
        denom = pair_mask.sum().clamp(min=1.0)
        return (dist.sum() / denom) / length_scale
    return dist.mean() / length_scale


def chi_angle_loss(pred_chi: torch.Tensor,
                   true_chi: torch.Tensor,
                   chi_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Periodic loss on side-chain χ-angles.

    Computes ``mean(1 − cos(χ_pred − χ_true))`` over valid entries,
    giving a smooth periodic loss with minimum 0 when angles agree.

    Parameters
    ----------
    pred_chi : torch.Tensor
        Predicted angles in radians; shape ``(..., 4)``.
    true_chi : torch.Tensor
        Ground-truth angles in radians; shape ``(..., 4)``.
    chi_mask : torch.Tensor, optional
        Boolean mask of the same shape flagging valid χ entries.

    Returns
    -------
    torch.Tensor
        Scalar loss in [0, 2].
    """
    if pred_chi.shape != true_chi.shape:
        raise ValueError('χ-angle shape mismatch.')
    residual = 1.0 - torch.cos(pred_chi - true_chi)
    if chi_mask is not None:
        if chi_mask.shape != pred_chi.shape:
            raise ValueError('chi_mask shape mismatch.')
        residual = residual * chi_mask
        denom = chi_mask.sum().clamp(min=1.0)
        return residual.sum() / denom
    return residual.mean()


def ligand_contact_loss(protein_ca: torch.Tensor,
                        ligand_coords: torch.Tensor,
                        contact_mask: torch.Tensor,
                        contact_distance: float = 6.0,
                        slope: float = 1.0) -> torch.Tensor:
    """Soft attraction term toward a ligand point cloud.

    For each residue flagged in ``contact_mask`` the loss is

    .. math::

        \\ell_i = \\max\\!\\big(0,
              \\min_a \\| x_i^{C\\alpha} - y_a \\|_2 - d_{\\rm contact}
        \\big)

    multiplied by ``slope``. The total loss is the mean over the
    flagged residues. Residues outside the mask contribute exactly
    zero (verified by the gradient-flow tests).

    Parameters
    ----------
    protein_ca : torch.Tensor
        Protein Cα coordinates of shape ``(L, 3)`` or ``(B, L, 3)``.
    ligand_coords : torch.Tensor
        Ligand atom coordinates of shape ``(N, 3)`` or ``(B, N, 3)``.
    contact_mask : torch.Tensor
        Boolean mask of shape ``(L,)`` or ``(B, L)`` flagging residues
        that must contact the ligand.
    contact_distance : float, default 6.0
        Distance d_contact at which the loss reaches zero.
    slope : float, default 1.0
        Multiplicative slope ``λ``.

    Returns
    -------
    torch.Tensor
        Scalar contact loss.
    """
    if protein_ca.ndim == 2:
        protein_ca = protein_ca.unsqueeze(0)
        ligand_coords = ligand_coords.unsqueeze(0)
        contact_mask = contact_mask.unsqueeze(0)
    if protein_ca.ndim != 3 or ligand_coords.ndim != 3:
        raise ValueError(
            'protein_ca and ligand_coords must be 2-D or 3-D tensors.')
    if contact_mask.shape != protein_ca.shape[:2]:
        raise ValueError('contact_mask must have shape (B, L).')
    diff = protein_ca.unsqueeze(2) - ligand_coords.unsqueeze(1)
    dist = torch.sqrt((diff * diff).sum(-1) + 1e-4)  # (B, L, N)
    nearest = dist.min(dim=-1).values  # (B, L)
    excess = torch.clamp(nearest - contact_distance, min=0.0)
    mask_f = contact_mask.to(excess.dtype)
    weighted = slope * excess * mask_f
    denom = mask_f.sum().clamp(min=1.0)
    return weighted.sum() / denom


def all_atom_l2_loss(pred_coords: torch.Tensor,
                     true_coords: torch.Tensor,
                     atom_mask: torch.Tensor) -> torch.Tensor:
    """Per-atom L₂ loss with strict atom-mask honouring.

    Computes ``mean( ||pred − true||²_2 )`` over atoms whose
    ``atom_mask`` is True. Masked atoms contribute *exactly* zero — the
    test suite asserts gradient sparsity at the granularity of single
    atoms (no information leak via the mean reduction).

    Parameters
    ----------
    pred_coords : torch.Tensor
        Predicted coordinates of shape ``(..., A, 3)``.
    true_coords : torch.Tensor
        Ground-truth coordinates of the same shape.
    atom_mask : torch.Tensor
        Boolean mask of shape ``(..., A)``. Masked atoms (``False``)
        receive zero gradient.

    Returns
    -------
    torch.Tensor
        Scalar L₂ loss averaged over unmasked atoms.
    """
    if pred_coords.shape != true_coords.shape:
        raise ValueError('Coordinate shape mismatch.')
    if atom_mask.shape != pred_coords.shape[:-1]:
        raise ValueError(
            f'atom_mask shape {atom_mask.shape} does not match '
            f'coords {pred_coords.shape}.')
    mask = atom_mask.to(pred_coords.dtype).unsqueeze(-1)
    diff = (pred_coords - true_coords) * mask
    sq = (diff * diff).sum(-1)
    denom = atom_mask.to(pred_coords.dtype).sum().clamp(min=1.0)
    return sq.sum() / denom


# Expose math.tau so doctest writers do not have to recompute it.
_PI: float = math.pi
