"""Symmetry helpers for RFDiffusion: enforcing Cₙ, Dₙ, T, O, I groups.

RFDiffusion enforces point-group symmetry on generated assemblies by
projecting per-residue updates (frames or coordinates) onto the
symmetric subspace defined by a group ``G`` [Watson2023]_. Concretely,
if ``L`` residues form ``|G|`` copies of an asymmetric unit of size
``L / |G|``, we average the per-update gradients across the orbit of
every residue under ``G`` so that all copies remain related by the
same rigid motion.

Notation
--------
* ``G`` — point group of order ``|G|``.
* ``U`` — asymmetric-unit size, with ``L = |G| · U``.
* Residue ``(g, u)`` denotes the ``u``-th residue of the ``g``-th
  copy. Indexing convention: ``index(g, u) = g · U + u``.

Implementation notes
--------------------
The orbit of residue ``u`` is the set ``{(0, u), (1, u), …, (|G|−1, u)}``.
The symmetric-projection operator ``Π`` averages per-orbit translations
and rotations *expressed in a common reference frame*, then re-applies
the group element to obtain each copy.

Developer-facing summary
------------------------
The group-construction helpers build canonical rotation sets such as
``cyclic_group(n)`` or ``dihedral_group(n)``. During generation, call
``symmetrise_coords`` or ``symmetrise_frames`` after each update step
to keep all copies of the asymmetric unit locked to the same symmetry.

References
----------
.. [Watson2023] Watson et al. "De novo design of protein structure and
   function with RFdiffusion." Nature 620 (2023) 1089-1100.
"""

import math
from typing import Iterable, List

try:
    import torch
except ModuleNotFoundError:
    raise ImportError(
        'rfdiffusion_symmetry requires PyTorch to be installed.')

__all__ = [
    'cyclic_group',
    'dihedral_group',
    'tetrahedral_group',
    'octahedral_group',
    'icosahedral_group',
    'symmetrise_frames',
    'symmetrise_coords',
]


# ---------------------------------------------------------------------
# Group generators
# ---------------------------------------------------------------------
def _axis_rotation(axis: torch.Tensor, angle: float) -> torch.Tensor:
    """Build a rotation matrix from an axis and angle (Rodrigues)."""
    axis = axis / axis.norm()
    x, y, z = axis.tolist()
    c = math.cos(angle)
    s = math.sin(angle)
    one_minus_c = 1.0 - c
    return torch.tensor([
        [c + x * x * one_minus_c, x * y * one_minus_c - z * s,
         x * z * one_minus_c + y * s],
        [y * x * one_minus_c + z * s, c + y * y * one_minus_c,
         y * z * one_minus_c - x * s],
        [z * x * one_minus_c - y * s, z * y * one_minus_c + x * s,
         c + z * z * one_minus_c],
    ], dtype=torch.float64)


def _close_group(generators: Iterable[torch.Tensor],
                 max_order: int = 128,
                 tol: float = 1e-8) -> List[torch.Tensor]:
    """Close a set of rotation generators into a finite group.

    Repeatedly composes the current set with the generators until no
    new elements are discovered or ``max_order`` is reached.
    """
    group: List[torch.Tensor] = []

    def _add(g: torch.Tensor) -> bool:
        for existing in group:
            if (g - existing).abs().max().item() < tol:
                return False
        group.append(g)
        return True

    eye = torch.eye(3, dtype=torch.float64)
    _add(eye)
    pending = list(generators)
    for gen in pending:
        _add(gen)
    changed = True
    while changed:
        changed = False
        for h in list(group):
            for g in pending:
                product = g @ h
                if _add(product):
                    changed = True
                    if len(group) > max_order:
                        raise RuntimeError(
                            f'Group exceeded max_order {max_order}.')
    return group


def cyclic_group(n: int) -> torch.Tensor:
    """Cyclic group :math:`C_n` of rotations about the z-axis.

    Parameters
    ----------
    n : int
        Order of the cyclic group (n ≥ 1).

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(n, 3, 3)`` containing all elements of
        :math:`C_n`.
    """
    if n < 1:
        raise ValueError('Cyclic order must be >= 1.')
    z = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    elements = [_axis_rotation(z, 2.0 * math.pi * k / n) for k in range(n)]
    return torch.stack(elements, dim=0)


def dihedral_group(n: int) -> torch.Tensor:
    """Dihedral group :math:`D_n`: ``C_n`` plus n perpendicular C₂ axes.

    Parameters
    ----------
    n : int
        Order parameter (n ≥ 2). The full group has order 2n.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(2n, 3, 3)``.
    """
    if n < 2:
        raise ValueError('Dihedral order must be >= 2.')
    z = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    x = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    generators = [_axis_rotation(z, 2.0 * math.pi / n),
                  _axis_rotation(x, math.pi)]
    group = _close_group(generators, max_order=4 * n)
    return torch.stack(group, dim=0)


def tetrahedral_group() -> torch.Tensor:
    """Rotational tetrahedral group T (order 12).

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(12, 3, 3)``.
    """
    # Generators: 3-fold about (1,1,1)/√3, 2-fold about z.
    z = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    diag = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
    gens = [_axis_rotation(diag, 2.0 * math.pi / 3.0),
            _axis_rotation(z, math.pi)]
    group = _close_group(gens, max_order=24)
    return torch.stack(group, dim=0)


def octahedral_group() -> torch.Tensor:
    """Rotational octahedral group O (order 24).

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(24, 3, 3)``.
    """
    z = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    diag = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
    gens = [_axis_rotation(z, math.pi * 0.5),
            _axis_rotation(diag, 2.0 * math.pi / 3.0)]
    group = _close_group(gens, max_order=48)
    return torch.stack(group, dim=0)


def icosahedral_group() -> torch.Tensor:
    """Rotational icosahedral group I (order 60).

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(60, 3, 3)``.
    """
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    # 5-fold axis through (1, φ, 0) / norm.
    five_fold = torch.tensor([1.0, phi, 0.0], dtype=torch.float64)
    # 3-fold axis through (1, 1, 1).
    three_fold = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
    gens = [_axis_rotation(five_fold, 2.0 * math.pi / 5.0),
            _axis_rotation(three_fold, 2.0 * math.pi / 3.0)]
    group = _close_group(gens, max_order=120)
    return torch.stack(group, dim=0)


# ---------------------------------------------------------------------
# Projection operators
# ---------------------------------------------------------------------
def _validate(group: torch.Tensor, frames_or_coords: torch.Tensor) -> int:
    if group.ndim != 3 or group.shape[-2:] != (3, 3):
        raise ValueError('group must have shape (|G|, 3, 3).')
    g = int(group.shape[0])
    length = frames_or_coords.shape[-2] if frames_or_coords.ndim >= 2 \
        else frames_or_coords.shape[0]
    if length % g != 0:
        raise ValueError(
            f'Sequence length {length} is not divisible by |G|={g}.')
    return g


def symmetrise_coords(coords: torch.Tensor,
                      group: torch.Tensor) -> torch.Tensor:
    """Project coordinates onto the symmetric subspace.

    Each asymmetric-unit copy is forced to be the rigid image of the
    same reference unit under the corresponding group element. The
    returned coords satisfy

    .. math::

        x^{\\text{sym}}_{(g, u)} = R_g\\, \\bar x_u,
        \\quad \\bar x_u = \\frac{1}{|G|}
            \\sum_{g'} R_{g'}^{\\top}\\, x_{(g', u)}

    Parameters
    ----------
    coords : torch.Tensor
        Coordinates of shape ``(..., L, 3)`` where ``L = |G| · U``.
    group : torch.Tensor
        Rotation matrices of the group, shape ``(|G|, 3, 3)``.

    Returns
    -------
    torch.Tensor
        Symmetrised coordinates of the same shape.
    """
    g = _validate(group, coords)
    *batch_shape, length, dim = coords.shape
    if dim != 3:
        raise ValueError('coords last dim must be 3.')
    unit = length // g
    coords = coords.to(dtype=group.dtype if coords.dtype != group.dtype
                       else coords.dtype)
    grouped = coords.reshape(*batch_shape, g, unit, 3)
    # Express every copy in the reference frame: x_ref_g = R_g^T x_g.
    rgt = group.transpose(-1, -2)  # (G, 3, 3)
    rotated_back = torch.einsum('gij,...guj->...gui', rgt, grouped)
    averaged = rotated_back.mean(dim=-3, keepdim=True)  # (..., 1, U, 3)
    # Re-apply each group element to the averaged unit.
    out = torch.einsum('gij,...uj->...gui', group, averaged.squeeze(-3))
    return out.reshape(*batch_shape, length, 3).to(dtype=coords.dtype)


def symmetrise_frames(rotations: torch.Tensor,
                      translations: torch.Tensor,
                      group: torch.Tensor) -> 'tuple[torch.Tensor, torch.Tensor]':
    """Project rigid frames onto the symmetric subspace.

    The translations are averaged in the reference frame using
    :func:`symmetrise_coords`; the rotations are projected onto SO(3)
    by averaging the rotation matrices in the reference frame and
    re-orthogonalising via SVD.

    Parameters
    ----------
    rotations : torch.Tensor
        Rotation matrices of shape ``(..., L, 3, 3)``.
    translations : torch.Tensor
        Translation vectors of shape ``(..., L, 3)``.
    group : torch.Tensor
        Rotation matrices of the group, shape ``(|G|, 3, 3)``.

    Returns
    -------
    tuple of torch.Tensor
        ``(R_sym, t_sym)`` of the same shapes as the inputs.
    """
    g = _validate(group, translations)
    *batch_shape, length, _ = translations.shape
    unit = length // g
    rotations = rotations.to(dtype=group.dtype)
    translations = translations.to(dtype=group.dtype)
    # Translations: use the coordinate symmetriser.
    t_sym = symmetrise_coords(translations, group)
    # Rotations: average R_g^T R_{g,u} in the reference frame, then
    # re-apply R_g and re-orthogonalise.
    grouped = rotations.reshape(*batch_shape, g, unit, 3, 3)
    rgt = group.transpose(-1, -2)
    ref_rots = torch.einsum('gij,...gujk->...guik', rgt, grouped)
    averaged = ref_rots.mean(dim=-4)  # (..., U, 3, 3)
    # SVD-based projection onto SO(3): R = U diag(1,1,det) Vᵀ.
    u_mat, _, v_mat = torch.linalg.svd(averaged)
    det = torch.det(u_mat @ v_mat)
    signs = torch.ones_like(det)
    signs = signs * det.sign().where(det.abs() > 1e-8, torch.ones_like(det))
    diag = torch.diag_embed(torch.stack(
        [torch.ones_like(signs), torch.ones_like(signs), signs], dim=-1))
    ref_rot = u_mat @ diag @ v_mat
    out = torch.einsum('gij,...ujk->...guik', group, ref_rot)
    r_sym = out.reshape(*batch_shape, length, 3, 3)
    return r_sym.to(dtype=rotations.dtype), t_sym.to(dtype=translations.dtype)
