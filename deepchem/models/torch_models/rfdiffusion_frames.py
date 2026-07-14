"""SE(3) rigid-frame math utilities for RFDiffusion.

This module contains the non-learned geometric primitives used by the
RFDiffusion stack:

* residue-local frame construction from backbone atoms `(N, CA, C)`
* rigid transform apply / inverse / invert / compose helpers
* Rodrigues exp/log maps between so(3) vectors and SO(3) rotations

The module is intentionally self-contained and depends only on PyTorch.
"""

from typing import Optional, Sequence, Tuple

try:
    import torch
except ModuleNotFoundError:
    raise ImportError("rfdiffusion_frames requires PyTorch to be installed.")

__all__ = [
    "apply_inverse_rigid",
    "apply_rigid",
    "build_backbone_frames",
    "compose_rigids",
    "invert_rigid",
    "make_identity_rigid",
    "so3_exp_map",
    "so3_log_map",
]

_SMALL_OMEGA: float = 1e-4


def _normalize(vector: torch.Tensor, eps: float) -> torch.Tensor:
    """Normalize vectors with epsilon stabilization.

    Parameters
    ----------
    vector : torch.Tensor
        Tensor of shape `(..., 3)`.
    eps : float
        Positive epsilon added to the norm.

    Returns
    -------
    torch.Tensor
        Normalized vectors.
    """
    norm = torch.linalg.norm(vector, dim=-1, keepdim=True)
    return vector / (norm + eps)


def build_backbone_frames(
        backbone: torch.Tensor,
        eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build residue-local frames from `(N, CA, C)` backbone coordinates.

    Parameters
    ----------
    backbone : torch.Tensor
        Backbone coordinates of shape `(..., 3, 3)` ordered as
        `(N, CA, C)`.
    eps : float, default 1e-8
        Positive stabilization constant.

    Returns
    -------
    rotations : torch.Tensor
        Rotation matrices of shape `(..., 3, 3)`.
    translations : torch.Tensor
        Frame origins of shape `(..., 3)`.
    """
    if backbone.shape[-2:] != (3, 3):
        raise ValueError("backbone must have shape (..., 3, 3).")
    if eps <= 0:
        raise ValueError("eps must be positive.")

    n_atom = backbone[..., 0, :]
    ca_atom = backbone[..., 1, :]
    c_atom = backbone[..., 2, :]

    x_axis = _normalize(c_atom - ca_atom, eps)
    n_direction = n_atom - ca_atom
    y_axis = n_direction - (n_direction * x_axis).sum(dim=-1,
                                                      keepdim=True) * x_axis
    y_axis = _normalize(y_axis, eps)
    z_axis = torch.linalg.cross(x_axis, y_axis, dim=-1)

    rotations = torch.stack((x_axis, y_axis, z_axis), dim=-1)
    return rotations, ca_atom


def make_identity_rigid(
        shape: Sequence[int],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create identity rigid transforms `(I, 0)`.

    Parameters
    ----------
    shape : sequence of int
        Batch shape.
    device : torch.device, optional
        Device for returned tensors.
    dtype : torch.dtype, optional
        Tensor dtype.

    Returns
    -------
    rotations : torch.Tensor
        Identity rotations of shape `(*shape, 3, 3)`.
    translations : torch.Tensor
        Zero translations of shape `(*shape, 3)`.
    """
    shape = tuple(shape)
    rotation = torch.eye(3, device=device, dtype=dtype)
    rotation = rotation.expand(*shape, 3, 3).contiguous().clone()
    translation = torch.zeros(*shape, 3, device=device, dtype=dtype)
    return rotation, translation


def _expand_rigid_to_points(
        rotations: torch.Tensor, translations: torch.Tensor,
        points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Broadcast rigid transforms across extra point dimensions."""
    extra_dims = points.dim() - translations.dim()
    if extra_dims < 0:
        raise ValueError(
            "points must have at least the transform batch dimensions.")
    for _ in range(extra_dims):
        rotations = rotations.unsqueeze(-3)
        translations = translations.unsqueeze(-2)
    return rotations, translations


def apply_rigid(rotations: torch.Tensor, translations: torch.Tensor,
                points: torch.Tensor) -> torch.Tensor:
    """Apply rigid transforms `x -> x @ R.T + t`.

    Parameters
    ----------
    rotations : torch.Tensor
        Rotation matrices of shape `(..., 3, 3)`.
    translations : torch.Tensor
        Translation vectors of shape `(..., 3)`.
    points : torch.Tensor
        Points of shape `(..., 3)` or with extra point axes.

    Returns
    -------
    torch.Tensor
        Transformed points.
    """
    rotations, translations = _expand_rigid_to_points(rotations, translations,
                                                      points)
    rotated = torch.matmul(points.unsqueeze(-2),
                           rotations.transpose(-1, -2)).squeeze(-2)
    return rotated + translations


def apply_inverse_rigid(rotations: torch.Tensor, translations: torch.Tensor,
                        points: torch.Tensor) -> torch.Tensor:
    """Apply inverse rigid transforms `y -> (y - t) @ R`."""
    rotations, translations = _expand_rigid_to_points(rotations, translations,
                                                      points)
    centered = points - translations
    return torch.matmul(centered.unsqueeze(-2), rotations).squeeze(-2)


def invert_rigid(
        rotations: torch.Tensor,
        translations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Invert rigid transforms.

    Parameters
    ----------
    rotations : torch.Tensor
        Rotation matrices of shape `(..., 3, 3)`.
    translations : torch.Tensor
        Translation vectors of shape `(..., 3)`.

    Returns
    -------
    inv_rotations : torch.Tensor
        Transposed rotations.
    inv_translations : torch.Tensor
        Inverse translations.
    """
    inv_rotations = rotations.transpose(-1, -2)
    inv_translations = -torch.matmul(translations.unsqueeze(-2),
                                     rotations).squeeze(-2)
    return inv_rotations, inv_translations


def compose_rigids(
        rotations_a: torch.Tensor, translations_a: torch.Tensor,
        rotations_b: torch.Tensor,
        translations_b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compose two rigid transforms `T_a o T_b`.

    Parameters
    ----------
    rotations_a, rotations_b : torch.Tensor
        Rotation matrices of shape `(..., 3, 3)`.
    translations_a, translations_b : torch.Tensor
        Translation vectors of shape `(..., 3)`.

    Returns
    -------
    rotations : torch.Tensor
        Composite rotations.
    translations : torch.Tensor
        Composite translations.
    """
    rotations = torch.matmul(rotations_a, rotations_b)
    translations = apply_rigid(rotations_a, translations_a, translations_b)
    return rotations, translations


def _safe_sin_div_x(x: torch.Tensor) -> torch.Tensor:
    """Return `sin(x) / x` with a stable Taylor branch near zero."""
    small = x.abs() < _SMALL_OMEGA
    safe_x = torch.where(small, torch.ones_like(x), x)
    return torch.where(small, 1.0 - x * x / 6.0, torch.sin(safe_x) / safe_x)


def _safe_one_minus_cos_div_x_sq(x: torch.Tensor) -> torch.Tensor:
    """Return `(1 - cos(x)) / x^2` with a stable Taylor branch."""
    small = x.abs() < _SMALL_OMEGA
    safe_x = torch.where(small, torch.ones_like(x), x)
    closed = (1.0 - torch.cos(safe_x)) / (safe_x * safe_x)
    series = 0.5 - x * x / 24.0
    return torch.where(small, series, closed)


def so3_exp_map(tangent: torch.Tensor) -> torch.Tensor:
    """Map tangent vectors in so(3) to rotation matrices.

    Parameters
    ----------
    tangent : torch.Tensor
        Tangent vectors of shape `(..., 3)`.

    Returns
    -------
    torch.Tensor
        Rotation matrices of shape `(..., 3, 3)`.
    """
    omega = tangent.norm(dim=-1, keepdim=True).clamp(min=0.0)
    zeros = torch.zeros_like(tangent[..., 0])
    tx, ty, tz = tangent[..., 0], tangent[..., 1], tangent[..., 2]
    skew = torch.stack([
        torch.stack([zeros, -tz, ty], dim=-1),
        torch.stack([tz, zeros, -tx], dim=-1),
        torch.stack([-ty, tx, zeros], dim=-1),
    ],
                       dim=-2)
    eye = torch.eye(3, dtype=tangent.dtype, device=tangent.device)
    sin_coeff = _safe_sin_div_x(omega).unsqueeze(-1)
    cos_coeff = _safe_one_minus_cos_div_x_sq(omega).unsqueeze(-1)
    skew_sq = torch.matmul(skew, skew)
    return eye + sin_coeff * skew + cos_coeff * skew_sq


def so3_log_map(rotations: torch.Tensor) -> torch.Tensor:
    """Map rotation matrices to tangent vectors on the principal branch.

    The rotation is converted to a unit quaternion first and then to an
    axis-angle vector. Routing through the quaternion keeps the result stable
    for rotations near ``pi``, where reading the angle straight from the trace
    loses accuracy because ``sin(omega)`` goes to zero.

    Parameters
    ----------
    rotations : torch.Tensor
        Rotation matrices of shape `(..., 3, 3)`.

    Returns
    -------
    torch.Tensor
        Tangent vectors of shape `(..., 3)` whose norm lies in ``[0, pi]``.
    """
    m00 = rotations[..., 0, 0]
    m11 = rotations[..., 1, 1]
    m22 = rotations[..., 2, 2]
    trace = m00 + m11 + m22

    # Read the unit quaternion from the rotation matrix. Each component
    # magnitude comes from the diagonal and its sign from the skew part.
    qw = 0.5 * torch.sqrt(torch.clamp(1.0 + trace, min=0.0))
    qx = 0.5 * torch.sqrt(torch.clamp(1.0 + m00 - m11 - m22, min=0.0))
    qy = 0.5 * torch.sqrt(torch.clamp(1.0 - m00 + m11 - m22, min=0.0))
    qz = 0.5 * torch.sqrt(torch.clamp(1.0 - m00 - m11 + m22, min=0.0))
    qx = torch.copysign(qx, rotations[..., 2, 1] - rotations[..., 1, 2])
    qy = torch.copysign(qy, rotations[..., 0, 2] - rotations[..., 2, 0])
    qz = torch.copysign(qz, rotations[..., 1, 0] - rotations[..., 0, 1])

    quat_vec = torch.stack([qx, qy, qz], dim=-1)
    sin_half = quat_vec.norm(dim=-1)
    cos_half = qw.clamp(-1.0, 1.0)
    omega = 2.0 * torch.atan2(sin_half, cos_half)

    small = sin_half < _SMALL_OMEGA
    safe_sin_half = torch.where(small, torch.ones_like(sin_half), sin_half)
    unit_axis = quat_vec / safe_sin_half.unsqueeze(-1)
    # Near zero rotation the axis is ill-defined; there the tangent vector
    # is approximately 2 * (qx, qy, qz).
    return torch.where(small.unsqueeze(-1), 2.0 * quat_vec,
                       omega.unsqueeze(-1) * unit_axis)
