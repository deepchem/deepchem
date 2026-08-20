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

# Threshold for small-angle Taylor expansions.
# For rotation angles theta < 1e-4, sin(theta)/theta ~ 1 - theta^2/6 and
# (1 - cos(theta))/theta^2 ~ 1/2 - theta^2/24. This avoids division-by-zero
# and float32 precision loss near the identity (Grassia, 1998).
_SMALL_OMEGA: float = 1e-4


def _normalize(vector: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize 3D vectors along the last dimension with epsilon stabilization.

    Parameters
    ----------
    vector : torch.Tensor
        Tensor of shape `(..., 3)`.
    eps : float, default 1e-8
        Small positive constant added to the norm denominator.

    Returns
    -------
    torch.Tensor
        Normalized vectors of shape `(..., 3)`.

    Examples
    --------
    >>> import torch
    >>> v = torch.tensor([[3.0, 0.0, 0.0], [0.0, 4.0, 0.0]])
    >>> _normalize(v)
    tensor([[1., 0., 0.],
            [0., 1., 0.]])
    """
    norm = torch.linalg.norm(vector, dim=-1, keepdim=True)
    return vector / (norm + eps)


def build_backbone_frames(
        backbone: torch.Tensor,
        eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor]:
    """Construct residue-local coordinate frames from protein backbone coordinates.

    Builds an orthonormal right-handed reference frame for each residue
    from N, CA, and C atom positions using Gram-Schmidt orthogonalization:
    - The x-axis points along the CA -> C bond.
    - The xy-plane contains the N, CA, and C atoms (with N in the positive y-half).
    - The z-axis is the cross product x x y.
    - The frame origin is positioned at the CA atom.

    Parameters
    ----------
    backbone : torch.Tensor
        Backbone coordinates of shape `(..., 3, 3)` ordered as `(N, CA, C)`
        along the second-to-last dimension.
    eps : float, default 1e-8
        Small positive constant to avoid division by zero during normalization.

    Returns
    -------
    rotations : torch.Tensor
        Rotation matrices of shape `(..., 3, 3)` representing local frame orientations.
    translations : torch.Tensor
        Frame origins of shape `(..., 3)` corresponding to CA coordinates.

    Raises
    ------
    ValueError
        If backbone does not have shape `(..., 3, 3)` or `eps <= 0`.

    Examples
    --------
    >>> import torch
    >>> n = torch.tensor([0.0, 1.0, 0.0])
    >>> ca = torch.tensor([0.0, 0.0, 0.0])
    >>> c = torch.tensor([1.0, 0.0, 0.0])
    >>> backbone = torch.stack([n, ca, c], dim=0)
    >>> R, t = build_backbone_frames(backbone)
    >>> R.shape
    torch.Size([3, 3])
    >>> t.shape
    torch.Size([3])
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
    """Create identity rigid transforms (identity rotation and zero translation).

    Parameters
    ----------
    shape : sequence of int
        Batch shape for the generated transforms.
    device : torch.device, optional
        Target device for the returned tensors.
    dtype : torch.dtype, optional
        Target data type for the returned tensors.

    Returns
    -------
    rotations : torch.Tensor
        Identity rotation matrices of shape `(*shape, 3, 3)`.
    translations : torch.Tensor
        Zero translation vectors of shape `(*shape, 3)`.

    Examples
    --------
    >>> import torch
    >>> R, t = make_identity_rigid((2, 4))
    >>> R.shape
    torch.Size([2, 4, 3, 3])
    >>> t.shape
    torch.Size([2, 4, 3])
    """
    shape = tuple(shape)
    rotation = torch.eye(3, device=device, dtype=dtype)
    rotation = rotation.expand(*shape, 3, 3).contiguous().clone()
    translation = torch.zeros(*shape, 3, device=device, dtype=dtype)
    return rotation, translation


def _expand_rigid_to_points(
        rotations: torch.Tensor, translations: torch.Tensor,
        points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Broadcast rigid transform batch dimensions across extra point dimensions.

    Parameters
    ----------
    rotations : torch.Tensor
        Rotation matrices of shape `(..., 3, 3)`.
    translations : torch.Tensor
        Translation vectors of shape `(..., 3)`.
    points : torch.Tensor
        Points tensor of shape `(..., *extra_dims, 3)`.

    Returns
    -------
    expanded_rotations : torch.Tensor
        Rotations with unsqueezed singleton dimensions matching points.
    expanded_translations : torch.Tensor
        Translations with unsqueezed singleton dimensions matching points.
    """
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
    """Apply rigid transforms to 3D points: `y = points @ R.T + t`.

    Transforms points from local coordinates to global coordinates.

    Parameters
    ----------
    rotations : torch.Tensor
        Rotation matrices of shape `(..., 3, 3)`.
    translations : torch.Tensor
        Translation vectors of shape `(..., 3)`.
    points : torch.Tensor
        Points tensor of shape `(..., 3)` or with extra point axes.

    Returns
    -------
    torch.Tensor
        Transformed points of the same shape as `points`.

    Examples
    --------
    >>> import torch
    >>> R, t = make_identity_rigid((2,))
    >>> pts = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    >>> out = apply_rigid(R, t, pts)
    >>> torch.allclose(out, pts)
    True
    """
    rotations, translations = _expand_rigid_to_points(rotations, translations,
                                                      points)
    rotated = torch.matmul(points.unsqueeze(-2),
                           rotations.transpose(-1, -2)).squeeze(-2)
    return rotated + translations


def apply_inverse_rigid(rotations: torch.Tensor, translations: torch.Tensor,
                        points: torch.Tensor) -> torch.Tensor:
    """Apply inverse rigid transforms to 3D points: `x = (points - t) @ R`.

    Transforms points from global coordinates back to local residue coordinates.

    Parameters
    ----------
    rotations : torch.Tensor
        Rotation matrices of shape `(..., 3, 3)`.
    translations : torch.Tensor
        Translation vectors of shape `(..., 3)`.
    points : torch.Tensor
        Points tensor of shape `(..., 3)` or with extra point axes.

    Returns
    -------
    torch.Tensor
        Transformed points of the same shape as `points`.

    Examples
    --------
    >>> import torch
    >>> R, t = make_identity_rigid((2,))
    >>> pts = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    >>> out = apply_inverse_rigid(R, t, pts)
    >>> torch.allclose(out, pts)
    True
    """
    rotations, translations = _expand_rigid_to_points(rotations, translations,
                                                      points)
    centered = points - translations
    return torch.matmul(centered.unsqueeze(-2), rotations).squeeze(-2)


def invert_rigid(
        rotations: torch.Tensor,
        translations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the inverse of rigid transforms `(R, t)^(-1) = (R.T, -t @ R)`.

    Parameters
    ----------
    rotations : torch.Tensor
        Rotation matrices of shape `(..., 3, 3)`.
    translations : torch.Tensor
        Translation vectors of shape `(..., 3)`.

    Returns
    -------
    inv_rotations : torch.Tensor
        Inverted rotation matrices of shape `(..., 3, 3)`.
    inv_translations : torch.Tensor
        Inverted translation vectors of shape `(..., 3)`.

    Examples
    --------
    >>> import torch
    >>> R = torch.eye(3)
    >>> t = torch.tensor([1.0, 2.0, 3.0])
    >>> inv_R, inv_t = invert_rigid(R, t)
    >>> inv_t
    tensor([-1., -2., -3.])
    """
    inv_rotations = rotations.transpose(-1, -2)
    inv_translations = -torch.matmul(translations.unsqueeze(-2),
                                     rotations).squeeze(-2)
    return inv_rotations, inv_translations


def compose_rigids(
        rotations_a: torch.Tensor, translations_a: torch.Tensor,
        rotations_b: torch.Tensor,
        translations_b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compose two rigid transforms `T_a o T_b = (R_a @ R_b, apply_rigid(T_a, t_b))`.

    Parameters
    ----------
    rotations_a, rotations_b : torch.Tensor
        Rotation matrices of shape `(..., 3, 3)`.
    translations_a, translations_b : torch.Tensor
        Translation vectors of shape `(..., 3)`.

    Returns
    -------
    rotations : torch.Tensor
        Composed rotation matrices of shape `(..., 3, 3)`.
    translations : torch.Tensor
        Composed translation vectors of shape `(..., 3)`.

    Examples
    --------
    >>> import torch
    >>> R1, t1 = make_identity_rigid(())
    >>> R2, t2 = make_identity_rigid(())
    >>> R_comp, t_comp = compose_rigids(R1, t1, R2, t2)
    >>> R_comp.shape
    torch.Size([3, 3])
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
    """Map so(3) tangent vectors (axis-angle) to SO(3) rotation matrices via Rodrigues' formula.

    Parameters
    ----------
    tangent : torch.Tensor
        Tangent vectors of shape `(..., 3)` where direction specifies the
        rotation axis and norm specifies the rotation angle in radians.

    Returns
    -------
    torch.Tensor
        Rotation matrices of shape `(..., 3, 3)`.

    Examples
    --------
    >>> import torch
    >>> w = torch.zeros(3)
    >>> R = so3_exp_map(w)
    >>> torch.allclose(R, torch.eye(3))
    True
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
    """Map SO(3) rotation matrices to so(3) tangent vectors on the principal branch.

    The rotation matrix is converted to a unit quaternion first and then to an
    axis-angle vector. Routing through the quaternion keeps the result stable
    for rotations near pi, where reading the angle from the matrix trace
    loses accuracy because sin(omega) goes to zero.

    Parameters
    ----------
    rotations : torch.Tensor
        Rotation matrices of shape `(..., 3, 3)`.

    Returns
    -------
    torch.Tensor
        Tangent vectors of shape `(..., 3)` whose norm lies in `[0, pi]`.

    Examples
    --------
    >>> import torch
    >>> R = torch.eye(3)
    >>> w = so3_log_map(R)
    >>> torch.allclose(w, torch.zeros(3))
    True
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
