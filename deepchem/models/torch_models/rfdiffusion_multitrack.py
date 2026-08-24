"""Frame-based multi-track denoiser that connects RFDiffusion's
RoseTTAFold-style network to combined SO(3) + translation diffusion.

The pieces built in earlier PRs -- rigid frames (``rfdiffusion_frames``),
Invariant Point Attention (``rfdiffusion_ipa``), the pair-track blocks
(``rfdiffusion_pair_track``), the sequence/structure track and
``RFDiffusionMultiTrackStack`` (``rfdiffusion_sequence_track``), and IGSO(3)
rotational diffusion (``rfdiffusion_so3``) -- are all standalone modules.
This module is the glue: it wraps ``RFDiffusionMultiTrackStack`` into a
denoiser that predicts a full backbone frame (rotation + translation) per
residue, and provides the forward-noising and reverse-sampling steps needed
to actually run combined rotational/translational diffusion end to end.

Unlike the coordinate-space baseline in
:mod:`deepchem.models.torch_models.layers` (``BackboneDiffusion``), which
predicts additive Gaussian noise on flat ``(N, CA, C)`` coordinates, this
denoiser follows RFdiffusion's own parameterization: the network predicts
the denoised structure ``x0`` (here, per-residue rigid frames) directly at
every step, matching the original repository's ``px0`` convention.

References
----------
.. [Watson2023] Watson, J. L., et al. "De novo design of protein structure
   and function with RFdiffusion." Nature 620.7976 (2023): 1089-1100.
.. [Yim2023] Yim, J., et al. "SE(3) diffusion model with application to
   protein backbone generation." ICML 2023.
.. [Jumper2021] Jumper, J., et al. "Highly accurate protein structure
   prediction with AlphaFold." Nature 596 (2021): 583-589.

Notes
-----
This module requires PyTorch to be installed.
"""

from typing import List, Optional, Tuple

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:
    raise ImportError(
        'rfdiffusion_multitrack requires PyTorch to be installed.')

from deepchem.models.torch_models.layers import (
    CosineSchedule,
    PositionalEncoding,
    ResidueEmbedding,
    SinusoidalTimestepEmbedding,
)
from deepchem.models.torch_models.rfdiffusion_frames import apply_rigid
from deepchem.models.torch_models.rfdiffusion_sequence_track import (
    RFDiffusionMultiTrackStack,)
from deepchem.models.torch_models.rfdiffusion_so3 import (
    IGSO3,
    so3_exp_map,
    so3_log_map,
)

__all__ = [
    'RFDiffusionMultiTrackDenoiser',
    'backbone_coords_from_frames',
    'multitrack_frame_loss',
    'sample_noisy_frames',
    'so3_x0_reverse_step',
    'translation_posterior_step',
]

# Idealized backbone geometry (CA at the origin, C along +x), taken from
# RFdiffusion's own ``chemical.py`` literature backbone coordinates. These
# are the exact values ``build_backbone_frames`` implicitly inverts: they
# let us turn a predicted (rotation, translation) frame back into
# (N, CA, C) coordinates without needing predicted bond lengths/angles.
_IDEAL_LOCAL_N: Tuple[float, float, float] = (-0.5272, 1.3593, 0.0)
_IDEAL_LOCAL_CA: Tuple[float, float, float] = (0.0, 0.0, 0.0)
_IDEAL_LOCAL_C: Tuple[float, float, float] = (1.5233, 0.0, 0.0)


def backbone_coords_from_frames(rotations: torch.Tensor,
                                translations: torch.Tensor) -> torch.Tensor:
    """Reconstruct idealized (N, CA, C) coordinates from rigid frames.

    Applies each rigid transform to the fixed literature backbone geometry
    used by RFdiffusion, giving the inverse of the frame construction done
    by :func:`~deepchem.models.torch_models.rfdiffusion_frames.build_backbone_frames`
    for perfectly idealized geometry.

    Parameters
    ----------
    rotations : torch.Tensor
        Rotation matrices of shape ``(..., 3, 3)``.
    translations : torch.Tensor
        Frame origins (CA positions) of shape ``(..., 3)``.

    Returns
    -------
    torch.Tensor
        Backbone coordinates of shape ``(..., 3, 3)`` ordered ``(N, CA, C)``
        along the second-to-last dimension.

    Examples
    --------
    >>> import torch
    >>> from deepchem.models.torch_models.rfdiffusion_multitrack import (
    ...     backbone_coords_from_frames)
    >>> R = torch.eye(3).expand(2, 5, 3, 3)
    >>> t = torch.zeros(2, 5, 3)
    >>> coords = backbone_coords_from_frames(R, t)
    >>> coords.shape
    torch.Size([2, 5, 3, 3])
    """
    ideal_local = torch.tensor(
        [_IDEAL_LOCAL_N, _IDEAL_LOCAL_CA, _IDEAL_LOCAL_C],
        device=rotations.device,
        dtype=rotations.dtype)
    batch_shape = rotations.shape[:-2]
    ideal_local = ideal_local.expand(*batch_shape, 3, 3)
    return apply_rigid(rotations, translations, ideal_local)


class RFDiffusionMultiTrackDenoiser(nn.Module):
    """Frame-based denoiser built on ``RFDiffusionMultiTrackStack``.

    Embeds noisy backbone coordinates and a diffusion timestep into a
    single-residue representation, runs the RoseTTAFold-style multi-track
    stack (pair track, sequence/structure track, Invariant Point
    Attention), and returns the stack's predicted rigid frames directly as
    the ``x0`` (denoised structure) prediction -- there is no separate
    output head, since ``RFDiffusionMultiTrackStack.forward_tracks``
    already returns updated ``(rotations, translations)``.

    Parameters
    ----------
    embed_dim : int, default 128
        Single-track channel size.
    pair_dim : int, default 64
        Pair-track channel size.
    time_dim : int, default 128
        Dimension of the sinusoidal timestep embedding.
    num_blocks : int, default 2
        Number of stacked multi-track blocks.
    num_heads : int, default 8
        Attention heads for the single track and IPA.
    pair_num_heads : int, default 4
        Attention heads for triangular self-attention.
    max_seq_len : int, default 512
        Maximum supported protein length in residues.
    dropout : float, default 0.0
        Shared dropout probability.

    Examples
    --------
    >>> import torch
    >>> from deepchem.models.torch_models.rfdiffusion_multitrack import (
    ...     RFDiffusionMultiTrackDenoiser)
    >>> denoiser = RFDiffusionMultiTrackDenoiser(
    ...     embed_dim=32, pair_dim=16, num_blocks=1, num_heads=4,
    ...     pair_num_heads=2)
    >>> noisy_coords = torch.randn(2, 6, 9)
    >>> rotations = torch.eye(3).expand(2, 6, 3, 3).contiguous()
    >>> translations = torch.zeros(2, 6, 3)
    >>> t = torch.tensor([3, 7])
    >>> mask = torch.ones(2, 6)
    >>> pred_r, pred_t = denoiser(
    ...     [noisy_coords, rotations, translations, t, mask])
    >>> pred_r.shape, pred_t.shape
    (torch.Size([2, 6, 3, 3]), torch.Size([2, 6, 3]))
    """

    def __init__(self,
                 embed_dim: int = 128,
                 pair_dim: int = 64,
                 time_dim: int = 128,
                 num_blocks: int = 2,
                 num_heads: int = 8,
                 pair_num_heads: int = 4,
                 max_seq_len: int = 512,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.time_embedding = SinusoidalTimestepEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.coord_embed = ResidueEmbedding(9, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, max_seq_len)
        self.stack = RFDiffusionMultiTrackStack(
            embed_dim=embed_dim,
            pair_dim=pair_dim,
            num_blocks=num_blocks,
            num_heads=num_heads,
            pair_num_heads=pair_num_heads,
            dropout=dropout,
        )

    def forward(
            self,
            inputs: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict denoised backbone frames from a noisy structure.

        Parameters
        ----------
        inputs : list of torch.Tensor
            ``[noisy_coords, rotations, translations, t, mask]`` where:

            - ``noisy_coords``: noisy ``(N, CA, C)`` coordinates flattened
              to shape ``(batch, num_residues, 9)``.
            - ``rotations``: noisy rotation matrices, shape ``(batch,
              num_residues, 3, 3)``.
            - ``translations``: noisy CA translations, shape ``(batch,
              num_residues, 3)``.
            - ``t``: integer diffusion timesteps, shape ``(batch,)``.
            - ``mask``: residue validity mask of shape ``(batch,
              num_residues)``; 1 for real residues, 0 for padding. May be
              ``None``.

        Returns
        -------
        pred_rotations : torch.Tensor
            Predicted denoised rotations, shape ``(batch, num_residues, 3,
            3)``.
        pred_translations : torch.Tensor
            Predicted denoised translations, shape ``(batch, num_residues,
            3)``.

        Examples
        --------
        >>> import torch
        >>> from deepchem.models.torch_models.rfdiffusion_multitrack import (
        ...     RFDiffusionMultiTrackDenoiser)
        >>> denoiser = RFDiffusionMultiTrackDenoiser(
        ...     embed_dim=32, pair_dim=16, num_blocks=1, num_heads=4,
        ...     pair_num_heads=2)
        >>> noisy_coords = torch.randn(2, 6, 9)
        >>> rotations = torch.eye(3).expand(2, 6, 3, 3).contiguous()
        >>> translations = torch.zeros(2, 6, 3)
        >>> t = torch.tensor([3, 7])
        >>> mask = torch.ones(2, 6)
        >>> pred_r, pred_t = denoiser.forward(
        ...     [noisy_coords, rotations, translations, t, mask])
        >>> pred_r.shape, pred_t.shape
        (torch.Size([2, 6, 3, 3]), torch.Size([2, 6, 3]))
        """
        noisy_coords, rotations, translations, t, mask = inputs
        t = t.long()
        t_emb = self.time_mlp(self.time_embedding(t))
        single = self.pos_encoding(self.coord_embed(noisy_coords))
        attn_mask = mask.bool() if mask is not None else None
        _, _, pred_rotations, pred_translations = self.stack.forward_tracks(
            single,
            t_emb,
            attention_mask=attn_mask,
            rotations=rotations,
            translations=translations,
        )
        return pred_rotations, pred_translations


def sample_noisy_frames(
    igso3: IGSO3,
    schedule: CosineSchedule,
    rotations0: torch.Tensor,
    translations0: torch.Tensor,
    t: torch.Tensor,
    fixed_mask: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Diffuse ground-truth backbone frames forward to timestep ``t``.

    Rotations are noised by left-composing an IGSO(3) perturbation sampled
    at the timestep's noise level; translations are noised with the
    standard Gaussian forward process from ``schedule.q_sample``. This
    mirrors RFdiffusion's ``Diffuser.diffuse_pose``, which combines an
    ``IGSO3`` rotation diffuser with a Euclidean translation diffuser.

    Parameters
    ----------
    igso3 : IGSO3
        Rotational noise distribution; ``igso3.sigmas`` must have at least
        ``t.max() + 1`` entries.
    schedule : CosineSchedule
        Translation noise schedule.
    rotations0 : torch.Tensor
        Clean rotation matrices of shape ``(batch, num_residues, 3, 3)``.
    translations0 : torch.Tensor
        Clean CA translations of shape ``(batch, num_residues, 3)``.
    t : torch.Tensor
        Integer timesteps of shape ``(batch,)``, one per batch element.
    fixed_mask : torch.Tensor, optional
        Boolean mask of shape ``(batch, num_residues)``. Positions marked
        ``True`` are motif/context residues that are held at their clean
        value instead of being diffused, matching RFdiffusion's
        ``diffusion_mask`` convention.
    generator : torch.Generator, optional
        PyTorch random generator for reproducibility.

    Returns
    -------
    noisy_rotations : torch.Tensor
        Shape ``(batch, num_residues, 3, 3)``.
    noisy_translations : torch.Tensor
        Shape ``(batch, num_residues, 3)``.

    Raises
    ------
    ValueError
        If `rotations0` and `t` do not share the same batch size.

    Examples
    --------
    >>> import torch
    >>> from deepchem.models.torch_models.layers import CosineSchedule
    >>> from deepchem.models.torch_models.rfdiffusion_so3 import (
    ...     IGSO3, log_beta_schedule)
    >>> from deepchem.models.torch_models.rfdiffusion_multitrack import (
    ...     sample_noisy_frames)
    >>> igso3 = IGSO3(log_beta_schedule(10), num_omega=256)
    >>> schedule = CosineSchedule(num_timesteps=10)
    >>> R0 = torch.eye(3).expand(2, 4, 3, 3).contiguous()
    >>> T0 = torch.zeros(2, 4, 3)
    >>> t = torch.tensor([1, 5])
    >>> R_t, T_t = sample_noisy_frames(igso3, schedule, R0, T0, t)
    >>> R_t.shape, T_t.shape
    (torch.Size([2, 4, 3, 3]), torch.Size([2, 4, 3]))
    """
    if rotations0.shape[0] != t.shape[0]:
        raise ValueError('rotations0 and t must share the same batch size.')
    batch, length = rotations0.shape[:2]
    device = rotations0.device
    dtype = rotations0.dtype

    perturbations = torch.stack([
        igso3.sample(sigma_index=int(t[i].item()),
                     shape=(length,),
                     generator=generator).to(device=device, dtype=dtype)
        for i in range(batch)
    ],
                                dim=0)
    noisy_rotations = torch.matmul(perturbations, rotations0)
    noisy_translations, _ = schedule.q_sample(translations0, t)

    if fixed_mask is not None:
        keep = fixed_mask.to(device=device, dtype=torch.bool)
        noisy_rotations = torch.where(keep[..., None, None], rotations0,
                                      noisy_rotations)
        noisy_translations = torch.where(keep[..., None], translations0,
                                         noisy_translations)
    return noisy_rotations, noisy_translations


def translation_posterior_step(
        schedule: CosineSchedule,
        pred_translations: torch.Tensor,
        noisy_translations: torch.Tensor,
        t: int,
        generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """One ``x0``-conditioned DDPM reverse step for translations.

    Implements the standard posterior mean ``q(x_{t-1} | x_t, x0)`` from
    Ho et al. 2020, reusing the ``posterior_mean_coef1/2`` and
    ``posterior_variance`` tensors already computed by ``CosineSchedule``.

    Parameters
    ----------
    schedule : CosineSchedule
        Translation noise schedule.
    pred_translations : torch.Tensor
        Model's predicted clean translations (``x0``), shape ``(batch,
        num_residues, 3)``.
    noisy_translations : torch.Tensor
        Current noisy translations (``x_t``), same shape.
    t : int
        Current (shared) timestep for the whole batch.
    generator : torch.Generator, optional
        PyTorch random generator for reproducibility.

    Returns
    -------
    torch.Tensor
        Translations at timestep ``t - 1``, same shape as ``x_t``.

    Raises
    ------
    ValueError
        If `t` is negative.

    Examples
    --------
    >>> import torch
    >>> from deepchem.models.torch_models.layers import CosineSchedule
    >>> from deepchem.models.torch_models.rfdiffusion_multitrack import (
    ...     translation_posterior_step)
    >>> schedule = CosineSchedule(num_timesteps=10)
    >>> x0 = torch.zeros(2, 4, 3)
    >>> x_t = torch.randn(2, 4, 3)
    >>> x_prev = translation_posterior_step(schedule, x0, x_t, t=5)
    >>> x_prev.shape
    torch.Size([2, 4, 3])
    """
    if t < 0:
        raise ValueError('t must be non-negative.')
    device = pred_translations.device
    coef1 = schedule.posterior_mean_coef1.to(device)[t]
    coef2 = schedule.posterior_mean_coef2.to(device)[t]
    mean = coef1 * pred_translations + coef2 * noisy_translations
    if t == 0:
        return mean
    variance = schedule.posterior_variance.to(device)[t]
    noise = torch.randn(pred_translations.shape,
                        device=device,
                        dtype=pred_translations.dtype,
                        generator=generator)
    return mean + torch.sqrt(variance) * noise


def so3_x0_reverse_step(
        igso3: IGSO3,
        rotations_t: torch.Tensor,
        pred_rotations: torch.Tensor,
        t: int,
        noise_level: float = 1.0,
        generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """One ``x0``-conditioned reverse step for IGSO(3) rotational diffusion.

    Follows RFdiffusion's own ``IGSO3.reverse_sample_vectorized``: the
    score at the current noise level is evaluated on the *relative*
    rotation between the noisy frame and the network's predicted clean
    frame, ``R_0t = R_t @ R_0_pred^T``, and used to take a small step from
    ``R_t`` toward ``R_0_pred`` plus IGSO(3)-scaled noise.

    Parameters
    ----------
    igso3 : IGSO3
        Rotational noise distribution used for training; ``igso3.sigmas``
        must have at least ``t + 1`` entries.
    rotations_t : torch.Tensor
        Current noisy rotations, shape ``(..., 3, 3)``.
    pred_rotations : torch.Tensor
        Model's predicted clean rotations, same shape.
    t : int
        Current (shared) timestep for the whole batch; must be positive
        (there is nothing to reverse at ``t = 0``).
    noise_level : float, default 1.0
        Scale applied to the stochastic term of the step.
    generator : torch.Generator, optional
        PyTorch random generator for reproducibility.

    Returns
    -------
    torch.Tensor
        Rotations at timestep ``t - 1``, same shape as ``rotations_t``.

    Raises
    ------
    ValueError
        If `t` is not positive.

    Examples
    --------
    >>> import torch
    >>> from deepchem.models.torch_models.rfdiffusion_so3 import (
    ...     IGSO3, log_beta_schedule)
    >>> from deepchem.models.torch_models.rfdiffusion_multitrack import (
    ...     so3_x0_reverse_step)
    >>> igso3 = IGSO3(log_beta_schedule(10), num_omega=256)
    >>> R_t = torch.eye(3).expand(2, 4, 3, 3).contiguous()
    >>> R_0 = torch.eye(3).expand(2, 4, 3, 3).contiguous()
    >>> R_prev = so3_x0_reverse_step(igso3, R_t, R_0, t=5)
    >>> R_prev.shape
    torch.Size([2, 4, 3, 3])
    """
    if t <= 0:
        raise ValueError('t must be positive; there is no reverse step at '
                         't = 0.')
    sigma_t = float(igso3.sigmas[t].item())
    sigma_prev = float(igso3.sigmas[t - 1].item())

    relative = torch.matmul(rotations_t, pred_rotations.transpose(-1, -2))
    tangent = so3_log_map(relative)
    omega = tangent.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    direction = tangent / omega
    score = igso3.score(omega.squeeze(-1), t)
    score = score.to(device=tangent.device, dtype=tangent.dtype).unsqueeze(-1)

    drift = (sigma_t**2 - sigma_prev**2) * score * direction
    diffusion_var = max(0.0, sigma_prev**2 * (1.0 - sigma_prev**2 / sigma_t**2))
    noise = torch.randn(tangent.shape,
                        device=tangent.device,
                        dtype=tangent.dtype,
                        generator=generator)
    perturb_tangent = drift + (diffusion_var**0.5) * noise_level * noise
    perturb = so3_exp_map(perturb_tangent)
    return torch.matmul(perturb, rotations_t)


def multitrack_frame_loss(
    outputs: List[torch.Tensor],
    labels: List[torch.Tensor],
    weights: List[torch.Tensor],
    rotation_weight: float = 1.0,
) -> torch.Tensor:
    """Masked frame loss for the multi-track denoiser.

    Combines a translation MSE term with an SO(3) geodesic rotation term,
    matching the ``x0``-prediction parameterization of
    ``RFDiffusionMultiTrackDenoiser``.

    Parameters
    ----------
    outputs : list of torch.Tensor
        ``[pred_rotations, pred_translations]`` from the denoiser.
    labels : list of torch.Tensor
        ``[true_rotations, true_translations]``, same shapes as
        ``outputs``.
    weights : list of torch.Tensor
        ``[mask]``, a residue validity mask of shape ``(batch,
        num_residues)``.
    rotation_weight : float, default 1.0
        Scale applied to the rotation term before summing with the
        translation term.

    Returns
    -------
    torch.Tensor
        Scalar loss.

    Examples
    --------
    >>> import torch
    >>> from deepchem.models.torch_models.rfdiffusion_multitrack import (
    ...     multitrack_frame_loss)
    >>> pred_r = torch.eye(3).expand(2, 4, 3, 3).contiguous()
    >>> pred_t = torch.zeros(2, 4, 3)
    >>> true_r = torch.eye(3).expand(2, 4, 3, 3).contiguous()
    >>> true_t = torch.zeros(2, 4, 3)
    >>> mask = torch.ones(2, 4)
    >>> loss = multitrack_frame_loss([pred_r, pred_t], [true_r, true_t],
    ...                              [mask])
    >>> round(float(loss), 6)
    0.0
    """
    pred_rotations, pred_translations = outputs
    true_rotations, true_translations = labels
    mask = weights[0]
    denom = torch.clamp(mask.sum(), min=1.0)

    translation_loss = (((pred_translations - true_translations)**2).sum(-1) *
                        mask).sum() / denom

    relative = torch.matmul(pred_rotations.transpose(-1, -2), true_rotations)
    angle = so3_log_map(relative).norm(dim=-1)
    rotation_loss = ((angle**2) * mask).sum() / denom

    return translation_loss + rotation_weight * rotation_loss
