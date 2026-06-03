"""SO(3) diffusion utilities for RFDiffusion.

This module implements the rotational component of the RFDiffusion noise
process [Watson2023]_: an IGSO(3) (Isotropic Gaussian on SO(3)) diffusion
with a logarithmic variance schedule and a reverse-time Euler-Maruyama
integrator on the so(3) tangent space [Leach2022]_, [Yim2023]_.

Conventions
-----------
* Rotations are represented as ``(..., 3, 3)`` proper rotation matrices
  R ∈ SO(3) with det(R) = +1.
* A rotation is parameterised on the tangent space so(3) ≅ ℝ³ by a
  rotation vector τ = ω · u where ω ∈ [0, π] is the rotation angle
  (geodesic distance from the identity) and u ∈ S² is the rotation axis.
* The matrix exponential of a tangent vector uses Rodrigues' formula
  R = I + sin(ω) [u]_× + (1 − cos(ω)) [u]_×² where [u]_× is the
  skew-symmetric matrix associated with u.

Mathematical reference
----------------------
The IGSO(3) density on the rotation angle ω is

.. math::

    f(\\omega; \\sigma) = \\sum_{l=0}^{\\infty}
        (2l+1)\\,
        \\exp\\!\\left(-\\tfrac{l(l+1)}{2}\\sigma^{2}\\right)\\,
        \\frac{\\sin\\!\\big((l+\\tfrac{1}{2})\\omega\\big)}
             {\\sin(\\omega/2)}

and the marginal density on ω ∈ [0, π] is

.. math::

    p(\\omega; \\sigma) = \\frac{1 - \\cos \\omega}{\\pi}\\, f(\\omega; \\sigma).

The score on the rotation angle is

.. math::

    s(\\omega; \\sigma) = \\partial_{\\omega}\\log f(\\omega; \\sigma)

and the score on so(3) — used as the drift term in the reverse SDE —
points along the geodesic from the identity to R with magnitude
``s(ω; σ)``.

Developer-facing summary
------------------------
This file is the rotation-noise backend for RFDiffusion. Use
``so3_exp_map`` and ``so3_log_map`` to move between tangent vectors and
rotation matrices, build an ``IGSO3`` table once for cached score /
sampling queries, and call ``so3_reverse_step`` during sampling to move
one denoising step from ``R_t`` to ``R_{t-1}``.

References
----------
.. [Watson2023] Watson et al. "De novo design of protein structure and
   function with RFdiffusion." Nature 620 (2023) 1089-1100.
.. [Leach2022] Leach et al. "Denoising Diffusion Probabilistic Models on
   SO(3) for Rotational Alignment." ICLR Workshop on Geometric and
   Topological Representation Learning (2022).
.. [Yim2023] Yim et al. "SE(3) Diffusion Model with Application to Protein
   Backbone Generation." ICML (2023).
"""

import math
from typing import Optional, Tuple

try:
    import torch
except ModuleNotFoundError:
    raise ImportError('rfdiffusion_so3 requires PyTorch to be installed.')

__all__ = [
    'IGSO3',
    'log_beta_schedule',
    'so3_log_map',
    'so3_exp_map',
    'so3_reverse_step',
]

# Threshold below which series expansions are used instead of the closed
# form to preserve numerical stability.
_SMALL_OMEGA: float = 1e-4
_SMALL_SIGMA: float = 5e-2


def _safe_sin_div_x(x: torch.Tensor) -> torch.Tensor:
    """Return ``sin(x) / x`` with a stable Taylor series near zero.

    Computes the second-order Taylor expansion ``1 - x²/6`` when
    |x| < ``_SMALL_OMEGA`` and the closed-form expression elsewhere.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of arbitrary shape.

    Returns
    -------
    torch.Tensor
        Tensor of the same shape as ``x`` containing ``sin(x) / x``.
    """
    small = x.abs() < _SMALL_OMEGA
    safe_x = torch.where(small, torch.ones_like(x), x)
    return torch.where(small, 1.0 - x * x / 6.0, torch.sin(safe_x) / safe_x)


def _safe_one_minus_cos_div_x_sq(x: torch.Tensor) -> torch.Tensor:
    """Return ``(1 − cos x) / x²`` with a Taylor series near zero.

    Computes ``1/2 − x²/24`` when |x| < ``_SMALL_OMEGA`` and the closed
    form ``(1 − cos x) / x²`` elsewhere.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of arbitrary shape.

    Returns
    -------
    torch.Tensor
        Tensor of the same shape as ``x``.
    """
    small = x.abs() < _SMALL_OMEGA
    safe_x = torch.where(small, torch.ones_like(x), x)
    closed = (1.0 - torch.cos(safe_x)) / (safe_x * safe_x)
    series = 0.5 - x * x / 24.0
    return torch.where(small, series, closed)


def so3_exp_map(tangent: torch.Tensor) -> torch.Tensor:
    """Map a tangent vector in so(3) ≅ ℝ³ to a rotation matrix.

    Implements Rodrigues' rotation formula

    .. math::

        R = I + \\frac{\\sin\\omega}{\\omega}\\,[\\tau]_{\\times}
            + \\frac{1 - \\cos\\omega}{\\omega^{2}}\\,[\\tau]_{\\times}^{2}

    where ω = ‖τ‖ and [τ]_× is the skew-symmetric matrix of τ. Small-ω
    Taylor expansions are used to maintain numerical stability.

    Parameters
    ----------
    tangent : torch.Tensor
        Tangent vector(s) of shape ``(..., 3)`` representing rotation
        vectors τ = ω · u.

    Returns
    -------
    torch.Tensor
        Rotation matrices of shape ``(..., 3, 3)``.
    """
    omega = tangent.norm(dim=-1, keepdim=True).clamp(min=0.0)  # (..., 1)
    # Build the skew matrix [τ]_×.
    zeros = torch.zeros_like(tangent[..., 0])
    tx, ty, tz = tangent[..., 0], tangent[..., 1], tangent[..., 2]
    skew = torch.stack([
        torch.stack([zeros, -tz, ty], dim=-1),
        torch.stack([tz, zeros, -tx], dim=-1),
        torch.stack([-ty, tx, zeros], dim=-1),
    ], dim=-2)  # (..., 3, 3)
    eye = torch.eye(3, dtype=tangent.dtype, device=tangent.device)
    sin_coeff = _safe_sin_div_x(omega).unsqueeze(-1)         # (..., 1, 1)
    cos_coeff = _safe_one_minus_cos_div_x_sq(omega).unsqueeze(-1)
    skew_sq = torch.matmul(skew, skew)
    return eye + sin_coeff * skew + cos_coeff * skew_sq


def so3_log_map(rotations: torch.Tensor) -> torch.Tensor:
    """Map rotation matrices to their tangent vectors in so(3).

    Computes the principal-branch logarithm: ω = arccos((tr R − 1) / 2)
    and τ = (ω / (2 sin ω)) · vee(R − Rᵀ), with stable expansions when
    ω is close to 0 or π.

    Parameters
    ----------
    rotations : torch.Tensor
        Rotation matrices of shape ``(..., 3, 3)``.

    Returns
    -------
    torch.Tensor
        Tangent vectors of shape ``(..., 3)``.
    """
    trace = rotations.diagonal(offset=0, dim1=-2, dim2=-1).sum(-1)
    cos_omega = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    omega = torch.acos(cos_omega)  # (..., ) in [0, π]
    # vee(R - Rᵀ) extracts the skew components.
    skew = rotations - rotations.transpose(-1, -2)
    vec = torch.stack([
        skew[..., 2, 1],
        skew[..., 0, 2],
        skew[..., 1, 0],
    ], dim=-1)  # (..., 3)
    # τ = ω / (2 sin ω) · vec.  Use Taylor expansion when sin ω small.
    sin_omega = torch.sin(omega)
    small = omega < _SMALL_OMEGA
    near_pi = (math.pi - omega) < 1e-3
    coeff_closed = omega / (2.0 * torch.where(
        small | near_pi, torch.ones_like(sin_omega), sin_omega))
    coeff = torch.where(small, torch.full_like(omega, 0.5), coeff_closed)
    tangent = coeff.unsqueeze(-1) * vec

    # The standard formula is ill-conditioned near π because sin(ω) is
    # close to zero. Recover the axis from the diagonal and use the skew
    # part only for signs, matching scipy Rotation.as_rotvec() on the
    # principal branch for rotations just below π.
    diag = rotations.diagonal(offset=0, dim1=-2, dim2=-1)
    axis_abs = torch.sqrt(((diag + 1.0) * 0.5).clamp(min=0.0))
    signs = torch.sign(vec)
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    axis = axis_abs * signs
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    tangent_near_pi = omega.unsqueeze(-1) * axis
    return torch.where(near_pi.unsqueeze(-1), tangent_near_pi, tangent)


def log_beta_schedule(num_steps: int,
                      beta_min: float = 0.1,
                      beta_max: float = 1.5) -> torch.Tensor:
    """Logarithmic variance schedule for IGSO(3) diffusion.

    Returns σ_t = exp(linear(log β_min, log β_max)) for t = 1 .. T. This
    is the schedule used by [Watson2023]_ and [Yim2023]_ for rotational
    diffusion. The schedule is **monotonically increasing** in t.

    Parameters
    ----------
    num_steps : int
        Number of diffusion steps T (must be ≥ 2).
    beta_min : float, default 0.1
        Smallest σ value (at t = 1).
    beta_max : float, default 1.5
        Largest σ value (at t = T).

    Returns
    -------
    torch.Tensor
        Schedule tensor of shape ``(num_steps,)`` containing σ_t values.
    """
    if num_steps < 2:
        raise ValueError('num_steps must be >= 2.')
    if beta_min <= 0 or beta_max <= 0 or beta_max <= beta_min:
        raise ValueError(
            'Require 0 < beta_min < beta_max; got '
            f'beta_min={beta_min}, beta_max={beta_max}.')
    log_min = math.log(beta_min)
    log_max = math.log(beta_max)
    return torch.exp(torch.linspace(log_min, log_max, num_steps))


class IGSO3:
    """Isotropic Gaussian distribution on SO(3).

    The IGSO(3) density on the rotation angle ω ∈ [0, π] is the heat
    kernel of the Laplace-Beltrami operator on SO(3) evaluated at
    rotations whose geodesic distance from the identity is ω
    [Nikolayev1997]_, [Leach2022]_.

    The class caches a discretised PDF / CDF over a ``num_omega`` grid
    for the supplied ``sigma`` values so that ``sample`` and ``score``
    queries are fast.

    Parameters
    ----------
    sigmas : torch.Tensor
        1-D tensor of strictly positive σ values (one per discrete
        diffusion step).
    num_omega : int, default 1024
        Number of grid points used to discretise ω ∈ (0, π).
    lmax : int, default 1999
        Maximum degree retained in the infinite-series PDF. This gives
        2000 terms, matching upstream RFdiffusion's default truncation
        level.
    cache : bool, default True
        If ``True`` precompute and cache the discretised PDF / CDF for
        each σ in ``sigmas``.

    Examples
    --------
    >>> sigmas = torch.tensor([0.5])
    >>> dist = IGSO3(sigmas, num_omega=256)
    >>> samples = dist.sample(sigma_index=0, shape=(8,))
    >>> samples.shape
    torch.Size([8, 3, 3])

    References
    ----------
    .. [Nikolayev1997] Nikolayev, D. I. & Savyolov, T. I. "Normal
       distribution on the rotation group SO(3)." Textures and
       Microstructures 29 (1997) 201-233.
    """

    def __init__(self,
                 sigmas: torch.Tensor,
                 num_omega: int = 1024,
                 lmax: int = 1999,
                 cache: bool = True) -> None:
        if sigmas.ndim != 1:
            raise ValueError('sigmas must be a 1-D tensor.')
        if (sigmas <= 0).any():
            raise ValueError('sigmas must be strictly positive.')
        if num_omega < 64:
            raise ValueError('num_omega must be at least 64.')
        if lmax < 1:
            raise ValueError('lmax must be at least 1.')
        self.sigmas = sigmas.to(dtype=torch.float64)
        self.num_omega = int(num_omega)
        self.lmax = int(lmax)
        # Upstream RFdiffusion discretises ω as linspace(0, π, N+1)[1:]:
        # zero is skipped, π is included.
        self.omegas = torch.linspace(
            0.0, math.pi, self.num_omega + 1, dtype=torch.float64)[1:]
        self.domega = float(self.omegas[1] - self.omegas[0])
        self._pdf: Optional[torch.Tensor] = None
        self._cdf: Optional[torch.Tensor] = None
        self._score: Optional[torch.Tensor] = None
        if cache:
            self._compute_tables()

    # ------------------------------------------------------------------
    # PDF / CDF / score on the rotation angle ω
    # ------------------------------------------------------------------
    @staticmethod
    def _f_omega(omega: torch.Tensor, sigma: torch.Tensor,
                 lmax: int) -> torch.Tensor:
        """Evaluate the heat-kernel series f(ω; σ).

        Parameters
        ----------
        omega : torch.Tensor
            Tensor of shape ``(N,)`` containing rotation angles ω.
        sigma : torch.Tensor
            Tensor of shape ``(M,)`` containing σ values.
        lmax : int
            Truncation degree of the series.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(M, N)`` with ``f(ω_n; σ_m)``.
        """
        l_values = torch.arange(lmax + 1, device=omega.device)[None]
        omega_col = omega[:, None]
        rows = []
        for sigma_i in sigma:
            sigma_sq = float(sigma_i.detach().cpu().item() ** 2)
            terms = ((2 * l_values + 1)
                     * torch.exp(-l_values * (l_values + 1) * sigma_sq / 2)
                     * torch.sin(omega_col * (l_values + 0.5))
                     / torch.sin(omega_col / 2))
            rows.append(terms.sum(dim=-1))
        return torch.stack(rows, dim=0)

    @staticmethod
    def _score_omega(omega: torch.Tensor, sigma: torch.Tensor,
                     lmax: int) -> torch.Tensor:
        """Evaluate ∂ω log f(ω; σ) using upstream-style autograd."""
        with torch.enable_grad():
            omega_var = omega.detach().clone().requires_grad_(True)
            f = IGSO3._f_omega(omega_var, sigma.detach(), lmax)
            scores = []
            for row in f:
                log_f = torch.log(row)
                grad = torch.autograd.grad(log_f.sum(), omega_var,
                                           retain_graph=True)[0]
                scores.append(grad.detach())
        return torch.stack(scores, dim=0)

    def _compute_tables(self) -> None:
        """Populate the PDF / CDF / score caches."""
        f = self._f_omega(self.omegas, self.sigmas, self.lmax)  # (M, N)
        # Marginal density on ω: p(ω) = (1 − cos ω)/π · f(ω).
        prefactor = (1.0 - torch.cos(self.omegas)) / math.pi  # (N,)
        pdf = prefactor[None, :] * f
        cdf = torch.cumsum(pdf, dim=-1) * self.domega
        # Score s(ω; σ) = ∂ω log f(ω; σ), matching the upstream
        # autograd derivative of the truncated heat-kernel series.
        score = self._score_omega(self.omegas, self.sigmas, self.lmax)
        self._pdf = pdf
        self._cdf = cdf
        self._score = score

    def pdf(self, omega: torch.Tensor,
            sigma_index: int) -> torch.Tensor:
        """Interpolate the cached PDF at the supplied angles.

        Parameters
        ----------
        omega : torch.Tensor
            Tensor of angles ω ∈ (0, π) of arbitrary shape.
        sigma_index : int
            Index of σ in ``self.sigmas`` whose PDF to evaluate.

        Returns
        -------
        torch.Tensor
            PDF evaluated at the supplied angles.
        """
        if self._pdf is None:
            self._compute_tables()
        return self._interp(omega, self._pdf[sigma_index])

    def cdf(self, omega: torch.Tensor,
            sigma_index: int) -> torch.Tensor:
        """Interpolate the cached CDF at the supplied angles."""
        if self._cdf is None:
            self._compute_tables()
        return self._interp(omega, self._cdf[sigma_index])

    def score(self, omega: torch.Tensor,
              sigma_index: int) -> torch.Tensor:
        """Interpolate the cached score s(ω; σ) at the supplied angles.

        For small σ the IGSO(3) collapses to a wrapped Gaussian on
        so(3); the score is then approximately −ω / σ². The cached
        table already encodes this limit, so no special branch is
        required at query time.
        """
        if self._score is None:
            self._compute_tables()
        return self._interp(omega, self._score[sigma_index])

    def _interp(self, omega: torch.Tensor,
                table: torch.Tensor) -> torch.Tensor:
        """Linear interpolation of a 1-D cached table on the ω grid."""
        omega = omega.to(dtype=torch.float64)
        x = (omega - self.omegas[0]) / self.domega
        x = x.clamp(0.0, float(self.num_omega - 1))
        lo = x.floor().long()
        hi = (lo + 1).clamp(max=self.num_omega - 1)
        frac = (x - lo.to(x.dtype))
        return (1.0 - frac) * table[lo] + frac * table[hi]

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def sample_angle(self, sigma_index: int,
                     shape: Tuple[int, ...],
                     generator: Optional[torch.Generator] = None
                     ) -> torch.Tensor:
        """Sample rotation angles ω from IGSO(3) by inverse CDF."""
        if self._cdf is None:
            self._compute_tables()
        cdf = self._cdf[sigma_index]
        u = torch.rand(shape, dtype=torch.float64, generator=generator)
        idx = torch.searchsorted(cdf, u.flatten()).clamp(
            max=self.num_omega - 1)
        return self.omegas[idx].reshape(shape)

    def sample(self, sigma_index: int,
               shape: Tuple[int, ...],
               generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Sample rotation matrices from IGSO(3, σ).

        Sampling proceeds by:

        1. drawing ω ~ p(ω; σ) via inverse CDF;
        2. drawing a uniform axis u ∈ S² (rejection-free);
        3. mapping τ = ω · u through Rodrigues' formula.

        Parameters
        ----------
        sigma_index : int
            Index of σ in ``self.sigmas``.
        shape : tuple of int
            Output batch shape; the returned tensor has shape
            ``shape + (3, 3)``.
        generator : torch.Generator, optional
            PyTorch random generator for reproducibility.

        Returns
        -------
        torch.Tensor
            Sampled rotation matrices of shape ``shape + (3, 3)`` in
            ``torch.float32``.
        """
        omega = self.sample_angle(sigma_index, shape, generator=generator)
        # Uniform axis on S² via standard normal normalisation.
        axis = torch.randn(shape + (3,), dtype=torch.float64,
                           generator=generator)
        axis = axis / axis.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        tangent = (omega.unsqueeze(-1) * axis).to(dtype=torch.float32)
        return so3_exp_map(tangent)

    # ------------------------------------------------------------------
    # Discrete normalisation check
    # ------------------------------------------------------------------
    def normalisation_error(self, sigma_index: int) -> float:
        """Return |∫p(ω;σ) dω − 1| on the cached grid.

        This mirrors the upstream RFdiffusion Riemann-sum check; the
        table is not renormalised after truncation.
        """
        if self._pdf is None:
            self._compute_tables()
        total = float(self._pdf[sigma_index].sum() * self.domega)
        return abs(total - 1.0)


def so3_reverse_step(rotations: torch.Tensor,
                     score: torch.Tensor,
                     sigma_t: float,
                     sigma_t_minus_1: float,
                     noise: Optional[torch.Tensor] = None
                     ) -> torch.Tensor:
    """Single reverse-time Euler-Maruyama step on SO(3).

    Implements the discretisation [Yim2023]_

    .. math::

        \\tau_{t-1} = (\\sigma_{t-1}^{2}/\\sigma_{t}^{2})\\,\\tau_{t}
                    + (\\sigma_{t}^{2} - \\sigma_{t-1}^{2})\\,
                      s(\\omega_t; \\sigma_t)\\,
                      \\frac{\\tau_t}{\\omega_t}
                    + \\sqrt{\\sigma_{t-1}^{2}\\,
                            (1 - \\sigma_{t-1}^{2}/\\sigma_{t}^{2})}\\,
                      \\xi

    where ξ ∼ N(0, I₃), τ_t = log(R_t) and the resulting τ_{t−1} is
    re-exponentiated via Rodrigues' formula to give R_{t−1}.

    Parameters
    ----------
    rotations : torch.Tensor
        Current rotation matrices of shape ``(..., 3, 3)``.
    score : torch.Tensor
        Score s(ω_t; σ_t) evaluated at the current rotation angles,
        broadcastable to the leading dimensions of ``rotations``.
    sigma_t : float
        Noise level at the current step.
    sigma_t_minus_1 : float
        Noise level at the previous step (must be < σ_t).
    noise : torch.Tensor, optional
        Standard normal noise of shape ``rotations.shape[:-2] + (3,)``.
        If ``None`` a fresh sample is drawn.

    Returns
    -------
    torch.Tensor
        Updated rotation matrices of shape ``(..., 3, 3)``.
    """
    if sigma_t <= 0 or sigma_t_minus_1 <= 0:
        raise ValueError('sigma values must be strictly positive.')
    if sigma_t_minus_1 >= sigma_t:
        raise ValueError(
            'Reverse step requires sigma_t_minus_1 < sigma_t; got '
            f'{sigma_t_minus_1} >= {sigma_t}.')
    tau_t = so3_log_map(rotations)
    omega = tau_t.norm(dim=-1, keepdim=True).clamp(min=_SMALL_OMEGA)
    direction = tau_t / omega
    score = score.unsqueeze(-1) if score.dim() == tau_t.dim() - 1 else score
    ratio = (sigma_t_minus_1 / sigma_t) ** 2
    drift = ratio * tau_t + (sigma_t ** 2 - sigma_t_minus_1 ** 2) * (
        score * direction)
    diffusion_std = math.sqrt(
        max(0.0, sigma_t_minus_1 ** 2 * (1.0 - ratio)))
    if noise is None:
        noise = torch.randn_like(tau_t)
    tau_prev = drift + diffusion_std * noise
    return so3_exp_map(tau_prev)
