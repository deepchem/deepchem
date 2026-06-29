"""Input embedding layers for the RFDiffusion protein backbone model.

This module adds three building-block layers that turn raw backbone
coordinates and diffusion timesteps into feature vectors before the
denoiser transformer sees them:

- ``SinusoidalTimestepEmbedding`` — turns an integer timestep into a
  sinusoidal vector so the network can tell how noisy the current input is.
- ``ResidueEmbedding`` — projects the 9 raw backbone coordinates (N, CA, C
  each with x, y, z) into a higher-dimensional space.
- ``PositionalEncoding`` — adds sinusoidal position information so the
  network knows where each residue sits in the chain.
- ``CosineSchedule`` — implements the improved cosine variance schedule
  for the forward (noise-adding) and reverse (denoising) diffusion steps.

References
----------
.. [1] Watson, J. L., et al. "De novo design of protein structure and function
   with RFdiffusion." Nature 620.7976 (2023): 1089-1100.
.. [2] Ho, J., Jain, A., & Abbeel, P. "Denoising diffusion probabilistic
   models." NeurIPS 2020.
.. [3] Nichol, A. Q., & Dhariwal, P. "Improved denoising diffusion
   probabilistic models." ICML 2021.
.. [4] Vaswani, A., et al. "Attention is all you need." NeurIPS 2017.

Notes
-----
This module requires PyTorch to be installed.
"""

import math
from typing import Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError:
    raise ImportError('These classes require PyTorch to be installed.')


class SinusoidalTimestepEmbedding(nn.Module):
    """Sinusoidal embedding for diffusion timesteps.

    Maps integer timesteps to continuous vector representations using
    sinusoidal functions, following the positional encoding scheme from
    the Transformer architecture [1]_.

    Parameters
    ----------
    dim : int
        Dimensionality of the output embedding.

    References
    ----------
    .. [1] Vaswani, A., et al. "Attention is all you need." NeurIPS 2017.

    Examples
    --------
    >>> import torch
    >>> from deepchem.models.torch_models.rfdiffusion import SinusoidalTimestepEmbedding
    >>> emb = SinusoidalTimestepEmbedding(64)
    >>> t = torch.tensor([0, 100, 500, 999])
    >>> output = emb(t)
    >>> output.shape
    torch.Size([4, 64])
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Compute timestep embeddings.

        Parameters
        ----------
        t : torch.Tensor
            Integer timesteps of shape ``(batch_size,)``.

        Returns
        -------
        torch.Tensor
            Embeddings of shape ``(batch_size, dim)``.
        """
        device = t.device
        half_dim = self.dim // 2
        log_scale: float = math.log(10000) / (half_dim - 1)
        freq = torch.exp(torch.arange(half_dim, device=device) * -log_scale)
        emb = t.float().unsqueeze(1) * freq.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class ResidueEmbedding(nn.Module):
    """Embed per-residue backbone coordinates into feature vectors.

    Each residue is represented by 9 values (N, CA, C atoms x 3 coordinates).
    This module projects them into a higher-dimensional feature space.

    Parameters
    ----------
    coord_dim : int, default 9
        Input coordinate dimension (3 atoms * 3 xyz = 9).
    embed_dim : int, default 256
        Output embedding dimension.

    Examples
    --------
    >>> import torch
    >>> from deepchem.models.torch_models.rfdiffusion import ResidueEmbedding
    >>> emb = ResidueEmbedding(9, 128)
    >>> x = torch.randn(2, 50, 9)
    >>> output = emb(x)
    >>> output.shape
    torch.Size([2, 50, 128])
    """

    def __init__(self, coord_dim: int = 9, embed_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(coord_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project coordinates to embeddings.

        Parameters
        ----------
        x : torch.Tensor
            Backbone coordinates of shape ``(batch, num_residues, coord_dim)``.

        Returns
        -------
        torch.Tensor
            Embeddings of shape ``(batch, num_residues, embed_dim)``.
        """
        return self.net(x)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for residue positions along the chain.

    Injects information about relative position of each residue in the
    protein sequence, following standard Transformer positional encoding.

    Parameters
    ----------
    embed_dim : int
        Feature dimension.
    max_len : int, default 512
        Maximum supported sequence length.

    Examples
    --------
    >>> import torch
    >>> from deepchem.models.torch_models.rfdiffusion import PositionalEncoding
    >>> pe = PositionalEncoding(128, max_len=256)
    >>> x = torch.randn(2, 50, 128)
    >>> output = pe(x)
    >>> output.shape
    torch.Size([2, 50, 128])
    """

    def __init__(self, embed_dim: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() *
            (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input features.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(batch, seq_len, embed_dim)``.

        Returns
        -------
        torch.Tensor
            Input with positional encoding added, same shape.
        """
        return x + self.pe[:, :x.size(1), :]


class CosineSchedule(nn.Module):
    """Cosine variance schedule for the diffusion process.

    Implements the cosine schedule from Improved DDPM [1]_, which tends to
    give better sample quality than a linear schedule because the noise is
    added more smoothly at the start and end of the diffusion trajectory.

    All schedule tensors are stored as ``register_buffer`` so they move
    with the module when you call ``.to(device)`` and are included in
    ``state_dict()`` for checkpointing.

    Parameters
    ----------
    num_timesteps : int, default 1000
        Total number of diffusion timesteps.
    s : float, default 0.008
        Small offset that prevents the schedule from reaching exactly zero
        at t=0.

    References
    ----------
    .. [1] Nichol, A. Q., & Dhariwal, P. "Improved denoising diffusion
       probabilistic models." ICML 2021.

    Examples
    --------
    >>> from deepchem.models.torch_models.rfdiffusion import CosineSchedule
    >>> schedule = CosineSchedule(num_timesteps=100)
    >>> import torch
    >>> x0 = torch.randn(2, 10, 9)
    >>> t = torch.tensor([0, 50])
    >>> noisy_x, noise = schedule.q_sample(x0, t)
    >>> noisy_x.shape
    torch.Size([2, 10, 9])
    """

    def __init__(self, num_timesteps: int = 1000, s: float = 0.008) -> None:
        super().__init__()
        self.num_timesteps = num_timesteps

        steps = num_timesteps + 1
        t = torch.linspace(0, num_timesteps, steps) / num_timesteps
        alpha_cumprod = torch.cos((t + s) / (1 + s) * math.pi / 2)**2
        alpha_cumprod = alpha_cumprod / alpha_cumprod[0]
        betas = 1 - (alpha_cumprod[1:] / alpha_cumprod[:-1])
        betas = torch.clamp(betas, 0, 0.999)

        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        alpha_cumprod_prev = F.pad(alpha_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_cumprod', alpha_cumprod)
        self.register_buffer('sqrt_alpha_cumprod', torch.sqrt(alpha_cumprod))
        self.register_buffer('sqrt_one_minus_alpha_cumprod',
                             torch.sqrt(1.0 - alpha_cumprod))
        self.register_buffer('sqrt_recip_alpha_cumprod',
                             torch.sqrt(1.0 / alpha_cumprod))
        self.register_buffer('sqrt_recipm1_alpha_cumprod',
                             torch.sqrt(1.0 / alpha_cumprod - 1))

        posterior_variance = (betas * (1.0 - alpha_cumprod_prev) /
                              (1.0 - alpha_cumprod))
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer(
            'posterior_log_variance',
            torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer(
            'posterior_mean_coef1',
            betas * torch.sqrt(alpha_cumprod_prev) / (1.0 - alpha_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1.0 - alpha_cumprod_prev) * torch.sqrt(alphas) /
                             (1.0 - alpha_cumprod))

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward diffusion: add noise to clean data at timestep t.

        Parameters
        ----------
        x0 : torch.Tensor
            Clean data of shape ``(batch, ...)``.
        t : torch.Tensor
            Integer timesteps of shape ``(batch,)``.
        noise : torch.Tensor, optional
            Pre-sampled noise tensor. Drawn from N(0, I) if not provided.

        Returns
        -------
        noisy_x : torch.Tensor
            Noisy data at timestep t, same shape as x0.
        noise : torch.Tensor
            The noise that was added, same shape as x0.
        """
        if noise is None:
            noise = torch.randn_like(x0)

        t_idx = t.to(self.sqrt_alpha_cumprod.device)
        sqrt_alpha = self.sqrt_alpha_cumprod[t_idx].to(x0.device)
        sqrt_one_minus = self.sqrt_one_minus_alpha_cumprod[t_idx].to(x0.device)

        while sqrt_alpha.dim() < x0.dim():
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus = sqrt_one_minus.unsqueeze(-1)

        noisy_x = sqrt_alpha * x0 + sqrt_one_minus * noise
        return noisy_x, noise

    @torch.no_grad()
    def p_sample(self, model: nn.Module, x_t: torch.Tensor,
                 t: torch.Tensor) -> torch.Tensor:
        """Reverse diffusion: denoise x_t by one step using the model.

        Parameters
        ----------
        model : nn.Module
            The denoiser network. Called as ``model([x_t, t])``.
        x_t : torch.Tensor
            Noisy data of shape ``(batch, num_residues, coord_dim)``.
        t : torch.Tensor
            Current timestep of shape ``(batch,)``.

        Returns
        -------
        torch.Tensor
            Denoised data at timestep t-1, same shape as x_t.
        """
        device = x_t.device
        noise_pred = model([x_t, t])

        t_idx = t.to(self.betas.device)

        sqrt_recip = self.sqrt_recip_alpha_cumprod[t_idx].to(device)
        sqrt_recipm1 = self.sqrt_recipm1_alpha_cumprod[t_idx].to(device)
        while sqrt_recip.dim() < x_t.dim():
            sqrt_recip = sqrt_recip.unsqueeze(-1)
            sqrt_recipm1 = sqrt_recipm1.unsqueeze(-1)
        x0_pred = sqrt_recip * x_t - sqrt_recipm1 * noise_pred

        coef1 = self.posterior_mean_coef1[t_idx].to(device)
        coef2 = self.posterior_mean_coef2[t_idx].to(device)
        while coef1.dim() < x_t.dim():
            coef1 = coef1.unsqueeze(-1)
            coef2 = coef2.unsqueeze(-1)
        posterior_mean = coef1 * x0_pred + coef2 * x_t

        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().to(device)
        while nonzero_mask.dim() < x_t.dim():
            nonzero_mask = nonzero_mask.unsqueeze(-1)

        var = self.posterior_variance[t_idx].to(device)
        while var.dim() < x_t.dim():
            var = var.unsqueeze(-1)

        return posterior_mean + nonzero_mask * torch.sqrt(var) * noise

    @torch.no_grad()
    def sample(self, model: nn.Module, shape: Tuple[int, ...],
               device: torch.device) -> torch.Tensor:
        """Generate samples by running the full reverse diffusion loop.

        Parameters
        ----------
        model : nn.Module
            The denoiser network.
        shape : tuple of int
            Shape of samples to produce: ``(batch, num_residues, coord_dim)``.
        device : torch.device
            Device to generate on.

        Returns
        -------
        torch.Tensor
            Generated samples of the given shape.
        """
        model.eval()
        x = torch.randn(shape, device=device)

        for t_val in reversed(range(self.num_timesteps)):
            t = torch.full((shape[0],), t_val, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t)

        return x
