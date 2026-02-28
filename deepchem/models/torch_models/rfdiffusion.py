"""Diffusion layers for protein backbone generation.

This module provides the neural network layers and noise scheduling used by
the RFDiffusion model for protein backbone generation. It implements a
Transformer-based denoiser with timestep conditioning and a cosine variance
schedule for the diffusion process.

References
----------
.. [1] Watson, J. L., et al. "De novo design of protein structure and function
   with RFdiffusion." Nature 620.7976 (2023): 1089-1100.
.. [2] Ho, J., Jain, A., & Abbeel, P. "Denoising diffusion probabilistic
   models." NeurIPS 2020.
.. [3] Nichol, A. Q., & Dhariwal, P. "Improved denoising diffusion
   probabilistic models." ICML 2021.

Notes
-----
This module requires PyTorch to be installed.
"""

import math
import logging
from typing import List, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError:
    raise ImportError('These classes require PyTorch to be installed.')

logger = logging.getLogger(__name__)


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
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
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


class DiffusionTransformerBlock(nn.Module):
    """Transformer block with diffusion timestep conditioning.

    Standard self-attention block augmented with adaptive layer norm
    conditioning on the diffusion timestep. The timestep embedding is
    used to compute scale and shift parameters that modulate the
    intermediate representations.

    Parameters
    ----------
    embed_dim : int
        Hidden dimension size.
    num_heads : int, default 8
        Number of attention heads.
    mlp_ratio : float, default 4.0
        Expansion ratio for the feedforward network.
    dropout : float, default 0.1
        Dropout probability.

    Examples
    --------
    >>> import torch
    >>> block = DiffusionTransformerBlock(128, num_heads=4)
    >>> x = torch.randn(2, 50, 128)
    >>> t_emb = torch.randn(2, 128)
    >>> output = block(x, t_emb)
    >>> output.shape
    torch.Size([2, 50, 128])
    """

    def __init__(self,
                 embed_dim: int,
                 num_heads: int = 8,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim,
                                          num_heads,
                                          dropout=dropout,
                                          batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout),
        )
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim * 2),
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """Forward pass with timestep conditioning.

        Parameters
        ----------
        x : torch.Tensor
            Input features of shape ``(batch, seq_len, embed_dim)``.
        t_emb : torch.Tensor
            Timestep embedding of shape ``(batch, embed_dim)``.

        Returns
        -------
        torch.Tensor
            Output features, same shape as input.
        """
        # Adaptive layer norm with time conditioning
        time_cond = self.time_mlp(t_emb)
        scale, shift = time_cond.chunk(2, dim=-1)
        scale = scale.unsqueeze(1)
        shift = shift.unsqueeze(1)

        h = self.norm1(x)
        h = h * (1 + scale) + shift
        attn_out, _ = self.attn(h, h, h)
        x = x + attn_out

        x = x + self.mlp(self.norm2(x))
        return x


class BackboneDiffusion(nn.Module):
    """Neural network for protein backbone denoising diffusion.

    This is the core denoiser network that predicts the noise added to
    protein backbone coordinates during the forward diffusion process.
    It processes flattened backbone coordinates (N, CA, C per residue)
    through a Transformer with timestep conditioning.

    The architecture uses:

    - Sinusoidal timestep embeddings
    - Per-residue coordinate embeddings
    - Positional encoding for sequence position
    - Transformer blocks with adaptive time conditioning
    - Zero-initialized output projection (standard for diffusion [1]_)

    Parameters
    ----------
    coord_dim : int, default 9
        Input coordinate dimension. Default is 9 for 3 backbone atoms
        (N, CA, C) times 3 spatial dimensions (x, y, z).
    embed_dim : int, default 256
        Hidden dimension for the Transformer.
    time_dim : int, default 128
        Dimension of the timestep embedding before projection.
    num_layers : int, default 8
        Number of Transformer blocks.
    num_heads : int, default 8
        Number of attention heads per block.
    max_seq_len : int, default 512
        Maximum supported protein length (in residues).
    dropout : float, default 0.1
        Dropout probability.

    References
    ----------
    .. [1] Ho, J., Jain, A., & Abbeel, P. "Denoising diffusion probabilistic
       models." NeurIPS 2020.

    Examples
    --------
    >>> import torch
    >>> model = BackboneDiffusion(coord_dim=9, embed_dim=128, num_layers=4)
    >>> noisy_coords = torch.randn(4, 50, 9)
    >>> timesteps = torch.randint(0, 1000, (4,))
    >>> noise_pred = model([noisy_coords, timesteps])
    >>> noise_pred.shape
    torch.Size([4, 50, 9])
    """

    def __init__(self,
                 coord_dim: int = 9,
                 embed_dim: int = 256,
                 time_dim: int = 128,
                 num_layers: int = 8,
                 num_heads: int = 8,
                 max_seq_len: int = 512,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.coord_dim = coord_dim
        self.embed_dim = embed_dim

        # Timestep embedding
        self.time_embedding = SinusoidalTimestepEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Coordinate embedding
        self.coord_embed = ResidueEmbedding(coord_dim, embed_dim)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(embed_dim, max_seq_len)

        # Transformer layers
        self.layers = nn.ModuleList([
            DiffusionTransformerBlock(embed_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Output projection to predict noise
        self.output_norm = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, coord_dim),
        )

        # Zero-initialize output for stable diffusion training
        nn.init.zeros_(self.output_proj[-1].weight)
        nn.init.zeros_(self.output_proj[-1].bias)

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        """Predict noise from noisy backbone coordinates and timesteps.

        Parameters
        ----------
        inputs : list of torch.Tensor
            A list of two tensors:
            - ``inputs[0]``: Noisy backbone coordinates of shape
              ``(batch, num_residues, coord_dim)``
            - ``inputs[1]``: Integer timesteps of shape ``(batch,)``

        Returns
        -------
        torch.Tensor
            Predicted noise of shape ``(batch, num_residues, coord_dim)``.
        """
        x_noisy, t = inputs[0], inputs[1]

        # Ensure t is long for embedding lookup
        t = t.long()

        # Timestep embedding
        t_emb = self.time_embedding(t)
        t_emb = self.time_mlp(t_emb)

        # Embed noisy coordinates
        h = self.coord_embed(x_noisy)

        # Add positional encoding
        h = self.pos_encoding(h)

        # Apply transformer blocks
        for layer in self.layers:
            h = layer(h, t_emb)

        # Predict noise
        h = self.output_norm(h)
        noise_pred = self.output_proj(h)

        return noise_pred


class CosineSchedule:
    """Cosine variance schedule for the diffusion process.

    Implements the cosine schedule from Improved DDPM [1]_, which provides
    better sample quality than the linear schedule for many domains.

    Parameters
    ----------
    num_timesteps : int, default 1000
        Total number of diffusion timesteps.
    s : float, default 0.008
        Small offset to prevent singularity at t=0.

    References
    ----------
    .. [1] Nichol, A. Q., & Dhariwal, P. "Improved denoising diffusion
       probabilistic models." ICML 2021.

    Examples
    --------
    >>> schedule = CosineSchedule(num_timesteps=100)
    >>> import torch
    >>> x0 = torch.randn(2, 10, 9)
    >>> t = torch.tensor([0, 50])
    >>> noisy_x, noise = schedule.q_sample(x0, t)
    >>> noisy_x.shape
    torch.Size([2, 10, 9])
    """

    def __init__(self, num_timesteps: int = 1000, s: float = 0.008) -> None:
        self.num_timesteps = num_timesteps

        # Compute cosine schedule
        steps = num_timesteps + 1
        t = torch.linspace(0, num_timesteps, steps) / num_timesteps
        alpha_cumprod = torch.cos((t + s) / (1 + s) * math.pi / 2)**2
        alpha_cumprod = alpha_cumprod / alpha_cumprod[0]
        betas = 1 - (alpha_cumprod[1:] / alpha_cumprod[:-1])
        betas = torch.clamp(betas, 0, 0.999)

        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        alpha_cumprod_prev = F.pad(alpha_cumprod[:-1], (1, 0), value=1.0)

        # Store precomputed values
        self.betas = betas
        self.alphas = alphas
        self.alpha_cumprod = alpha_cumprod
        self.sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - alpha_cumprod)
        self.sqrt_recip_alpha_cumprod = torch.sqrt(1.0 / alpha_cumprod)
        self.sqrt_recipm1_alpha_cumprod = torch.sqrt(1.0 / alpha_cumprod - 1)

        # Posterior variance for reverse sampling
        self.posterior_variance = (betas * (1.0 - alpha_cumprod_prev) /
                                   (1.0 - alpha_cumprod))
        self.posterior_log_variance = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20))
        self.posterior_mean_coef1 = (betas * torch.sqrt(alpha_cumprod_prev) /
                                     (1.0 - alpha_cumprod))
        self.posterior_mean_coef2 = ((1.0 - alpha_cumprod_prev) *
                                     torch.sqrt(alphas) / (1.0 - alpha_cumprod))

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward diffusion: add noise to clean data.

        Samples from q(x_t | x_0) = N(x_t; sqrt(alpha_cumprod_t) * x_0,
        (1 - alpha_cumprod_t) * I).

        Parameters
        ----------
        x0 : torch.Tensor
            Clean data of shape ``(batch, ...)``.
        t : torch.Tensor
            Integer timesteps of shape ``(batch,)``.
        noise : torch.Tensor, optional
            Pre-sampled noise. If None, noise is sampled from N(0, I).

        Returns
        -------
        noisy_x : torch.Tensor
            Noisy data, same shape as x0.
        noise : torch.Tensor
            The noise that was added, same shape as x0.
        """
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alpha = self.sqrt_alpha_cumprod[t].to(x0.device)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alpha_cumprod[t].to(
            x0.device)

        # Expand for broadcasting
        while sqrt_alpha.dim() < x0.dim():
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)

        noisy_x = sqrt_alpha * x0 + sqrt_one_minus_alpha * noise
        return noisy_x, noise

    @torch.no_grad()
    def p_sample(self, model: nn.Module, x_t: torch.Tensor,
                 t: torch.Tensor) -> torch.Tensor:
        """Reverse diffusion: denoise by one step.

        Samples from p(x_{t-1} | x_t) using the model's noise prediction.

        Parameters
        ----------
        model : nn.Module
            The denoiser network.
        x_t : torch.Tensor
            Current noisy data of shape ``(batch, num_residues, coord_dim)``.
        t : torch.Tensor
            Current timestep of shape ``(batch,)``.

        Returns
        -------
        torch.Tensor
            Denoised data at timestep t-1, same shape as x_t.
        """
        device = x_t.device
        noise_pred = model([x_t, t])

        # Use CPU indices for schedule tensor indexing (buffers stay on CPU)
        t_cpu = t.cpu()

        # Predict x0
        sqrt_recip = self.sqrt_recip_alpha_cumprod[t_cpu].to(device)
        sqrt_recipm1 = self.sqrt_recipm1_alpha_cumprod[t_cpu].to(device)
        while sqrt_recip.dim() < x_t.dim():
            sqrt_recip = sqrt_recip.unsqueeze(-1)
            sqrt_recipm1 = sqrt_recipm1.unsqueeze(-1)
        x0_pred = sqrt_recip * x_t - sqrt_recipm1 * noise_pred

        # Posterior mean
        coef1 = self.posterior_mean_coef1[t_cpu].to(device)
        coef2 = self.posterior_mean_coef2[t_cpu].to(device)
        while coef1.dim() < x_t.dim():
            coef1 = coef1.unsqueeze(-1)
            coef2 = coef2.unsqueeze(-1)
        posterior_mean = coef1 * x0_pred + coef2 * x_t

        # Add noise (except at t=0)
        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().to(device)
        while nonzero_mask.dim() < x_t.dim():
            nonzero_mask = nonzero_mask.unsqueeze(-1)

        var = self.posterior_variance[t_cpu].to(device)
        while var.dim() < x_t.dim():
            var = var.unsqueeze(-1)

        return posterior_mean + nonzero_mask * torch.sqrt(var) * noise

    @torch.no_grad()
    def sample(self, model: nn.Module, shape: Tuple[int, ...],
               device: torch.device) -> torch.Tensor:
        """Generate samples via full reverse diffusion.

        Starts from Gaussian noise and iteratively denoises for
        ``num_timesteps`` steps to produce clean samples.

        Parameters
        ----------
        model : nn.Module
            The denoiser network.
        shape : tuple of int
            Shape of the samples to generate ``(batch, num_residues, coord_dim)``.
        device : torch.device
            Device to generate samples on.

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
