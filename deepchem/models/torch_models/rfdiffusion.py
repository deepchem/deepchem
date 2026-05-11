"""Baseline diffusion layers for protein backbone coordinate generation.

This module implements a denoising diffusion probabilistic model (DDPM)
scaffold for protein backbone coordinates. It is inspired by the public
RFDiffusion architecture, but it is not a full reproduction of RFDiffusion:
the denoiser here is a coordinate Transformer, not a RoseTTAFold/SE(3)
equivariant network. It provides:

- Neural network layers (embeddings, Transformer blocks) for the denoiser
- A cosine variance schedule for the forward/reverse diffusion process
- ``RFDiffusionModel``, a ``TorchModel`` wrapper for DeepChem workflows

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
import numpy as np
from typing import Iterable, List, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError:
    raise ImportError('These classes require PyTorch to be installed.')

from deepchem.data import Dataset
from deepchem.models.torch_models.torch_model import TorchModel

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
        if dim <= 0:
            raise ValueError('dim must be positive.')
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
        half_dim = (self.dim + 1) // 2
        if half_dim == 1:
            emb = torch.ones(1, device=device)
        else:
            emb = math.log(10000) / (half_dim - 1)
            emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb[:, :self.dim]


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
        if embed_dim <= 0:
            raise ValueError('embed_dim must be positive.')
        if max_len <= 0:
            raise ValueError('max_len must be positive.')
        self.max_len = max_len
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() *
            (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])
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
        if x.size(1) > self.max_len:
            raise ValueError(
                f'Sequence length {x.size(1)} exceeds max_len {self.max_len}.')
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

    def forward(self,
                x: torch.Tensor,
                t_emb: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with timestep conditioning.

        Parameters
        ----------
        x : torch.Tensor
            Input features of shape ``(batch, seq_len, embed_dim)``.
        t_emb : torch.Tensor
            Timestep embedding of shape ``(batch, embed_dim)``.
        attention_mask : torch.Tensor, optional
            Boolean or numeric mask of shape ``(batch, seq_len)`` where true
            values indicate valid residues.

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
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.to(dtype=torch.bool)
        attn_out, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask)
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
    - Optional self-conditioning on previous denoised estimates [2]_

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
    self_conditioning : bool, default False
        If True, the network accepts a third input — the model's
        previous x0 prediction — and adds its embedding to the
        coordinate representation.  This can improve sample quality
        by letting the model refine its own estimates [2]_.

    References
    ----------
    .. [1] Ho, J., Jain, A., & Abbeel, P. "Denoising diffusion probabilistic
       models." NeurIPS 2020.
    .. [2] Watson, J. L., et al. "De novo design of protein structure and
       function with RFdiffusion." Nature 620.7976 (2023): 1089-1100.

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
                 dropout: float = 0.1,
                 self_conditioning: bool = False) -> None:
        super().__init__()
        if coord_dim <= 0:
            raise ValueError('coord_dim must be positive.')
        if embed_dim <= 0:
            raise ValueError('embed_dim must be positive.')
        if time_dim <= 0:
            raise ValueError('time_dim must be positive.')
        if num_layers <= 0:
            raise ValueError('num_layers must be positive.')
        if num_heads <= 0:
            raise ValueError('num_heads must be positive.')
        if embed_dim % num_heads != 0:
            raise ValueError('embed_dim must be divisible by num_heads.')
        if max_seq_len <= 0:
            raise ValueError('max_seq_len must be positive.')
        self.coord_dim = coord_dim
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.self_conditioning = self_conditioning

        # Timestep embedding
        self.time_embedding = SinusoidalTimestepEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Coordinate embedding
        self.coord_embed = ResidueEmbedding(coord_dim, embed_dim)

        # Self-conditioning projection: embed the model's previous x0
        # prediction and add it to the coordinate embedding
        if self_conditioning:
            self.self_cond_proj = ResidueEmbedding(coord_dim, embed_dim)

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
            A list containing:
            - ``inputs[0]``: Noisy backbone coordinates of shape
              ``(batch, num_residues, coord_dim)``
            - ``inputs[1]``: Integer timesteps of shape ``(batch,)``
            - ``inputs[2]`` (optional): Self-conditioning input of shape
              ``(batch, num_residues, coord_dim)``, the model's
              previous x0 prediction. Only used when
              ``self_conditioning=True``.

        Returns
        -------
        torch.Tensor
            Predicted noise of shape ``(batch, num_residues, coord_dim)``.
        """
        if len(inputs) < 2:
            raise ValueError(
                'BackboneDiffusion expects at least coordinates and timesteps.')

        x_noisy, t = inputs[0], inputs[1]
        if x_noisy.dim() != 3:
            raise ValueError(
                'Coordinates must have shape (batch, num_residues, coord_dim).')
        if x_noisy.size(-1) != self.coord_dim:
            raise ValueError(
                f'Expected coord_dim {self.coord_dim}, got {x_noisy.size(-1)}.')
        if x_noisy.size(1) > self.max_seq_len:
            raise ValueError(
                f'Sequence length {x_noisy.size(1)} exceeds max_seq_len {self.max_seq_len}.'
            )
        if t.dim() != 1 or t.size(0) != x_noisy.size(0):
            raise ValueError('Timesteps must have shape (batch,).')

        x0_prev = None
        attention_mask = None
        for extra in inputs[2:]:
            if extra is None:
                continue
            if extra.dim() == x_noisy.dim() and tuple(extra.shape) == tuple(
                    x_noisy.shape):
                if not self.self_conditioning:
                    raise ValueError(
                        'Self-conditioning input was provided to a model with self_conditioning=False.'
                    )
                x0_prev = extra
            else:
                attention_mask = extra

        if attention_mask is not None:
            if attention_mask.dim() == 3 and attention_mask.size(-1) == 1:
                attention_mask = attention_mask.squeeze(-1)
            if attention_mask.dim(
            ) != 2 or attention_mask.shape != x_noisy.shape[:2]:
                raise ValueError(
                    'attention_mask must have shape (batch, num_residues).')
            attention_mask = attention_mask.to(device=x_noisy.device,
                                               dtype=torch.bool)

        # Ensure t is long for embedding lookup
        t = t.long()

        # Timestep embedding
        t_emb = self.time_embedding(t)
        t_emb = self.time_mlp(t_emb)

        # Embed noisy coordinates
        h = self.coord_embed(x_noisy)

        # Add self-conditioning if provided
        if x0_prev is not None:
            h = h + self.self_cond_proj(x0_prev)

        # Add positional encoding
        h = self.pos_encoding(h)
        if attention_mask is not None:
            h = h * attention_mask.unsqueeze(-1)

        # Apply transformer blocks
        for layer in self.layers:
            h = layer(h, t_emb, attention_mask=attention_mask)
            if attention_mask is not None:
                h = h * attention_mask.unsqueeze(-1)

        # Predict noise
        h = self.output_norm(h)
        noise_pred = self.output_proj(h)
        if attention_mask is not None:
            noise_pred = noise_pred * attention_mask.unsqueeze(-1)

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
        if num_timesteps <= 0:
            raise ValueError('num_timesteps must be positive.')
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

        t = t.to(dtype=torch.long)
        if (t < 0).any() or (t >= self.num_timesteps).any():
            raise ValueError('Timesteps are out of range for this schedule.')

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
    def p_sample(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        x0_prev: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reverse diffusion: denoise by one step.

        Samples from p(x_{t-1} | x_t) using the model's noise prediction.
        When self-conditioning is used, ``x0_prev`` (the predicted clean
        sample from the previous step) is passed as a third model input.

        Parameters
        ----------
        model : nn.Module
            The denoiser network.
        x_t : torch.Tensor
            Current noisy data of shape ``(batch, num_residues, coord_dim)``.
        t : torch.Tensor
            Current timestep of shape ``(batch,)``.
        x0_prev : torch.Tensor or None, optional
            Previous x0 prediction for self-conditioning.

        Returns
        -------
        x_prev : torch.Tensor
            Denoised data at timestep t-1, same shape as x_t.
        x0_pred : torch.Tensor
            Predicted clean sample (x0) at this step.
        """
        device = x_t.device
        t = t.to(dtype=torch.long)
        if x0_prev is not None:
            noise_pred = model([x_t, t, x0_prev])
        else:
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

        x_prev = posterior_mean + nonzero_mask * torch.sqrt(var) * noise
        return x_prev, x0_pred

    @torch.no_grad()
    def sample(self,
               model: nn.Module,
               shape: Tuple[int, ...],
               device: torch.device,
               self_conditioning: bool = False) -> torch.Tensor:
        """Generate samples via full reverse diffusion.

        Starts from Gaussian noise and iteratively denoises for
        ``num_timesteps`` steps to produce clean samples.  When
        ``self_conditioning`` is True, the predicted x0 from each step
        is fed back into the model as conditioning for the next step.

        Parameters
        ----------
        model : nn.Module
            The denoiser network.
        shape : tuple of int
            Shape of the samples to generate
            ``(batch, num_residues, coord_dim)``.
        device : torch.device
            Device to generate samples on.
        self_conditioning : bool, default False
            Whether to use self-conditioning during sampling.

        Returns
        -------
        torch.Tensor
            Generated samples of the given shape.
        """
        was_training = model.training
        model.eval()
        try:
            x = torch.randn(shape, device=device)
            x0_pred = None

            for t_val in reversed(range(self.num_timesteps)):
                t = torch.full((shape[0],),
                               t_val,
                               device=device,
                               dtype=torch.long)
                x0_cond = x0_pred if self_conditioning else None
                x, x0_pred = self.p_sample(model, x, t, x0_prev=x0_cond)

            return x
        finally:
            model.train(was_training)


class RFDiffusionModel(TorchModel):
    """RFDiffusion model for protein backbone generation using DeepChem.

    This model implements a denoising diffusion probabilistic model (DDPM) for
    protein backbone coordinate generation. It wraps the BackboneDiffusion
    neural network in DeepChem's TorchModel interface, enabling training and
    model management through standard DeepChem workflows. Use ``generate`` for
    sampling; ``predict`` is intentionally unsupported because there is no
    deterministic target prediction for this generative model.

    The model operates on protein backbone coordinates represented as
    flattened arrays of shape ``(num_residues, 9)``, where 9 corresponds
    to 3 backbone atoms (N, CA, C) times 3 spatial dimensions (x, y, z).

    Training uses the standard DDPM objective: sample a random timestep,
    add noise to the clean coordinates, and train the model to predict
    the added noise. This is handled internally by overriding
    ``default_generator`` so that ``model.fit(dataset)`` works directly
    with DeepChem datasets.

    During generation the model applies the coordinate scale observed during
    training when those statistics are available. This returns coordinates on
    the training data scale, but it does not by itself guarantee physically
    valid protein geometry.

    Parameters
    ----------
    embed_dim : int, default 256
        Hidden dimension for the Transformer backbone.
    time_dim : int, default 128
        Dimension of the timestep embedding.
    num_layers : int, default 8
        Number of Transformer blocks.
    num_heads : int, default 8
        Number of attention heads per block.
    num_diffusion_steps : int, default 1000
        Number of timesteps in the diffusion process.
    max_seq_len : int, default 512
        Maximum supported protein length in residues.
    dropout : float, default 0.1
        Dropout probability.
    self_conditioning : bool, default False
        If True, the model conditions on its own previous x0 prediction
        during both training and sampling.  During training, 50%% of
        the time the model first predicts x0 (with gradients detached)
        and uses that as an additional input.  This technique is
        described in the RFDiffusion paper [1]_.
    batch_size : int, default 4
        Batch size for training.
    learning_rate : float, default 1e-4
        Learning rate for the Adam optimizer.
    device : torch.device, optional
        Device to use. If None, automatically selects GPU if available.
    **kwargs
        Additional keyword arguments passed to ``TorchModel``.

    Examples
    --------
    Training on a DeepChem dataset:

    >>> import numpy as np
    >>> import deepchem as dc
    >>> # Create a small dataset of protein backbone coordinates
    >>> # Each protein: (num_residues, 9) where 9 = 3 atoms * 3 xyz
    >>> proteins = [np.random.randn(20, 9).astype(np.float32) for _ in range(10)]
    >>> X = np.empty(10, dtype=object)
    >>> for i, p in enumerate(proteins):
    ...     X[i] = p
    >>> y = np.zeros((10, 1), dtype=np.float32)
    >>> dataset = dc.data.NumpyDataset(X=X, y=y)
    >>> model = dc.models.RFDiffusionModel(
    ...     embed_dim=64, num_layers=2, num_heads=4,
    ...     num_diffusion_steps=100, batch_size=2)
    >>> loss = model.fit(dataset, nb_epoch=1)

    Generating new protein backbones:

    >>> samples = model.generate(num_samples=2, seq_length=20)
    >>> samples.shape
    (2, 20, 9)

    Using with the CATH dataset:

    >>> tasks, datasets, transformers = dc.molnet.load_cath(  # doctest: +SKIP
    ...     featurizer='ProteinBackbone', splitter='random')
    >>> train, valid, test = datasets  # doctest: +SKIP
    >>> model = dc.models.RFDiffusionModel(  # doctest: +SKIP
    ...     embed_dim=256, num_layers=8, batch_size=4)
    >>> loss = model.fit(train, nb_epoch=100)  # doctest: +SKIP

    References
    ----------
    .. [1] Watson, J. L., et al. "De novo design of protein structure and
       function with RFdiffusion." Nature 620.7976 (2023): 1089-1100.
    .. [2] Ho, J., Jain, A., & Abbeel, P. "Denoising diffusion probabilistic
       models." NeurIPS 2020.

    Notes
    -----
    This class requires PyTorch to be installed.
    """

    def __init__(self,
                 embed_dim: int = 256,
                 time_dim: int = 128,
                 num_layers: int = 8,
                 num_heads: int = 8,
                 num_diffusion_steps: int = 1000,
                 max_seq_len: int = 512,
                 dropout: float = 0.1,
                 self_conditioning: bool = False,
                 batch_size: int = 4,
                 learning_rate: float = 1e-4,
                 device: Optional[torch.device] = None,
                 **kwargs) -> None:
        if embed_dim <= 0:
            raise ValueError('embed_dim must be positive.')
        if time_dim <= 0:
            raise ValueError('time_dim must be positive.')
        if num_layers <= 0:
            raise ValueError('num_layers must be positive.')
        if num_heads <= 0:
            raise ValueError('num_heads must be positive.')
        if num_diffusion_steps <= 0:
            raise ValueError('num_diffusion_steps must be positive.')
        if max_seq_len <= 0:
            raise ValueError('max_seq_len must be positive.')

        self.num_diffusion_steps = num_diffusion_steps
        self.max_seq_len = max_seq_len
        self.coord_dim = 9  # N, CA, C backbone atoms * 3 xyz
        self._self_conditioning = self_conditioning

        # Running statistics for denormalization
        self._train_mean: Optional[np.ndarray] = None
        self._train_std: Optional[float] = None

        # Create the diffusion schedule
        self.schedule = CosineSchedule(num_timesteps=num_diffusion_steps)

        # Create the denoiser network
        model = BackboneDiffusion(
            coord_dim=self.coord_dim,
            embed_dim=embed_dim,
            time_dim=time_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            dropout=dropout,
            self_conditioning=self_conditioning,
        )

        # Custom loss function for diffusion training.
        # The model predicts noise, and we compute MSE against the true noise.
        # The default_generator yields (inputs=[noisy_coords, timesteps],
        #   labels=[noise], weights=[ones]), so the loss receives
        #   outputs=[predicted_noise] and labels=[true_noise].
        def diffusion_loss(outputs: List, labels: List,
                           weights: List) -> torch.Tensor:
            pred_noise = outputs[0]
            true_noise = labels[0]
            w = weights[0]

            # MSE loss per element
            loss = F.mse_loss(pred_noise, true_noise, reduction='none')

            # Apply weights and average over valid coordinate entries.
            if w.dim() < loss.dim():
                w = w.reshape(w.shape + (1,) * (loss.dim() - w.dim()))
            denom = torch.clamp(w.expand_as(loss).sum(), min=1.0)
            loss = (loss * w).sum() / denom
            return loss

        super(RFDiffusionModel, self).__init__(model,
                                               loss=diffusion_loss,
                                               batch_size=batch_size,
                                               learning_rate=learning_rate,
                                               device=device,
                                               **kwargs)

    def _normalize_coords(self, coords: np.ndarray) -> np.ndarray:
        """Normalize protein coordinates for diffusion training.

        Centers coordinates around the CA centroid and scales to unit
        variance. This normalization is important for stable diffusion
        training since the noise schedule assumes data near zero.

        Parameters
        ----------
        coords : np.ndarray
            Backbone coordinates of shape ``(num_residues, 3, 3)`` or
            ``(num_residues, 9)``.

        Returns
        -------
        np.ndarray
            Normalized coordinates of shape ``(num_residues, 9)``.
        """
        if coords.ndim == 3 and coords.shape[1] == 3 and coords.shape[2] == 3:
            # Shape (L, 3, 3) -> (L, 9)
            coords = coords.reshape(-1, 9)
        elif coords.ndim != 2 or coords.shape[1] != self.coord_dim:
            raise ValueError(
                'coords must have shape (num_residues, 3, 3) or (num_residues, 9).'
            )

        if coords.shape[0] == 0:
            raise ValueError('coords must contain at least one residue.')

        # Center around CA centroid (CA is indices 3,4,5 in the flattened rep)
        ca_coords = coords[:, 3:6]
        centroid = ca_coords.mean(axis=0, keepdims=True)
        coords = coords - np.tile(centroid, 3)  # Subtract from all 3 atoms

        # Scale to unit variance
        std = coords.std()
        if std > 1e-6:
            coords = coords / std

        return coords.astype(np.float32)

    def _pad_coords(self, coords: np.ndarray,
                    max_len: int) -> Tuple[np.ndarray, int]:
        """Pad coordinates to a fixed length.

        Parameters
        ----------
        coords : np.ndarray
            Coordinates of shape ``(num_residues, 9)``.
        max_len : int
            Target length to pad to.

        Returns
        -------
        padded : np.ndarray
            Padded coordinates of shape ``(max_len, 9)``.
        orig_len : int
            Original length before padding.
        """
        orig_len = coords.shape[0]
        if orig_len > max_len:
            raise ValueError(
                f'Protein length {orig_len} exceeds target length {max_len}.')
        padded = np.zeros((max_len, self.coord_dim), dtype=np.float32)
        padded[:orig_len] = coords
        return padded, orig_len

    def default_generator(
            self,
            dataset: Dataset,
            epochs: int = 1,
            mode: str = 'fit',
            deterministic: bool = True,
            pad_batches: bool = True) -> Iterable[Tuple[List, List, List]]:
        """Create a generator for diffusion training batches.

        This overrides the parent ``default_generator`` to implement the
        diffusion training procedure. For each batch of protein coordinates:

        1. Normalize and pad coordinates to a consistent length
        2. Sample random diffusion timesteps
        3. Add noise according to the cosine schedule
        4. Yield ``(inputs, labels, weights)`` where:
           - inputs = [noisy_coords, timesteps, residue_mask]
           - labels = [true_noise]
           - weights = [residue_mask]

        Training statistics (mean and standard deviation) are accumulated
        so that ``generate`` can denormalize the output.

        Parameters
        ----------
        dataset : Dataset
            A DeepChem dataset where ``X`` contains protein backbone
            coordinate arrays.
        epochs : int, default 1
            Number of epochs to iterate over the data.
        mode : str, default 'fit'
            One of 'fit', 'predict', or 'uncertainty'.
        deterministic : bool, default True
            Whether to iterate in order or shuffle each epoch.
        pad_batches : bool, default True
            Whether to pad the last batch.

        Yields
        ------
        tuple of (list, list, list)
            ``([noisy_coords, timesteps, residue_mask], [noise], [weights])``
        """
        if mode != 'fit':
            raise NotImplementedError(
                'RFDiffusionModel does not support predict/uncertainty mode. '
                'Use generate() to sample protein backbones.')

        for epoch in range(epochs):
            for (X_b, y_b, w_b,
                 ids_b) in dataset.iterbatches(batch_size=self.batch_size,
                                               deterministic=deterministic,
                                               pad_batches=pad_batches):
                batch_coords = []
                batch_size = len(X_b)
                sample_weights = np.asarray(w_b, dtype=np.float32)
                if sample_weights.ndim == 0:
                    sample_weights = np.full((batch_size,),
                                             float(sample_weights),
                                             dtype=np.float32)
                else:
                    sample_weights = sample_weights.reshape(batch_size, -1)
                    sample_weights = sample_weights.max(axis=1).astype(
                        np.float32)

                # Find max length in this batch for padding
                lengths = []
                normalized = []
                for i in range(batch_size):
                    coords = X_b[i]
                    if isinstance(coords, np.ndarray) and coords.size > 0:
                        c = self._normalize_coords(coords)
                        normalized.append(c)
                        lengths.append(c.shape[0])
                    else:
                        # Handle empty or invalid entries (from pad_batches)
                        normalized.append(
                            np.zeros((1, self.coord_dim), dtype=np.float32))
                        lengths.append(1)

                max_len = max(lengths)
                if max_len > self.max_seq_len:
                    raise ValueError(
                        f'Protein length {max_len} exceeds max_seq_len {self.max_seq_len}. '
                        'Increase max_seq_len or filter/crop the dataset before training.'
                    )

                # Accumulate training statistics for denormalization
                if mode == 'fit':
                    all_raw = []
                    for i in range(batch_size):
                        if sample_weights[i] <= 0:
                            continue
                        coords = X_b[i]
                        if isinstance(coords, np.ndarray) and coords.size > 0:
                            flat = coords.reshape(-1, 9) if (coords.ndim
                                                             == 3) else coords
                            all_raw.append(flat)
                    if all_raw:
                        raw = np.concatenate(all_raw, axis=0)
                        ca = raw[:, 3:6]
                        centroid = ca.mean(axis=0, keepdims=True)
                        centered = raw - np.tile(centroid, 3)
                        batch_std = float(centered.std())
                        if self._train_std is None:
                            self._train_mean = centroid[0]
                            self._train_std = batch_std
                        else:
                            # Exponential moving average
                            alpha = 0.1
                            self._train_mean = ((1 - alpha) * self._train_mean +
                                                alpha * centroid[0])
                            self._train_std = ((1 - alpha) * self._train_std +
                                               alpha * batch_std)

                # Pad all to same length
                batch_masks = []
                for c in normalized:
                    padded, orig_len = self._pad_coords(c, max_len)
                    batch_coords.append(padded)
                    mask = np.zeros((max_len,), dtype=np.float32)
                    mask[:orig_len] = 1.0
                    batch_masks.append(mask)

                # Stack into batch
                coords_batch = np.stack(batch_coords, axis=0)  # (B, max_len, 9)
                mask_batch = np.stack(batch_masks, axis=0)  # (B, max_len)

                # Diffusion training: sample timesteps and add noise
                t = np.random.randint(0,
                                      self.num_diffusion_steps,
                                      size=(batch_size,))

                coords_tensor = torch.tensor(coords_batch, dtype=torch.float32)
                t_tensor = torch.tensor(t, dtype=torch.long)

                noisy_coords, noise = self.schedule.q_sample(
                    coords_tensor, t_tensor)

                # Convert back to numpy for TorchModel pipeline
                noisy_np = noisy_coords.numpy()
                noise_np = noise.numpy()
                t_np = t.astype(np.int64)

                weights = (mask_batch[:, :, None].astype(np.float32) *
                           sample_weights[:, None, None])

                # Self-conditioning: 50% of the time, compute x0 estimate
                # from a first pass (detached) and pass it as extra input.
                if self._self_conditioning and np.random.rand() > 0.5:
                    with torch.no_grad():
                        mask_tensor = torch.tensor(mask_batch,
                                                   dtype=torch.float32)
                        noise_est = self.model([
                            noisy_coords.to(self.device),
                            t_tensor.to(self.device),
                            mask_tensor.to(self.device)
                        ])
                        # Predict x0 from noisy coords and estimated noise
                        sc = self.schedule
                        t_cpu = t_tensor.cpu()
                        sr = sc.sqrt_recip_alpha_cumprod[t_cpu]
                        srm = sc.sqrt_recipm1_alpha_cumprod[t_cpu]
                        while sr.dim() < noisy_coords.dim():
                            sr = sr.unsqueeze(-1)
                            srm = srm.unsqueeze(-1)
                        x0_est = (
                            sr.to(self.device) * noisy_coords.to(self.device) -
                            srm.to(self.device) * noise_est)
                        x0_np = x0_est.cpu().numpy()
                    yield ([noisy_np, t_np, x0_np,
                            mask_batch], [noise_np], [weights])
                else:
                    yield ([noisy_np, t_np, mask_batch], [noise_np], [weights])

    def generate(self,
                 num_samples: int = 1,
                 seq_length: int = 50,
                 device: Optional[torch.device] = None) -> np.ndarray:
        """Generate new protein backbone structures.

        Starts from pure Gaussian noise and iteratively denoises for
        ``num_diffusion_steps`` steps using the learned reverse process.
        The raw output (in normalized space) is then rescaled with training
        statistics when available. This puts samples on the training coordinate
        scale, but does not guarantee physically valid protein geometry.

        If the model has not been trained (no statistics available), the
        raw samples are returned without denormalization.

        Parameters
        ----------
        num_samples : int, default 1
            Number of protein structures to generate.
        seq_length : int, default 50
            Length of each generated protein (number of residues).
        device : torch.device, optional
            Device to use for generation. Defaults to model device.

        Returns
        -------
        np.ndarray
            Generated backbone coordinates of shape
            ``(num_samples, seq_length, 9)``.

        Examples
        --------
        >>> import deepchem as dc
        >>> model = dc.models.RFDiffusionModel(
        ...     embed_dim=64, num_layers=2, num_heads=4,
        ...     num_diffusion_steps=50, batch_size=2)
        >>> samples = model.generate(num_samples=2, seq_length=30)
        >>> samples.shape
        (2, 30, 9)
        """
        if num_samples <= 0:
            raise ValueError('num_samples must be positive.')
        if seq_length <= 0:
            raise ValueError('seq_length must be positive.')
        if seq_length > self.max_seq_len:
            raise ValueError(
                f'seq_length {seq_length} exceeds max_seq_len {self.max_seq_len}.'
            )
        if device is None:
            device = self.device

        was_training = self.model.training
        shape = (num_samples, seq_length, self.coord_dim)
        try:
            samples = self.schedule.sample(
                self.model,
                shape,
                device,
                self_conditioning=self._self_conditioning)
        finally:
            self.model.train(was_training)

        result = samples.cpu().numpy()

        # Denormalize: reverse the centering and scaling applied during
        # training so that the output is in physical Angstrom coordinates.
        if self._train_std is not None and self._train_std > 1e-6:
            result = result * self._train_std
        if self._train_mean is not None:
            result = result + np.tile(self._train_mean, 3)

        return result

    def save_checkpoint(self,
                        max_checkpoints_to_keep: int = 5,
                        model_dir: Optional[str] = None) -> None:
        """Save model weights, optimizer state, and training statistics."""
        super().save_checkpoint(max_checkpoints_to_keep=max_checkpoints_to_keep,
                                model_dir=model_dir)
        if max_checkpoints_to_keep == 0:
            return
        checkpoint = sorted(self.get_checkpoints(model_dir))[0]
        data = torch.load(checkpoint, map_location=self.device)
        data['rf_diffusion_train_mean'] = (None if self._train_mean is None else
                                           self._train_mean.tolist())
        data['rf_diffusion_train_std'] = self._train_std
        torch.save(data, checkpoint)

    def restore(self,
                checkpoint: Optional[str] = None,
                model_dir: Optional[str] = None,
                strict: Optional[bool] = True) -> None:
        """Restore model weights, optimizer state, and training statistics."""
        if checkpoint is None:
            checkpoints = sorted(self.get_checkpoints(model_dir))
            if len(checkpoints) == 0:
                raise ValueError('No checkpoint found')
            checkpoint = checkpoints[0]
        super().restore(checkpoint=checkpoint,
                        model_dir=model_dir,
                        strict=strict)
        data = torch.load(checkpoint, map_location=self.device)
        train_mean = data.get('rf_diffusion_train_mean')
        self._train_mean = (None if train_mean is None else np.asarray(
            train_mean, dtype=np.float32))
        self._train_std = data.get('rf_diffusion_train_std')
