"""Neural network layers and noise schedule for RFDiffusion.

This module implements the building blocks for a protein backbone diffusion
model: input embeddings, the cosine noise schedule, and the Transformer
denoiser that predicts the noise at each diffusion step.

Classes
-------
- ``SinusoidalTimestepEmbedding`` — maps a diffusion timestep integer to a
  sinusoidal vector.
- ``ResidueEmbedding`` — projects the 9 backbone coordinates per residue into
  a feature space.
- ``PositionalEncoding`` — adds chain-position information to residue features.
- ``CosineSchedule`` — implements the improved cosine variance schedule for the
  forward and reverse diffusion steps.
- ``DiffusionTransformerBlock`` — a single Transformer block with adaptive
  timestep conditioning via scale-and-shift.
- ``BackboneDiffusion`` — the full denoiser network that stacks transformer
  blocks and predicts noise from noisy backbone coordinates.

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
from typing import List, Optional, Tuple

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

    The schedule tensors are computed once in ``__init__`` and kept as plain
    attributes (``self.betas``, ``self.alpha_cumprod`` and so on). They are
    fixed given ``num_timesteps`` and ``s``, so they are rebuilt on
    construction rather than saved in the checkpoint.

    Parameters
    ----------
    num_timesteps : int, default 1000
        Total number of diffusion timesteps. This is the ``T`` term from
        equation 17 of the Improved DDPM paper [1]_.
    s : float, default 0.008
        Small offset that keeps the schedule from reaching exactly zero at
        ``t=0``. The default 0.008 is the value from the Improved DDPM
        paper [1]_: it is chosen so ``sqrt(beta_0)`` sits just below the
        pixel bin size ``1 / 127.5 ~ 0.00784``.

    Notes
    -----
    ``RFDiffusionModel`` uses this schedule in two places: ``q_sample`` adds
    noise at a random timestep during training, and ``sample`` runs the full
    reverse loop to generate new backbones.

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

        # The schedule tensors are fixed given num_timesteps and s, so they
        # are kept as plain attributes rather than registered buffers.
        self.betas = betas
        self.alphas = alphas
        self.alpha_cumprod = alpha_cumprod
        self.sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - alpha_cumprod)
        self.sqrt_recip_alpha_cumprod = torch.sqrt(1.0 / alpha_cumprod)
        self.sqrt_recipm1_alpha_cumprod = torch.sqrt(1.0 / alpha_cumprod - 1)

        posterior_variance = (betas * (1.0 - alpha_cumprod_prev) /
                              (1.0 - alpha_cumprod))
        self.posterior_variance = posterior_variance
        self.posterior_log_variance = torch.log(
            torch.clamp(posterior_variance, min=1e-20))
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

        Examples
        --------
        >>> import torch
        >>> from deepchem.models.torch_models.rfdiffusion import CosineSchedule
        >>> schedule = CosineSchedule(num_timesteps=100)
        >>> x0 = torch.randn(2, 10, 9)
        >>> noisy_x, noise = schedule.q_sample(x0, torch.tensor([0, 50]))
        >>> noisy_x.shape
        torch.Size([2, 10, 9])
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

        Examples
        --------
        >>> import torch
        >>> from deepchem.models.torch_models.rfdiffusion import CosineSchedule
        >>> class DummyDenoiser(torch.nn.Module):
        ...     def forward(self, inputs):
        ...         return torch.zeros_like(inputs[0])
        >>> schedule = CosineSchedule(num_timesteps=10)
        >>> x_t = torch.randn(2, 6, 9)
        >>> x_prev = schedule.p_sample(
        ...     DummyDenoiser(), x_t, torch.tensor([5, 5]))
        >>> x_prev.shape
        torch.Size([2, 6, 9])
        """
        with torch.no_grad():
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

        Examples
        --------
        >>> import torch
        >>> from deepchem.models.torch_models.rfdiffusion import CosineSchedule
        >>> class DummyDenoiser(torch.nn.Module):
        ...     def forward(self, inputs):
        ...         return torch.zeros_like(inputs[0])
        >>> schedule = CosineSchedule(num_timesteps=10)
        >>> samples = schedule.sample(
        ...     DummyDenoiser(), (2, 6, 9), torch.device('cpu'))
        >>> samples.shape
        torch.Size([2, 6, 9])
        """
        model.eval()
        with torch.no_grad():
            x = torch.randn(shape, device=device)

            for t_val in reversed(range(self.num_timesteps)):
                t = torch.full((shape[0],),
                               t_val,
                               device=device,
                               dtype=torch.long)
                x = self.p_sample(model, x, t)

            return x


class DiffusionTransformerBlock(nn.Module):
    """Single Transformer block with diffusion timestep conditioning.

    A standard multi-head self-attention + MLP block, augmented with
    adaptive layer norm: the timestep embedding is projected into scale and
    shift parameters that modulate the intermediate features. This lets the
    denoiser behave differently at different noise levels without needing
    separate weights per timestep.

    Parameters
    ----------
    embed_dim : int
        Hidden dimension size.
    num_heads : int, default 8
        Number of attention heads.
    mlp_ratio : float, default 4.0
        Expansion ratio for the feedforward network.
    dropout : float, default 0.1
        Dropout probability applied in the attention and MLP sub-layers.

    Examples
    --------
    >>> import torch
    >>> from deepchem.models.torch_models.rfdiffusion import DiffusionTransformerBlock
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
        """Run the block with timestep conditioning.

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
    """Transformer denoiser for protein backbone diffusion.

    This is the core denoiser. It takes noisy backbone coordinates
    ``(batch, num_residues, 9)`` and a timestep vector ``(batch,)`` as
    input and outputs a noise prediction of the same shape. During training
    the model learns to predict the noise that was added to clean
    coordinates; during sampling the predicted noise is used to step
    backwards through the diffusion process.

    The output projection is zero-initialized so training starts in a stable
    state where the model initially predicts no noise.

    Parameters
    ----------
    coord_dim : int, default 9
        Input coordinate dimension (3 atoms × 3 xyz = 9).
    embed_dim : int, default 256
        Hidden dimension for the Transformer.
    time_dim : int, default 128
        Dimension of the timestep embedding.
    num_layers : int, default 8
        Number of ``DiffusionTransformerBlock`` layers.
    num_heads : int, default 8
        Number of attention heads per block.
    max_seq_len : int, default 512
        Maximum supported protein length in residues.
    dropout : float, default 0.1
        Dropout probability.

    References
    ----------
    .. [1] Ho, J., Jain, A., & Abbeel, P. "Denoising diffusion probabilistic
       models." NeurIPS 2020.

    Examples
    --------
    >>> import torch
    >>> from deepchem.models.torch_models.rfdiffusion import BackboneDiffusion
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

        self.time_embedding = SinusoidalTimestepEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.coord_embed = ResidueEmbedding(coord_dim, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, max_seq_len)
        self.layers = nn.ModuleList([
            DiffusionTransformerBlock(embed_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.output_norm = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, coord_dim),
        )
        # Zero-init so training starts from a stable baseline
        nn.init.zeros_(self.output_proj[-1].weight)
        nn.init.zeros_(self.output_proj[-1].bias)

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        """Predict noise from noisy backbone coordinates and timesteps.

        Parameters
        ----------
        inputs : list of torch.Tensor
            ``[noisy_coords, timesteps]`` where ``noisy_coords`` has shape
            ``(batch, num_residues, coord_dim)`` and ``timesteps`` has
            shape ``(batch,)``.

        Returns
        -------
        torch.Tensor
            Predicted noise of shape ``(batch, num_residues, coord_dim)``.
        """
        x_noisy, t = inputs[0], inputs[1]
        t = t.long()
        t_emb = self.time_mlp(self.time_embedding(t))
        h = self.pos_encoding(self.coord_embed(x_noisy))
        for layer in self.layers:
            h = layer(h, t_emb)
        return self.output_proj(self.output_norm(h))
