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

import logging
import math

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
        self.register_buffer(
            'posterior_mean_coef2',
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
    def p_sample(self,
                 model: nn.Module,
                 x_t: torch.Tensor,
                 t: torch.Tensor,
                 x0_prev: Optional[torch.Tensor] = None
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reverse diffusion: denoise x_t by one step using the model.

        Parameters
        ----------
        model : nn.Module
            The denoiser network. Called as ``model([x_t, t])`` or
            ``model([x_t, t, x0_prev])`` when self-conditioning.
        x_t : torch.Tensor
            Noisy data of shape ``(batch, num_residues, coord_dim)``.
        t : torch.Tensor
            Current timestep of shape ``(batch,)``.
        x0_prev : torch.Tensor or None, optional
            Previous step's x0 estimate for self-conditioning.

        Returns
        -------
        x_prev : torch.Tensor
            Denoised data at timestep t-1, same shape as x_t.
        x0_pred : torch.Tensor
            Predicted clean sample at this step (used for self-conditioning).
        """
        device = x_t.device
        if x0_prev is not None:
            noise_pred = model([x_t, t, x0_prev])
        else:
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

        x_prev = posterior_mean + nonzero_mask * torch.sqrt(var) * noise
        return x_prev, x0_pred

    @torch.no_grad()
    def sample(self,
               model: nn.Module,
               shape: Tuple[int, ...],
               device: torch.device,
               self_conditioning: bool = False) -> torch.Tensor:
        """Generate samples by running the full reverse diffusion loop.

        Parameters
        ----------
        model : nn.Module
            The denoiser network.
        shape : tuple of int
            Shape of samples to produce: ``(batch, num_residues, coord_dim)``.
        device : torch.device
            Device to generate on.
        self_conditioning : bool, default False
            If True, feed the predicted x0 from each step back into the
            model as an extra conditioning input for the next step.

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


class RFDiffusionModel(TorchModel):
    """TorchModel wrapper for the RFDiffusion protein backbone model.

    Wraps ``BackboneDiffusion`` in DeepChem's ``TorchModel`` interface so
    you can train with ``model.fit(dataset)`` and generate with
    ``model.generate()``. Uses the standard DDPM noise-prediction objective:
    at each training step a random timestep is sampled, noise is added to
    the clean coordinates, and the model learns to predict the noise.

    Parameters
    ----------
    embed_dim : int, default 256
        Hidden dimension for the Transformer backbone.
    time_dim : int, default 128
        Timestep embedding dimension.
    num_layers : int, default 8
        Number of Transformer blocks.
    num_heads : int, default 8
        Number of attention heads per block.
    num_diffusion_steps : int, default 1000
        Number of diffusion timesteps.
    max_seq_len : int, default 512
        Maximum supported protein length in residues.
    dropout : float, default 0.1
        Dropout probability.
    self_conditioning : bool, default False
        If True, the model conditions on its own previous x0 prediction
        during training (50% of the time) and sampling. This is the
        self-conditioning trick from the RFDiffusion paper [1]_.
    batch_size : int, default 4
        Training batch size.
    learning_rate : float, default 1e-4
        Learning rate for the Adam optimizer.
    device : torch.device, optional
        Device to use. Defaults to GPU if available.
    **kwargs
        Passed through to ``TorchModel``.

    Examples
    --------
    >>> import numpy as np
    >>> import deepchem as dc
    >>> proteins = [np.random.randn(20, 9).astype(np.float32) for _ in range(6)]
    >>> X = np.empty(6, dtype=object)
    >>> for i, p in enumerate(proteins):
    ...     X[i] = p
    >>> dataset = dc.data.NumpyDataset(X=X, y=np.zeros((6, 1), dtype=np.float32))
    >>> model = dc.models.RFDiffusionModel(
    ...     embed_dim=64, num_layers=2, num_heads=4,
    ...     num_diffusion_steps=50, batch_size=2)
    >>> loss = model.fit(dataset, nb_epoch=1)
    >>> samples = model.generate(num_samples=2, seq_length=20)
    >>> samples.shape
    (2, 20, 9)

    References
    ----------
    .. [1] Watson, J. L., et al. "De novo design of protein structure and
       function with RFdiffusion." Nature 620.7976 (2023): 1089-1100.
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
        self.coord_dim = 9
        self._self_conditioning = self_conditioning
        self._train_mean: Optional[np.ndarray] = None
        self._train_std: Optional[float] = None

        self.schedule = CosineSchedule(num_timesteps=num_diffusion_steps)
        model = BackboneDiffusion(
            coord_dim=self.coord_dim,
            embed_dim=embed_dim,
            time_dim=time_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )
        # Attach schedule so it moves with model.to(device)
        model.schedule = self.schedule

        def diffusion_loss(outputs: List, labels: List,
                           weights: List) -> torch.Tensor:
            pred_noise = outputs[0]
            true_noise = labels[0]
            w = weights[0]
            loss = F.mse_loss(pred_noise, true_noise, reduction='none')
            if w.dim() < loss.dim():
                w = w.reshape(w.shape + (1,) * (loss.dim() - w.dim()))
            denom = torch.clamp(w.expand_as(loss).sum(), min=1.0)
            return (loss * w).sum() / denom

        super(RFDiffusionModel, self).__init__(model,
                                               loss=diffusion_loss,
                                               batch_size=batch_size,
                                               learning_rate=learning_rate,
                                               device=device,
                                               **kwargs)

    def _normalize_coords(self, coords: np.ndarray) -> np.ndarray:
        """Center and scale backbone coordinates for diffusion training.

        Parameters
        ----------
        coords : np.ndarray
            Shape ``(L, 3, 3)`` or ``(L, 9)``.

        Returns
        -------
        np.ndarray
            Normalized coordinates of shape ``(L, 9)``.
        """
        if coords.ndim == 3 and coords.shape[1] == 3 and coords.shape[2] == 3:
            coords = coords.reshape(-1, 9)
        elif coords.ndim != 2 or coords.shape[1] != self.coord_dim:
            raise ValueError(
                'coords must have shape (L, 3, 3) or (L, 9).')
        if coords.shape[0] == 0:
            raise ValueError('coords must have at least one residue.')
        ca_coords = coords[:, 3:6]
        centroid = ca_coords.mean(axis=0, keepdims=True)
        coords = coords - np.tile(centroid, 3)
        std = coords.std()
        if std > 1e-6:
            coords = coords / std
        return coords.astype(np.float32)

    def _pad_coords(self, coords: np.ndarray,
                    max_len: int) -> Tuple[np.ndarray, int]:
        """Pad coordinate array to ``max_len`` with zeros.

        Parameters
        ----------
        coords : np.ndarray
            Shape ``(L, 9)``.
        max_len : int
            Target padded length.

        Returns
        -------
        padded : np.ndarray
            Shape ``(max_len, 9)``.
        orig_len : int
            Unpadded length.
        """
        orig_len = coords.shape[0]
        if orig_len > max_len:
            raise ValueError(
                f'Protein length {orig_len} exceeds target {max_len}.')
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
        """Yield diffusion training batches from a DeepChem dataset.

        For each batch the generator normalizes coordinates, pads them to
        a common length, samples random diffusion timesteps, adds noise
        using ``CosineSchedule.q_sample``, and yields
        ``([noisy_coords, timesteps, mask], [noise], [weights])``.

        Parameters
        ----------
        dataset : Dataset
            DeepChem dataset whose ``X`` entries are backbone coordinate arrays.
        epochs : int, default 1
        mode : str, default 'fit'
            Only 'fit' is supported; use ``generate()`` for sampling.
        deterministic : bool, default True
        pad_batches : bool, default True

        Yields
        ------
        tuple
            ``([noisy_coords, timesteps, mask], [noise], [weights])``.
        """
        if mode != 'fit':
            raise NotImplementedError(
                'RFDiffusionModel does not support predict/uncertainty mode. '
                'Use generate() instead.')

        for _epoch in range(epochs):
            for (X_b, _y_b, w_b,
                 _ids_b) in dataset.iterbatches(batch_size=self.batch_size,
                                                deterministic=deterministic,
                                                pad_batches=pad_batches):
                batch_size = len(X_b)
                sample_weights = np.asarray(w_b, dtype=np.float32)
                if sample_weights.ndim == 0:
                    sample_weights = np.full((batch_size,),
                                             float(sample_weights),
                                             dtype=np.float32)
                else:
                    sample_weights = sample_weights.reshape(
                        batch_size, -1).max(axis=1).astype(np.float32)

                normalized = []
                lengths = []
                for i in range(batch_size):
                    coords = X_b[i]
                    if isinstance(coords, np.ndarray) and coords.size > 0:
                        c = self._normalize_coords(coords)
                        normalized.append(c)
                        lengths.append(c.shape[0])
                    else:
                        normalized.append(
                            np.zeros((1, self.coord_dim), dtype=np.float32))
                        lengths.append(1)

                max_len = max(lengths)
                if max_len > self.max_seq_len:
                    raise ValueError(
                        f'Protein length {max_len} exceeds max_seq_len '
                        f'{self.max_seq_len}. Increase max_seq_len or crop.')

                # Update running training statistics for denormalization
                all_raw = [
                    X_b[i].reshape(-1, 9) if X_b[i].ndim == 3 else X_b[i]
                    for i in range(batch_size)
                    if (sample_weights[i] > 0 and isinstance(X_b[i], np.ndarray)
                        and X_b[i].size > 0)
                ]
                if all_raw:
                    raw = np.concatenate(all_raw, axis=0)
                    centroid = raw[:, 3:6].mean(axis=0, keepdims=True)
                    centered = raw - np.tile(centroid, 3)
                    batch_std = float(centered.std())
                    if self._train_std is None:
                        self._train_mean = centroid[0]
                        self._train_std = batch_std
                    else:
                        alpha = 0.1
                        assert self._train_mean is not None
                        self._train_mean = ((1 - alpha) * self._train_mean +
                                            alpha * centroid[0])
                        self._train_std = ((1 - alpha) * self._train_std +
                                           alpha * batch_std)

                batch_coords = []
                batch_masks = []
                for c in normalized:
                    padded, orig_len = self._pad_coords(c, max_len)
                    batch_coords.append(padded)
                    mask = np.zeros((max_len,), dtype=np.float32)
                    mask[:orig_len] = 1.0
                    batch_masks.append(mask)

                coords_batch = np.stack(batch_coords, axis=0)
                mask_batch = np.stack(batch_masks, axis=0)
                t = np.random.randint(0,
                                      self.num_diffusion_steps,
                                      size=(batch_size,))
                coords_tensor = torch.tensor(coords_batch, dtype=torch.float32)
                t_tensor = torch.tensor(t, dtype=torch.long)
                noisy_coords, noise = self.schedule.q_sample(
                    coords_tensor, t_tensor)

                noisy_np = noisy_coords.numpy()
                noise_np = noise.numpy()
                t_np = t.astype(np.int64)
                weights = (mask_batch[:, :, None].astype(np.float32) *
                           sample_weights[:, None, None])

                if self._self_conditioning and np.random.rand() > 0.5:
                    with torch.no_grad():
                        mask_tensor = torch.tensor(mask_batch,
                                                   dtype=torch.float32)
                        noise_est = self.model([
                            noisy_coords.to(self.device),
                            t_tensor.to(self.device),
                            mask_tensor.to(self.device)
                        ])
                        sc = self.schedule
                        t_idx = t_tensor.to(sc.betas.device)
                        sr = sc.sqrt_recip_alpha_cumprod[t_idx]
                        srm = sc.sqrt_recipm1_alpha_cumprod[t_idx]
                        while sr.dim() < noisy_coords.dim():
                            sr = sr.unsqueeze(-1)
                            srm = srm.unsqueeze(-1)
                        x0_est = (sr.to(self.device) *
                                  noisy_coords.to(self.device) -
                                  srm.to(self.device) * noise_est)
                        x0_np = x0_est.cpu().numpy()
                    yield ([noisy_np, t_np, x0_np, mask_batch], [noise_np],
                           [weights])
                else:
                    yield ([noisy_np, t_np, mask_batch], [noise_np], [weights])

    def generate(self,
                 num_samples: int = 1,
                 seq_length: int = 50,
                 device: Optional[torch.device] = None) -> np.ndarray:
        """Generate new protein backbone structures.

        Parameters
        ----------
        num_samples : int, default 1
            Number of structures to generate.
        seq_length : int, default 50
            Number of residues per structure.
        device : torch.device, optional
            Device to generate on. Defaults to the model's device.

        Returns
        -------
        np.ndarray
            Generated coordinates of shape
            ``(num_samples, seq_length, 9)``.
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
        if self._train_std is not None and self._train_std > 1e-6:
            result = result * self._train_std
        if self._train_mean is not None:
            result = result + np.tile(self._train_mean, 3)
        return result

    def save_checkpoint(self,
                        max_checkpoints_to_keep: int = 5,
                        model_dir: Optional[str] = None) -> None:
        """Save model weights plus training normalization statistics."""
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
        """Restore model weights and training normalization statistics."""
        if checkpoint is None:
            checkpoints = sorted(self.get_checkpoints(model_dir))
            if not checkpoints:
                raise ValueError('No checkpoint found.')
            checkpoint = checkpoints[0]
        super().restore(checkpoint=checkpoint,
                        model_dir=model_dir,
                        strict=strict)
        data = torch.load(checkpoint, map_location=self.device)
        train_mean = data.get('rf_diffusion_train_mean')
        self._train_mean = (None if train_mean is None else
                            np.asarray(train_mean, dtype=np.float32))
        self._train_std = data.get('rf_diffusion_train_std')
