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

None of these have learnable parameters that depend on later modules, so
they are shipped separately as the first small, reviewable PR.

References
----------
.. [1] Watson, J. L., et al. "De novo design of protein structure and function
   with RFdiffusion." Nature 620.7976 (2023): 1089-1100.
.. [2] Vaswani, A., et al. "Attention is all you need." NeurIPS 2017.

Notes
-----
This module requires PyTorch to be installed.
"""

import math

try:
    import torch
    import torch.nn as nn
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
