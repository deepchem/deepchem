from deepchem.models.torch_models.layers import SpectralConv
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple


class FNOBlock(nn.Module):
    """A single Fourier Neural Operator block.
    This block combines spectral convolution in Fourier space with a standard
    convolution to learn both global and local features. The spectral convolution
    operates on the Fourier coefficients of the input, while the standard convolution
    provides a residual connection.
    The forward pass computes:
    FNO_block(x) = ReLU(SpectralConv(x) + Conv(x))
    Example
    -------------
    >>> import torch
    >>> from deepchem.models.torch_models.fno import FNOBlock
    >>> block = FNOBlock(width=128, modes=8, dims=2)
    >>> x = torch.randn(1, 128, 16, 16)
    >>> output = block(x)
    """

    def __init__(self, width: int, modes: Union[int, Tuple[int, ...]],
                 dims: int) -> None:
        """Initialize the FNO block.
        Parameters
        ----------
        width: int
            Number of channels/features in the block
        modes: int or tuple
            Number of Fourier modes to keep in spectral convolution
        dims: int
            Spatial dimensionality (1, 2, or 3)
        """
        super().__init__()
        self.spectral_conv = SpectralConv(width, width, modes, dims=dims)
        self.w: Union[nn.Conv1d, nn.Conv2d, nn.Conv3d]

        if dims == 1:
            self.w = nn.Conv1d(width, width, 1)
        elif dims == 2:
            self.w = nn.Conv2d(width, width, 1)
        elif dims == 3:
            self.w = nn.Conv3d(width, width, 1)
        else:
            raise ValueError(f"Invalid dimension: {dims}. Must be 1, 2, or 3.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the FNO block.
        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape (batch, width, *spatial_dims)
        Returns
        -------
        torch.Tensor
            Output tensor of same shape as input after spectral and local convolution
        """
        x1 = self.spectral_conv(x)
        x2 = self.w(x)
        return F.relu(x1 + x2)
