"""
Contains implementations of layers used in ChemCeption .
"""
import torch
import torch.nn as nn


class Stem(nn.Module):
    """
    Implements Stem Layer as defined in https://arxiv.org/abs/1710.02238.
    This layer downsamples the image to reduce computational complexity before passing it to deeper layers.

    Examples
    --------
    >>> import numpy as np
    >>> import torch
    >>> import deepchem.models.torch_models.chemnet_layers as layers
    >>> in_channels = 3
    >>> out_channels = 4
    >>> input_tensor = np.random.rand(1, in_channels, 32, 32).astype(np.float32)  # (Batch, Channels, Height, Width)
    >>> input_tensor_torch = torch.tensor(input_tensor)
    >>> layer = layers.Stem(in_channels, out_channels)
    >>> output_tensor = layer(input_tensor_torch)
    >>> output_tensor.shape
    torch.Size([1, 4, 15, 15])
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """
        Parameters
        ----------
        in_channels: int
            Number of input channels
        out_channels: int
            Number of output channels.

        """
        super(Stem, self).__init__()

        self.conv_layer = nn.Conv2d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=4,
                                    stride=2)
        self.activation_layer = nn.ReLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Stem layer.
        """
        conv1 = self.conv_layer(inputs)
        return self.activation_layer(conv1)
