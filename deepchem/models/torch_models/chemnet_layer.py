"""
Contains implementations of layers used in ChemCeption.
"""

import torch
import torch.nn as nn


class Stem(nn.Module):
    """
    Implements the Stem layer from ChemCeption as described in the paper:
    https://arxiv.org/abs/1710.02238.

    This layer downsamples the input image to reduce computational complexity
    before passing it to deeper layers.
    """

    def __init__(self, num_filters: int, input_shape: tuple) -> None:
        """
        Initializes the Stem layer.

        Args:
            num_filters (int): Number of output convolutional filters.
            input_shape (tuple): Shape of the input image as (C, H, W).
                - For `img_spec="std"`, use (1, H, W)  (Grayscale image).
                - For `img_spec="engd"`, use (4, H, W)  (Multi-channel image).
        """
        super().__init__()

        in_channels = input_shape[0]  # Extracts channels from (C, H, W)

        self.conv_layer = nn.Conv2d(in_channels=in_channels,
                                    out_channels=num_filters,
                                    kernel_size=4,
                                    stride=2)
        self.activation_layer = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Stem layer.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W).

        Returns:
            torch.Tensor: Output tensor after convolution and activation.
        """
        x = self.conv_layer(x)
        return self.activation_layer(x)
