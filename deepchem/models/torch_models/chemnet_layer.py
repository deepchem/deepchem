"""
Contains implementations of layers used in ChemCeption .
"""
import torch
import torch.nn as nn


class Stem(nn.Module):
    """
    Stem Layer as defined in https://arxiv.org/abs/1710.02238.
    This layer downsamples the image to reduce computational complexity before passing it to deeper layers.
    """

    def __init__(self: "Stem", num_filters: int, input_shape: tuple,
                 **kwargs) -> None:
        """
        Parameters
        ----------
        num_filters: int
            Number of convolutional filters.
        input_shape: tuple
            Shape of the input image in (H, W, C) format.
            - For `img_spec="std"`, use input_shape=(H, W, 1)  (Grayscale image)
            - For `img_spec="engd"`, use input_shape=(H, W, 4)  (Multi-channel image)
        """
        super(Stem, self).__init__()
        in_channels = input_shape[-1]  # Extracts channels from (H, W, C)

        self.conv_layer = nn.Conv2d(in_channels=in_channels,
                                    out_channels=num_filters,
                                    kernel_size=4,
                                    stride=2)  # No padding
        self.activation_layer = nn.ReLU()

    def forward(self: "Stem", inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Stem layer.
        """
        conv1 = self.conv_layer(inputs)
        return self.activation_layer(conv1)
