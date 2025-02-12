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

    def __init__(self, num_filters: int, **kwargs) -> None:
        """
        Parameters
        ----------
        in_channels: int,
            Number of input channels.
        num_filters: int,
            Number of convolutional filters.
        """
        super(Stem, self).__init__()
        self.conv_layer = nn.Conv2d(in_channels=1,
                                    out_channels=num_filters,
                                    kernel_size=4,
                                    stride=2)  # No padding
        self.activation_layer = nn.ReLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Stem layer.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of shape (B, C_in, H_in, W_in)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, num_filters, H_out, W_out)
        """
        conv1 = self.conv_layer(inputs)
        return self.activation_layer(conv1)
