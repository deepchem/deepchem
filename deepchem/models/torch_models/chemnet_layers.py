"""
Contains implementations of layers used in ChemCeption.

ChemCeption is a deep learning model designed for molecular property prediction using convolutional neural networks (CNNs).
It adapts the Inception-ResNet architecture to process molecular images effectively.

References:
- ChemCeption: https://arxiv.org/abs/1710.02238
"""
import torch
import torch.nn as nn


class Stem(nn.Module):
    """
    Implements the Stem Layer as defined in https://arxiv.org/abs/1710.02238.

    This layer serves as the initial processing block in ChemCeption,
    downsampling input images to reduce computational complexity
    before they pass through deeper network layers. The convolutional layer
    with stride 2 helps in feature extraction while reducing spatial dimensions.

    Examples
    --------
    >>> import numpy as np
    >>> import torch
    >>> from deepchem.models.torch_models.chemnet_layers import Stem
    >>> in_channels = 3
    >>> out_channels = 4
    >>> input_tensor = np.random.rand(1, in_channels, 32, 32).astype(np.float32)  # (Batch, Channels, Height, Width)
    >>> input_tensor_torch = torch.from_numpy(input_tensor)
    >>> layer = Stem(in_channels, out_channels)
    >>> output_tensor = layer(input_tensor_torch)
    >>> output_tensor.shape
    torch.Size([1, 4, 15, 15])
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """
        Initializes the Stem layer.

        Parameters
        ----------
        in_channels : int
            The number of channels in the input tensor.
        out_channels : int
            The number of filters applied in the convolution operation.
        """
        super(Stem, self).__init__()

        self.conv_layer = nn.Conv2d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=4,
                                    stride=2)

        self.activation_layer = nn.ReLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Stem layer.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of shape (batch_size, in_channels, H, W).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, out_channels, H_out, W_out), where
            H_out and W_out are reduced due to downsampling. The output is a
            feature map with extracted spatial representations.
        """
        conv1 = self.conv_layer(inputs)
        return self.activation_layer(conv1)
