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


class InceptionResnetA(nn.Module):
    """
    Implements the Inception-ResNet-A block from the Inception-ResNet architecture
    as described in https://arxiv.org/abs/1710.0223.

    This block combines multiple convolutional branches with varying receptive fields,
    concatenates their outputs, projects them back to the input dimensions using a
    1x1 convolution, and adds the result to the original input (residual connection).
    A ReLU activation is applied at the end.

    Examples
    --------
    >>> import numpy as np
    >>> import torch
    >>> from deepchem.models.torch_models.chemnet_layers import InceptionResnetA
    >>> in_channels = 64
    >>> out_channels = 32
    >>> input_tensor = np.random.rand(1, in_channels, 28, 28).astype(np.float32) # (Batch, Channels, Height, Width)
    >>> input_tensor_torch = torch.from_numpy(input_tensor)
    >>> layer = InceptionResnetA(in_channels, out_channels)
    >>> output_tensor = layer(input_tensor_torch)
    >>> output_tensor.shape
    torch.Size([1, 64, 28, 28])
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """
        Initializes the Inception-ResNet-A block.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of filters in the convolutional branches.
        """
        super(InceptionResnetA, self).__init__()

        self.branch1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Conv2d(out_channels,
                      int(out_channels * 1.5),
                      kernel_size=3,
                      padding=1),
            nn.Conv2d(int(out_channels * 1.5),
                      out_channels * 2,
                      kernel_size=3,
                      padding=1))

        self.conv_linear = nn.Conv2d(out_channels * 4,
                                     in_channels,
                                     kernel_size=1)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Inception-ResNet-A block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_channels, H, W).

        Returns
        -------
        torch.Tensor
            Output tensor of the same shape as input (batch_size, in_channels, H, W),
            after applying the Inception-ResNet-A transformations and residual connection.
        """
        branches = [self.branch1(x), self.branch2(x), self.branch3(x)]
        concat = torch.cat(branches, dim=1)
        linear = self.conv_linear(concat)
        return self.activation(x + linear)


class InceptionResnetB(nn.Module):
    """
    Implements the Inception-ResNet-B block from the Inception-ResNet architecture
    as described in https://arxiv.org/abs/1710.0223.

    This block consists of two parallel branches:
    - A simple 1x1 convolution.
    - A deeper sequence with asymmetric convolutions (1x7 followed by 7x1) for
      efficient large receptive field learning.

    Outputs from both branches are concatenated and passed through a 1x1 convolution
    to project back to the original input dimension, and added to the input
    (residual connection). A ReLU activation follows.

    Examples
    --------
    >>> import numpy as np
    >>> import torch
    >>> from deepchem.models.torch_models.chemnet_layers import InceptionResnetB
    >>> in_channels = 64
    >>> out_channels = 32
    >>> input_tensor = np.random.rand(1, in_channels, 28, 28).astype(np.float32)
    >>> input_tensor_torch = torch.from_numpy(input_tensor)
    >>> layer = InceptionResnetB(in_channels, out_channels)
    >>> output_tensor = layer(input_tensor_torch)
    >>> output_tensor.shape
    torch.Size([1, 64, 28, 28])
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """
        Initializes the Inception-ResNet-B block.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of filters used in the convolutional branches.
        """
        super().__init__()

        # Branch 1: 1x1 Convolution
        self.branch1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # Branch 2: 1x1 → 1x7 → 7x1 Convolutions
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Conv2d(out_channels,
                      int(out_channels * 1.25),
                      kernel_size=(1, 7),
                      padding=(0, 3)),
            nn.Conv2d(int(out_channels * 1.25),
                      int(out_channels * 1.5),
                      kernel_size=(7, 1),
                      padding=(3, 0)))

        # Project concatenated features back to input channel dimension
        self.conv_linear = nn.Conv2d(out_channels + int(out_channels * 1.5),
                                     in_channels,
                                     kernel_size=1)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Inception-ResNet-B block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_channels, H, W)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, in_channels, H, W)
        """
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        concat = torch.cat([branch1, branch2], dim=1)
        linear = self.conv_linear(concat)
        return self.activation(x + linear)
