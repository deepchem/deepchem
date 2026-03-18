import torch.nn as nn
from torch import Tensor
from deepchem.models.torch_models import TorchModel
import math
import deepchem as dc
import numpy as np
from typing import List, Union, cast
from deepchem.utils.data_utils import load_from_disk, save_to_disk


class InvertedResidual(nn.Module):
    """
    Inverted Residual block used in MobileNetV2 architecture.

    This block uses a combination of pointwise, depthwise, and another pointwise convolution
    with optional residual connections based on input/output channels and stride.

    Parameters
    ----------
    inp : int
        Number of input channels.
    oup : int
        Number of output channels.
    stride : int
        Stride for depthwise convolution. Must be 1 or 2.
    expand_ratio : float
        Expansion ratio for the hidden dimension. If 1, the input is not expanded.

    Returns
    -------
    use_res_connect : bool
        Whether to use the residual connection.
    conv : nn.Sequential
        The core convolutional operations in the block.

    Examples
    --------
    >>> import torch
    >>> x = torch.randn(1, 16, 32, 32)
    >>> block = InvertedResidual(inp=16, oup=16, stride=1, expand_ratio=1)
    >>> out = block(x)
    >>> out.shape
    torch.Size([1, 16, 32, 32])

    """

    def __init__(self, inp: int, oup: int, stride: int,
                 expand_ratio: float) -> None:
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2], "Stride must be 1 or 2."

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim,
                          hidden_dim,
                          3,
                          stride,
                          1,
                          groups=hidden_dim,
                          bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim,
                          hidden_dim,
                          3,
                          stride,
                          1,
                          groups=hidden_dim,
                          bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the InvertedResidual block.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (N, C, H, W)

        Returns
        -------
        Tensor
            Output tensor of shape (N, oup, H_out, W_out)
        """
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    """
    MobileNetV2 architecture with support for arbitrary input channels (e.g., for 6-channel images).

    This version of MobileNetV2 is modified to accept custom input channels and configurable width scaling.
    It uses inverted residual blocks for efficient mobile/edge inference.

    Parameters
    ----------
    n_class : int, default=1000
        Number of output classes (for classification) or output units (for regression).
    input_size : int, default=224
        Spatial resolution of the input image. Must be divisible by 32.
    width_mult : float, default=1.0
        Width multiplier for the entire network to adjust model size.
    in_channels : int, default=6
        Number of input channels for the first convolutional layer.
    """

    def __init__(self,
                 n_class: int = 1000,
                 input_size: int = 224,
                 width_mult: float = 1.0,
                 in_channels: int = 6) -> None:
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0, "Input size must be divisible by 32."
        self.last_channel = self._make_divisible(
            last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features: Union[List[nn.Module], nn.Sequential] = [
            self._conv_bn(in_channels, input_channel, 2)
        ]  # Modified to support multi-channel input
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = self._make_divisible(c *
                                                  width_mult) if t > 1 else c
            for i in range(n):
                if i == 0:
                    self.features.append(
                        block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(
                        block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        self.features.append(self._conv_1x1_bn(input_channel,
                                               self.last_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Linear(self.last_channel, n_class)

        self._initialize_weights()

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for MobileNetV2.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (N, in_channels, H, W)

        Returns
        -------
        Tensor
            Output tensor of shape (N, n_class)
        """
        features = cast(nn.Sequential, self.features)
        x = features(x)
        x = x.mean(3).mean(2)  # Global average pooling
        x = self.classifier(x)
        return x

    def _initialize_weights(self) -> None:
        """Initialize weights using He initialization for conv and standard init for BN and Linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def _conv_bn(self, inp: int, oup: int, stride: int) -> nn.Sequential:
        """3x3 Convolution + BatchNorm + ReLU6"""
        return nn.Sequential(nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                             nn.BatchNorm2d(oup), nn.ReLU6(inplace=True))

    def _conv_1x1_bn(self, inp: int, oup: int) -> nn.Sequential:
        """1x1 Convolution + BatchNorm + ReLU6"""
        return nn.Sequential(nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                             nn.BatchNorm2d(oup), nn.ReLU6(inplace=True))

    def _make_divisible(self, x: float, divisible_by: int = 8) -> int:
        """
        Make channel size divisible by `divisible_by`, used for efficient tensor ops.

        Parameters
        ----------
        x : float
            Original value
        divisible_by : int
            Factor to make x divisible by

        Returns
        -------
        int
            Adjusted value
        """
        return int(np.ceil(x * 1. / divisible_by) * divisible_by)


class MobileNetV2Model(TorchModel):
    """
    MobileNetV2 with multi-channel support and classification/regression modes.

    MobileNetV2 is a lightweight and efficient convolutional neural network
    architecture designed for mobile and edge devices. It builds on the success
    of depthwise separable convolutions introduced in MobileNetV1, but introduces
    two key innovations: inverted residual blocks and linear bottlenecks. Unlike
    traditional residual blocks that compress then expand features, MobileNetV2
    first expands the input channels, applies a depthwise convolution (which
    processes each channel independently), and then projects it back down to a
    lower-dimensional space using a pointwise (1x1) convolution. This "inverted"
    structure preserves rich information in the high-dimensional space while
    maintaining efficiency. A linear layer (without non-linearity) at the bottleneck
    helps retain feature information during compression. Residual connections are
    selectively added when the input and output shapes match, which stabilizes
    training and improves accuracy.

    Parameters
    ----------
    n_tasks: int
        Number of tasks (output dimensions)
    in_channels: int, default 6
        Number of input channels
    input_size: int, default 224
        Input image size (must be divisible by 32)
    mode: str, default "classification"
        Either "regression" or "classification"
    n_classes: int, default 2
        Number of classes to predict (only used in classification mode)
    width_mult: float, default 1.0
        Width multiplier for the network

    References
    ----------
    .. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). MobileNetV2: Inverted residuals and linear bottlenecks. *arXiv preprint arXiv:1801.04381*.

    """

    def __init__(self,
                 n_tasks: int,
                 in_channels: int = 6,
                 input_size: int = 224,
                 mode: str = "classification",
                 n_classes: int = 2,
                 width_mult: float = 1.0,
                 **kwargs):
        if mode not in ['classification', 'regression']:
            raise ValueError(
                "mode must be either 'classification' or 'regression'")

        self.n_tasks = n_tasks
        self.in_channels = in_channels
        self.mode = mode
        self.n_classes = n_classes

        if mode == 'classification':
            output_dim = n_tasks * n_classes
        else:
            output_dim = n_tasks

        model = MobileNetV2(n_class=output_dim,
                            input_size=input_size,
                            width_mult=width_mult,
                            in_channels=in_channels)

        loss: Union[dc.models.losses.L2Loss,
                    dc.models.losses.SparseSoftmaxCrossEntropy]
        if mode == 'classification':
            loss = dc.models.losses.SparseSoftmaxCrossEntropy()
        else:
            loss = dc.models.losses.L2Loss()

        super(MobileNetV2Model, self).__init__(model=model, loss=loss, **kwargs)

    def save(self):
        """Saves model to disk using joblib."""
        save_to_disk(self.model, self.get_model_filename(self.model_dir))

    def reload(self):
        """Loads model from joblib file on disk."""
        self.model = load_from_disk(self.get_model_filename(self.model_dir))
