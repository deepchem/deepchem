import torch
import torch.nn as nn
import torch.nn.functional as F
from deepchem.models.losses import CategoricalCrossEntropy
from deepchem.models.torch_models import TorchModel
from deepchem.data import Dataset
from deepchem.models.optimizers import RMSProp
from typing import Optional, List, Callable, Any, Union
from deepchem.utils.data_utils import load_from_disk, save_to_disk
from deepchem.utils.typing import LossFn


class InceptionV3(nn.Module):
    """
    InceptionV3 model architecture for image classification.
    """

    def __init__(self,
                 num_classes: int = 1000,
                 aux_logits: bool = True,
                 transform_input: bool = False,
                 in_channels: int = 6,
                 dropout_rate: float = 0.5) -> None:
        super(InceptionV3, self).__init__()
        self.aux_logits = aux_logits

        self.Conv2d_1a_3x3 = BasicConv2d(in_channels,
                                         32,
                                         kernel_size=(3, 3),
                                         stride=(2, 2),
                                         padding=(0, 0))
        self.Conv2d_2a_3x3 = BasicConv2d(32,
                                         32,
                                         kernel_size=(3, 3),
                                         stride=(1, 1),
                                         padding=(0, 0))
        self.Conv2d_2b_3x3 = BasicConv2d(32,
                                         64,
                                         kernel_size=(3, 3),
                                         stride=(1, 1),
                                         padding=(1, 1))
        self.maxpool1 = nn.MaxPool2d((3, 3), (2, 2))
        self.Conv2d_3b_1x1 = BasicConv2d(64,
                                         80,
                                         kernel_size=(1, 1),
                                         stride=(1, 1),
                                         padding=(0, 0))
        self.Conv2d_4a_3x3 = BasicConv2d(80,
                                         192,
                                         kernel_size=(3, 3),
                                         stride=(1, 1),
                                         padding=(0, 0))
        self.maxpool2 = nn.MaxPool2d((3, 3), (2, 2))

        self.Mixed_5b = InceptionA(192, 32)
        self.Mixed_5c = InceptionA(256, 64)
        self.Mixed_5d = InceptionA(288, 64)

        self.Mixed_6a = InceptionB(288)

        self.Mixed_6b = InceptionC(768, 128)
        self.Mixed_6c = InceptionC(768, 160)
        self.Mixed_6d = InceptionC(768, 160)
        self.Mixed_6e = InceptionC(768, 192)

        # Auxiliary classifier (only used during training)
        if aux_logits:
            self.AuxLogits = InceptionAux(768, num_classes)

        self.Mixed_7a = InceptionD(768)
        self.Mixed_7b = InceptionE(1280)
        self.Mixed_7c = InceptionE(2048)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate, True)
        self.fc = nn.Linear(2048, num_classes)

        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                stddev = float(module.stddev) if hasattr(module,
                                                         "stddev") else 0.1
                torch.nn.init.trunc_normal_(module.weight,
                                            mean=0.0,
                                            std=stddev,
                                            a=-2,
                                            b=2)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor):
        # N x 6 x 299 x 299
        x = self.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = self.maxpool1(x)
        # N x 64 x 73 x 73
        x = self.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = self.maxpool2(x)
        # N x 192 x 35 x 35
        x = self.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6e(x)

        # Auxiliary output if training
        if self.aux_logits and self.training:
            aux = self.AuxLogits(x)

        # N x 768 x 17 x 17
        x = self.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.Mixed_7c(x)
        # N x 2048 x 8 x 8
        # Adaptive average pooling
        x = self.avgpool(x)
        # N x 2048 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 2048 x 1 x 1
        x = self.dropout(x)
        # N x 2048
        x = self.fc(x)
        # N x 3 (num_classes)

        if self.aux_logits and self.training:
            return x, aux
        else:
            return x


# Helper layers
class BasicConv2d(nn.Module):
    """
    A basic convolutional layer with Conv2d, BatchNorm2d, and ReLU activation.

    Examples
    --------
    >>> import torch
    >>> from deepchem.models.torch_models.inceptionv3 import BasicConv2d
    >>> layer = BasicConv2d(6, 32, kernel_size=3, stride=2)
    >>> x = torch.randn(5, 6, 299, 299)
    >>> output = layer(x)
    >>> output.shape
    torch.Size([5, 32, 149, 149])
    """

    def __init__(self, in_channels: int, out_channels: int,
                 **kwargs: Any) -> None:
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the BasicConv2d layer.
        """
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class InceptionA(nn.Module):
    """
    An InceptionA module as part of the InceptionV3 network architecture.
    This module performs parallel convolutions on the input using 1x1,
    5x5, and double 3x3 kernel sizes, as well as an average pooling layer,
    before concatenating the results along the channel dimension.

    Examples
    --------
    >>> import torch
    >>> from deepchem.models.torch_models.inceptionv3 import InceptionA
    >>> layer = InceptionA(192, pool_features=32)
    >>> x = torch.randn(5, 192, 35, 35)
    >>> output = layer(x)
    >>> output.shape
    torch.Size([5, 256, 35, 35])
    """

    def __init__(
        self,
        in_channels: int,
        pool_features: int,
    ) -> None:
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels,
                                     64,
                                     kernel_size=(1, 1),
                                     stride=(1, 1),
                                     padding=(0, 0))

        self.branch5x5_1 = BasicConv2d(in_channels,
                                       48,
                                       kernel_size=(1, 1),
                                       stride=(1, 1),
                                       padding=(0, 0))
        self.branch5x5_2 = BasicConv2d(48,
                                       64,
                                       kernel_size=(5, 5),
                                       stride=(1, 1),
                                       padding=(2, 2))

        self.branch3x3dbl_1 = BasicConv2d(in_channels,
                                          64,
                                          kernel_size=(1, 1),
                                          stride=(1, 1),
                                          padding=(0, 0))
        self.branch3x3dbl_2 = BasicConv2d(64,
                                          96,
                                          kernel_size=(3, 3),
                                          stride=(1, 1),
                                          padding=(1, 1))
        self.branch3x3dbl_3 = BasicConv2d(96,
                                          96,
                                          kernel_size=(3, 3),
                                          stride=(1, 1),
                                          padding=(1, 1))

        self.branch_pool = BasicConv2d(in_channels,
                                       pool_features,
                                       kernel_size=(1, 1),
                                       stride=(1, 1),
                                       padding=(0, 0))
        # self.avgpool = nn.AvgPool2d((3, 3), (1, 1), (1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the InceptionA module.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (N, in_channels, H, W).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (N, out_channels, H, W),
            where out_channels is the total number of output channels
            from all branches combined.
        """
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        out = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        output = torch.cat(out, 1)

        return output


class InceptionB(nn.Module):
    """
    An InceptionB module as part of the InceptionV3 network architecture.
    This module performs parallel operations on the input, including a
    3x3 convolution, a double 3x3 convolution, and a max-pooling layer,
    before concatenating the results along the channel dimension.

    Examples
    --------
    >>> import torch
    >>> from deepchem.models.torch_models.inceptionv3 import InceptionB
    >>> layer = InceptionB(288)
    >>> x = torch.randn(5, 288, 35, 35)
    >>> output = layer(x)
    >>> output.shape
    torch.Size([5, 768, 17, 17])
    """

    def __init__(
        self,
        in_channels: int,
    ) -> None:
        super(InceptionB, self).__init__()

        self.branch3x3 = BasicConv2d(in_channels,
                                     384,
                                     kernel_size=(3, 3),
                                     stride=(2, 2),
                                     padding=(0, 0))

        self.branch3x3dbl_1 = BasicConv2d(in_channels,
                                          64,
                                          kernel_size=(1, 1),
                                          stride=(1, 1),
                                          padding=(0, 0))
        self.branch3x3dbl_2 = BasicConv2d(64,
                                          96,
                                          kernel_size=(3, 3),
                                          stride=(1, 1),
                                          padding=(1, 1))
        self.branch3x3dbl_3 = BasicConv2d(96,
                                          96,
                                          kernel_size=(3, 3),
                                          stride=(2, 2),
                                          padding=(0, 0))

        # self.maxpool = nn.MaxPool2d((3, 3), (2, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the InceptionB module.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (N, in_channels, H, W).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (N, out_channels, H, W),
            where out_channels is the total number of output channels
            from all branches combined.
        """
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)

        out = [branch3x3, branch3x3dbl, branch_pool]
        output = torch.cat(out, 1)

        return output


class InceptionC(nn.Module):
    """
    An InceptionC module as part of the InceptionV3 network
    architecture. This module performs parallel operations on the
    input, including 1x1, 7x7, double 7x7 convolutions, and an
    average pooling layer, before concatenating the results along
    the channel dimension.

    Examples
    --------
    >>> import torch
    >>> from deepchem.models.torch_models.inceptionv3 import InceptionC
    >>> layer = InceptionC(768, channels_7x7=128)
    >>> x = torch.randn(5, 768, 17, 17)
    >>> output = layer(x)
    >>> output.shape
    torch.Size([5, 768, 17, 17])
    """

    def __init__(
        self,
        in_channels: int,
        channels_7x7: int,
    ) -> None:
        super(InceptionC, self).__init__()

        self.branch1x1 = BasicConv2d(in_channels, 192, kernel_size=1)

        self.branch7x7_1 = BasicConv2d(in_channels,
                                       channels_7x7,
                                       kernel_size=(1, 1),
                                       stride=(1, 1),
                                       padding=(0, 0))
        self.branch7x7_2 = BasicConv2d(channels_7x7,
                                       channels_7x7,
                                       kernel_size=(1, 7),
                                       stride=(1, 1),
                                       padding=(0, 3))
        self.branch7x7_3 = BasicConv2d(channels_7x7,
                                       192,
                                       kernel_size=(7, 1),
                                       stride=(1, 1),
                                       padding=(3, 0))

        self.branch7x7dbl_1 = BasicConv2d(in_channels,
                                          channels_7x7,
                                          kernel_size=1)
        self.branch7x7dbl_2 = BasicConv2d(channels_7x7,
                                          channels_7x7,
                                          kernel_size=(7, 1),
                                          stride=(1, 1),
                                          padding=(3, 0))
        self.branch7x7dbl_3 = BasicConv2d(channels_7x7,
                                          channels_7x7,
                                          kernel_size=(1, 7),
                                          stride=(1, 1),
                                          padding=(0, 3))
        self.branch7x7dbl_4 = BasicConv2d(channels_7x7,
                                          channels_7x7,
                                          kernel_size=(7, 1),
                                          stride=(1, 1),
                                          padding=(3, 0))
        self.branch7x7dbl_5 = BasicConv2d(channels_7x7,
                                          192,
                                          kernel_size=(1, 7),
                                          stride=(1, 1),
                                          padding=(0, 3))

        # self.avgpool = nn.AvgPool2d((3, 3), (1, 1), (1, 1))
        self.branch_pool = BasicConv2d(in_channels,
                                       192,
                                       kernel_size=(1, 1),
                                       stride=(1, 1),
                                       padding=(0, 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the InceptionC module.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (N, in_channels, H, W).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (N, out_channels, H, W),
            where out_channels is the total number of output channels
            from all branches combined.
        """
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        out = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        output = torch.cat(out, 1)

        return output


class InceptionD(nn.Module):
    """
    An InceptionD module as part of the InceptionV3 network
    architecture. This module performs parallel operations on the input,
    including 3x3, 7x7x3 convolutions, and a max-pooling layer, before
    concatenating the results along the channel dimension.

    Examples
    --------
    >>> import torch
    >>> from deepchem.models.torch_models.inceptionv3 import InceptionD
    >>> layer = InceptionD(768)
    >>> x = torch.randn(5, 768, 17, 17)
    >>> output = layer(x)
    >>> output.shape
    torch.Size([5, 1280, 8, 8])
    """

    def __init__(
        self,
        in_channels: int,
    ) -> None:
        super(InceptionD, self).__init__()
        self.branch3x3_1 = BasicConv2d(in_channels,
                                       192,
                                       kernel_size=(1, 1),
                                       stride=(1, 1),
                                       padding=(0, 0))
        self.branch3x3_2 = BasicConv2d(192,
                                       320,
                                       kernel_size=(3, 3),
                                       stride=(2, 2),
                                       padding=(0, 0))

        self.branch7x7x3_1 = BasicConv2d(in_channels,
                                         192,
                                         kernel_size=(1, 1),
                                         stride=(1, 1),
                                         padding=(0, 0))
        self.branch7x7x3_2 = BasicConv2d(192,
                                         192,
                                         kernel_size=(1, 7),
                                         stride=(1, 1),
                                         padding=(0, 3))
        self.branch7x7x3_3 = BasicConv2d(192,
                                         192,
                                         kernel_size=(7, 1),
                                         stride=(1, 1),
                                         padding=(3, 0))
        self.branch7x7x3_4 = BasicConv2d(192,
                                         192,
                                         kernel_size=(3, 3),
                                         stride=(2, 2),
                                         padding=(0, 0))

        # self.maxpool = nn.MaxPool2d((3, 3), (2, 2))

    def forward(self, x):
        """
        Forward pass for the InceptionD module.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (N, in_channels, H, W).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (N, out_channels, H, W),
            where out_channels is the total number of output channels
            from all branches combined.
        """
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)

        out = [branch3x3, branch7x7x3, branch_pool]
        output = torch.cat(out, 1)

        return output


class InceptionE(nn.Module):
    """
    InceptionE module as part of the InceptionV3 architecture.
    This module performs parallel operations on the input,
    including 1x1, 3x3, and double 3x3 convolutions, along with an
    average pooling branch, then concatenates the results along the
    channel dimension.

    Examples
    --------
    >>> import torch
    >>> from deepchem.models.torch_models.inceptionv3 import InceptionE
    >>> layer = InceptionE(1280)
    >>> x = torch.randn(5,1280, 8, 8)
    >>> output = layer(x)
    >>> output.shape
    torch.Size([5, 2048, 8, 8])
    """

    def __init__(
        self,
        in_channels: int,
    ) -> None:
        super(InceptionE, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels,
                                     320,
                                     kernel_size=(1, 1),
                                     stride=(1, 1),
                                     padding=(0, 0))

        self.branch3x3_1 = BasicConv2d(in_channels,
                                       384,
                                       kernel_size=(1, 1),
                                       stride=(1, 1),
                                       padding=(0, 0))
        self.branch3x3_2a = BasicConv2d(384,
                                        384,
                                        kernel_size=(1, 3),
                                        stride=(1, 1),
                                        padding=(0, 1))
        self.branch3x3_2b = BasicConv2d(384,
                                        384,
                                        kernel_size=(3, 1),
                                        stride=(1, 1),
                                        padding=(1, 0))

        self.branch3x3dbl_1 = BasicConv2d(in_channels,
                                          448,
                                          kernel_size=(1, 1),
                                          stride=(1, 1),
                                          padding=(0, 0))
        self.branch3x3dbl_2 = BasicConv2d(448,
                                          384,
                                          kernel_size=(3, 3),
                                          stride=(1, 1),
                                          padding=(1, 1))
        self.branch3x3dbl_3a = BasicConv2d(384,
                                           384,
                                           kernel_size=(1, 3),
                                           stride=(1, 1),
                                           padding=(0, 1))
        self.branch3x3dbl_3b = BasicConv2d(384,
                                           384,
                                           kernel_size=(3, 1),
                                           stride=(1, 1),
                                           padding=(1, 0))

        # self.avgpool = nn.AvgPool2d((3, 3), (1, 1), (1, 1))
        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the InceptionE module.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (N, in_channels, H, W).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (N, out_channels, H, W),
            where out_channels is the total number of output channels
            from all branches combined.
        """
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = torch.cat(
            [self.branch3x3_2a(branch3x3),
             self.branch3x3_2b(branch3x3)], 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = torch.cat([
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl)
        ], 1)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        out = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        output = torch.cat(out, 1)

        return output


class InceptionAux(nn.Module):
    """
    An auxiliary classifier module used in the InceptionV3 architecture.
    This module is intended to provide an auxiliary output for
    intermediate supervision during training.

    Examples
    --------
    >>> import torch
    >>> from deepchem.models.torch_models.inceptionv3 import InceptionAux
    >>> layer = InceptionAux(768, num_classes=3)
    >>> x = torch.randn(5, 768, 17, 17)
    >>> output = layer(x)
    >>> output.shape
    torch.Size([5, 3])
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        # self.avgpool1 = nn.AvgPool2d((5, 5), (3, 3))
        self.conv0 = BasicConv2d(in_channels,
                                 128,
                                 kernel_size=(1, 1),
                                 stride=(1, 1),
                                 padding=(0, 0))
        self.conv1 = BasicConv2d(128,
                                 768,
                                 kernel_size=(5, 5),
                                 stride=(1, 1),
                                 padding=(0, 0))
        self.conv1.stddev = 0.01  # type: ignore[assignment]
        # self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(768, num_classes)
        self.fc.stddev = 0.001  # type: ignore[assignment]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the InceptionAux module.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (N, in_channels, H, W).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (N, num_classes) representing class
            scores for each input.
        """
        # N x 768 x 17 x 17
        x = F.avg_pool2d(x, kernel_size=5, stride=3)
        # N x 768 x 5 x 5
        x = self.conv0(x)
        # N x 128 x 5 x 5
        x = self.conv1(x)
        # N x 768 x 1 x 1
        x = F.adaptive_avg_pool2d(x, (1, 1))
        # N x 768 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 768
        x = self.fc(x)
        # N x 3 (num_classes)

        return x


class InceptionV3Model(TorchModel):
    """
    Implementation of the InceptionV3 model architecture for image
    classification, modified for use with the DeepVariant framework
    in DeepChem.

    It builds on the original Inception design by utilizing a
    network-in-network approach, where convolutional filters of various
    sizes (e.g., 1x1, 3x3, 5x5) are applied in parallel within each module.
    This enables the model to capture features at multiple scales.
    InceptionV3 has factorized convolutions (breaking down larger
    convolutions into smaller ones, like 3x3 into 1x3 and 3x1)
    and the use of auxiliary classifiers that assist the model’s training
    by acting as regularizers. It uses dimensionality reduction to control
    the computational complexity.This model supports custom learning rate
    schedules with warmup and decay steps, utilizing the RMSProp optimizer.

    Examples
    --------
    >>> from deepchem.models.torch_models import InceptionV3Model
    >>> import deepchem as dc
    >>> import numpy as np
    >>> model = InceptionV3Model()
    >>> input_shape = (5, 6, 299, 299)
    >>> input_samples = np.random.randn(*input_shape).astype(np.float32)
    >>> output_samples = np.random.randint(0, 3, (5,)).astype(np.int64)
    >>> one_hot_output_samples = np.eye(3)[output_samples]
    >>> dataset = dc.data.ImageDataset(input_samples, one_hot_output_samples)
    >>> loss = model.fit(dataset, nb_epoch=1)
    >>> predictions = model.predict(dataset)
    >>> predictions.shape
    (5, 3)

    """

    def __init__(self,
                 in_channels=6,
                 warmup_steps=10000,
                 learning_rate=0.064,
                 dropout_rate=0.2,
                 decay_rate=0.94,
                 decay_steps=2,
                 rho=0.9,
                 momentum=0.9,
                 epsilon=1.0,
                 **kwargs):
        # weight_decay = 0.00004

        # Initialize the InceptionV3 model architecture
        model = InceptionV3(num_classes=3,
                            in_channels=in_channels,
                            aux_logits=False,
                            dropout_rate=dropout_rate)

        loss = CategoricalCrossEntropy()

        # Define optimizer as DeepChem's RMSProp
        optimizer = RMSProp(
            learning_rate=learning_rate,
            momentum=momentum,
            decay=rho,  # Using decay as rho
            epsilon=epsilon)

        # Custom attributes for learning rate decay and warmup
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.warmup_steps = warmup_steps
        self.current_step = 0

        # Initialize base TorchModel
        super(InceptionV3Model, self).__init__(model=model,
                                               optimizer=optimizer,
                                               loss=loss,
                                               **kwargs)

    def adjust_learning_rate(self):
        """
        Adjusts learning rate manually based on warmup and decay steps.
        """
        if self.current_step < self.warmup_steps:
            lr = self.learning_rate * (self.current_step / self.warmup_steps)
        else:
            decay_factor = self.decay_rate**(self.current_step //
                                             self.decay_steps)
            lr = self.learning_rate * decay_factor

        self.optimizer.learning_rate = lr

    def fit(self,
            dataset: Dataset,
            nb_epoch: int = 10,
            max_checkpoints_to_keep: int = 5,
            checkpoint_interval: int = 1000,
            deterministic: bool = False,
            restore: bool = False,
            variables: Optional[List[torch.nn.Parameter]] = None,
            loss: Optional[LossFn] = None,
            callbacks: Union[Callable, List[Callable]] = [],
            all_losses: Optional[List[float]] = None) -> float:
        """
        Trains the model on the given dataset, adjusting learning rate
        with warmup and decay.

        Parameters
        ----------
        dataset: Dataset
            Dataset to be used for training.
        nb_epoch: int, optional (default 1)
            Number of epochs to train the model.
        max_checkpoints_to_keep: int, optional
            Number of checkpoints to keep.
        checkpoint_interval: int, optional
            Interval for saving checkpoints.
        deterministic: bool, optional
            If True, runs in deterministic mode.
        restore: bool, optional
            If True, restores the model from the last checkpoint.
        variables: list, optional
            List of parameters to train.
        loss: callable, optional
            Custom loss function.
        callbacks: callable or list of callables, optional
            Callbacks to run during training.
        all_losses: list of floats, optional
            List to store all losses during training.

        Returns
        -------
        float
            The final loss value after training.
        """
        if all_losses is None:
            all_losses = []

        for epoch in range(nb_epoch):

            self.current_step = epoch
            self.adjust_learning_rate(
            )  # Adjust learning rate before each epoch

            epoch_loss = super(InceptionV3Model, self).fit(
                dataset,
                nb_epoch=1,
                max_checkpoints_to_keep=max_checkpoints_to_keep,
                checkpoint_interval=checkpoint_interval,
                deterministic=deterministic,
                restore=restore,
                variables=variables,
                loss=loss,
                callbacks=callbacks,
                all_losses=all_losses)
            all_losses.append(epoch_loss)

        return all_losses[-1] if all_losses else 0.0

    def save(self):
        """Saves model to disk using joblib."""
        save_to_disk(self.model, self.get_model_filename(self.model_dir))

    def reload(self):
        """Loads model from joblib file on disk."""
        self.model = load_from_disk(self.get_model_filename(self.model_dir))
