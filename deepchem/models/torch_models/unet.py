from deepchem.models.torch_models.modular import TorchModel
from deepchem.models.losses import BinaryCrossEntropy
import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    """
    UNet model for image segmentation.

    UNet is a convolutional neural network architecture for fast and precise segmentation of images
    based on the works of Ronneberger et al. [1]. The architecture consists of an encoder, a bottleneck,
    and a decoder. The encoder downsamples the input image to capture the context of the image. The
    bottleneck captures the most important features of the image. The decoder upsamples the image to
    generate the segmentation mask. The encoder and decoder are connected by skip connections to preserve
    spatial information.

    Examples
    --------
    Importing necessary modules

    >>> import numpy as np
    >>> import deepchem as dc
    >>> from deepchem.models.torch_models import UNet

    Creating a random dataset of 5 32x32 pixel RGB input images and 5 32x32 pixel grey scale output images

    >>> x = np.random.randn(5, 3, 32, 32).astype(np.float32)
    >>> y = np.random.rand(5, 1, 32, 32).astype(np.float32)
    >>> dataset = dc.data.NumpyDataset(x, y)

    We will create a UNet model with 3 input channels and 1 output channel. We will then fit the model on the dataset for 5 epochs and predict the output images.

    >>> model = UNetModel(in_channels=3, out_channels=1)
    >>> loss = model.fit(dataset, nb_epoch=5)
    >>> predictions = model.predict(dataset)

    Notes
    -----
    1. This implementation of the UNet model makes some changes to the padding of the inputs to the convolutional layers.
    The padding is set to 'same' to ensure that the output size of the convolutional layers is the same as the input size.
    This is done to preserve the spatial information of the input image and to keep the output size of the encoder and decoder the same.

    2. The input image size must be divisible by 2^4 = 16 to ensure that the output size of the encoder and decoder is the same.

    References
    ----------
    .. [1] Ronneberger, O., Fischer, P., & Brox, T. (2015, May 18). U-NET: Convolutional Networks for Biomedical Image Segmentation. arXiv.org. https://arxiv.org/abs/1505.04597
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 1):
        """
        Parameters
        ----------
        in_channels: int (default 3)
            Number of input channels.
        out_channels: int (default 1)
            Number of output channels.
        """
        super(UNet, self).__init__()

        # Encoder
        self.encoder1 = self.conv_block(in_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)

        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)

        # Decoder
        self.decoder4 = self.conv_block(1024 + 512, 512)
        self.decoder3 = self.conv_block(512 + 256, 256)
        self.decoder2 = self.conv_block(256 + 128, 128)
        self.decoder1 = self.conv_block(128 + 64, 64)

        # Maxpooling
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Upsampling
        self.upsample = nn.Upsample(scale_factor=2,
                                    mode='bilinear',
                                    align_corners=True)

        # Output
        self.output = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels: int, out_channels: int):
        """
        Parameters
        ----------
        in_channels: int
            Number of input channels.
        out_channels: int
            Number of output channels.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3,
                      padding='same'), nn.ReLU(inplace=True))

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x: Tensor
            Input tensor.

        Returns
        -------
        x: Tensor
            Output tensor.
        """

        # Encoder
        x1 = self.encoder1(x)
        x = self.maxpool(x1)
        x2 = self.encoder2(x)
        x = self.maxpool(x2)
        x3 = self.encoder3(x)
        x = self.maxpool(x3)
        x4 = self.encoder4(x)
        x = self.maxpool(x4)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        x = self.upsample(x)
        x = self.decoder4(torch.cat([x, x4], 1))
        x = self.upsample(x)
        x = self.decoder3(torch.cat([x, x3], 1))
        x = self.upsample(x)
        x = self.decoder2(torch.cat([x, x2], 1))
        x = self.upsample(x)
        x = self.decoder1(torch.cat([x, x1], 1))

        # Output
        x = self.output(x)
        x = F.sigmoid(x)
        return x


class UNetModel(TorchModel):
    """
    UNet model for image segmentation.

    UNet is a convolutional neural network architecture for fast and precise segmentation of images
    based on the works of Ronneberger et al. [1]. The architecture consists of an encoder, a bottleneck,
    and a decoder. The encoder downsamples the input image to capture the context of the image. The
    bottleneck captures the most important features of the image. The decoder upsamples the image to
    generate the segmentation mask. The encoder and decoder are connected by skip connections to preserve
    spatial information.

    Examples
    --------
    Importing necessary modules

    >>> import numpy as np
    >>> import deepchem as dc
    >>> from deepchem.models.torch_models import UNet

    Creating a random dataset of 5 32x32 pixel RGB input images and 5 32x32 pixel grey scale output images

    >>> x = np.random.randn(5, 3, 32, 32).astype(np.float32)
    >>> y = np.random.rand(5, 1, 32, 32).astype(np.float32)
    >>> dataset = dc.data.NumpyDataset(x, y)

    We will create a UNet model with 3 input channels and 1 output channel. We will then fit the model on the dataset for 5 epochs and predict the output images.

    >>> model = UNetModel(in_channels=3, out_channels=1)
    >>> loss = model.fit(dataset, nb_epoch=5)
    >>> predictions = model.predict(dataset)

    Notes
    -----
    1. This implementation of the UNet model makes some changes to the padding of the inputs to the convolutional layers.
    The padding is set to 'same' to ensure that the output size of the convolutional layers is the same as the input size.
    This is done to preserve the spatial information of the input image and to keep the output size of the encoder and decoder the same.

    2. The input image size must be divisible by 2^4 = 16 to ensure that the output size of the encoder and decoder is the same.

    References
    ----------
    .. [1] Ronneberger, O., Fischer, P., & Brox, T. (2015, May 18). U-NET: Convolutional Networks for Biomedical Image Segmentation. arXiv.org. https://arxiv.org/abs/1505.04597
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 1, **kwargs):
        """
        Parameters
        ----------
        input_channels: int (default 3)
            Number of input channels.
        output_channels: int (default 1)
            Number of output channels.
        """
        if in_channels <= 0:
            raise ValueError("input_channels must be greater than 0")

        if out_channels <= 0:
            raise ValueError("output_channels must be greater than 0")

        model = UNet(in_channels=in_channels, out_channels=out_channels)

        if 'loss' not in kwargs:
            kwargs['loss'] = BinaryCrossEntropy()

        super(UNetModel, self).__init__(model, **kwargs)
