from deepchem.models.torch_models.modular import TorchModel
from deepchem.models.losses import BinaryCrossEntropy
import torch
import torch.nn as nn


class UNet(nn.Module):
    
    def __init__(self,
                 in_channels: int = 3,
                 out_channels: int = 1):
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
        self.decoder4 = self.conv_block(1024+512, 512)
        self.decoder3 = self.conv_block(512+256, 256)
        self.decoder2 = self.conv_block(256+128, 128)
        self.decoder1 = self.conv_block(128+64, 64)

        # Maxpooling
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Upsampling
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Output
        self.output = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def conv_block(self, in_channels, out_channels):
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
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
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
        return x

class UNetModel(TorchModel):

    def __init__(self,
                 input_channels: int = 3,
                 output_channels: int = 1,
                 **kwargs):
        """
        Parameters
        ----------
        input_channels: int (default 3)
            Number of input channels.
        output_channels: int (default 1)
            Number of output channels.
        """
        if input_channels <= 0:
            raise ValueError("input_channels must be greater than 0")

        if output_channels <= 0:
            raise ValueError("output_channels must be greater than 0")

        model = UNet(input_channels=input_channels, output_channels=output_channels)

        if 'loss' not in kwargs:
            kwargs['loss'] = BinaryCrossEntropy()
        
        super(UNetModel, self).__init__(model, **kwargs)