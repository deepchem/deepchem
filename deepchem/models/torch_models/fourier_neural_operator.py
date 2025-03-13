import torch
import torch.nn as nn
import torch.fft

##########################################################################
# Generic Spectral Convolution Layer (for 1D, 2D, 3D)
##########################################################################

class SpectralConv(nn.Module):
    """
    n-Dimensional Fourier layer.

    It applies an n-dimensional FFT on the spatial dimensions,
    keeps only a specified number of Fourier modes (for each spatial dimension),
    applies a learned complex multiplication, and returns to physical space
    via the inverse FFT.
    """
    def __init__(self, in_channels, out_channels, modes, dims=2):
        """
        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
            modes: either an int (same number of modes in every dimension)
                   or a tuple of ints (number of modes per spatial dimension).
            dims: number of spatial dimensions (typically 1, 2, or 3).
        """
        super(SpectralConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dims = dims

        # Ensure modes is a tuple with length equal to dims.
        if isinstance(modes, int):
            self.modes = (modes,) * dims
        elif isinstance(modes, (tuple, list)):
            if len(modes) != dims:
                raise ValueError("Length of modes must equal dims.")
            self.modes = tuple(modes)
        else:
            raise ValueError("modes must be int or tuple/list of ints.")

        # The weight parameter is a learnable tensor for the low-frequency modes.
        weight_shape = (in_channels, out_channels) + self.modes
        self.scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            self.scale * torch.rand(*weight_shape, dtype=torch.cfloat)
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, in_channels, *spatial_dims).
        Returns:
            Output tensor of shape (batch, out_channels, *spatial_dims).
        """
        batchsize = x.shape[0]
        # Apply n-dimensional real FFT on the last dims (the spatial dimensions).
        # We assume that the spatial dimensions are the ones after the first two.
        x_ft = torch.fft.rfftn(x, dim=range(2, x.ndim))
        # Prepare an output tensor in Fourier space with zeros.
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            *x_ft.shape[2:],
            dtype=torch.cfloat,
            device=x.device
        )

        # Build a tuple of slices to select the low-frequency modes.
        slices = tuple(slice(0, m) for m in self.modes)
        # Multiply the selected Fourier modes with the learned weights.
        # x_ft[:, :, slices] has shape (batch, in_channels, *self.modes)
        # self.weights has shape (in_channels, out_channels, *self.modes)
        # The einsum performs a summation over the in_channels:
        out_ft[(slice(None), slice(None)) + slices] = torch.einsum(
            "b i ... , i o ... -> b o ...", x_ft[(slice(None), slice(None)) + slices], self.weights
        )

        # Inverse FFT to return to physical space.
        # We provide the original spatial shape (which is x.shape[2:]) as s.
        x_out = torch.fft.irfftn(out_ft, s=x.shape[2:], dim=range(2, x.ndim))
        return x_out

##########################################################################
# Helper: Get a pointwise convolution layer based on the spatial dimension.
##########################################################################

def get_pointwise_conv(dims, in_channels, out_channels, kernel_size=1):
    if dims == 1:
        return nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size)
    elif dims == 2:
        return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size)
    elif dims == 3:
        return nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size)
    else:
        raise ValueError("Only dims=1,2,3 are supported for the pointwise conv.")

##########################################################################
# Dynamic FNO Module (for 1D, 2D, or 3D)
##########################################################################

class FNO(nn.Module):
    """
    Dynamic Fourier Neural Operator.

    This network assumes inputs defined on a regular grid with an additional
    feature channel (e.g. a scalar field). It first concatenates coordinate
    information, lifts the input to a higher-dimensional representation,
    applies several Fourier layers (each combined with a pointwise conv),
    and then projects back to the target output.
    """
    def __init__(self, dims, modes, width, num_layers=4, input_channels=1, output_channels=1):
        """
        Args:
            dims: Number of spatial dimensions (1, 2, or 3).
            modes: Number of Fourier modes to keep (int or tuple of ints, length=dims).
            width: Width of the intermediate representations (number of channels).
            num_layers: Number of Fourier layers.
            input_channels: Number of input channels (not counting coordinate channels).
            output_channels: Number of output channels.
        """
        super(FNO, self).__init__()
        self.dims = dims
        self.width = width
        self.num_layers = num_layers
        self.input_channels = input_channels
        self.output_channels = output_channels

        # The lifting layer: It maps from (input_channels + dims) to width.
        # (We add dims coordinate channels.)
        self.fc0 = nn.Linear(input_channels + dims, width)

        # Create a list of Fourier layers and pointwise convolution layers.
        self.spectral_layers = nn.ModuleList()
        self.pointwise_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.spectral_layers.append(SpectralConv(width, width, modes, dims))
            self.pointwise_layers.append(get_pointwise_conv(dims, width, width, kernel_size=1))

        # Projection layers: applied on the last channel at every spatial location.
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, output_channels)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, *spatial_dims, input_channels)
               For example, for a 2D problem: (batch, height, width, 1).
        Returns:
            Output tensor of shape (batch, *spatial_dims, output_channels)
        """
        batch_size = x.shape[0]
        spatial_shape = x.shape[1:-1]  # spatial dimensions

        # Coordinate grids for each spatial dimension in [0, 1].
        grids = torch.meshgrid(
            [torch.linspace(0, 1, steps=n, device=x.device) for n in spatial_shape],
            indexing='ij'
        )
        grid = torch.stack(grids, dim=-1)
        grid = grid.unsqueeze(0).expand(batch_size, *spatial_shape, self.dims)
        x = torch.cat([x, grid], dim=-1)

        x = self.fc0(x)  # shape: (batch, *spatial_dims, width)
        x = x.permute(0, -1, *range(1, x.ndim - 1))

        # Applying the Fourier layers.
        for spectral_conv, pointwise_conv in zip(self.spectral_layers, self.pointwise_layers):
            x = spectral_conv(x) + pointwise_conv(x)
            x = torch.relu(x)

        x = x.permute(0, *range(2, x.ndim), 1)

        # Applying the projection layers pointwise.
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x

##########################################################################
# Example Usage
##########################################################################

if __name__ == '__main__':
    # Example 1: 1D FNO.
    batch_size = 4
    length = 64  # spatial length
    input_channels = 1
    # Input shape for 1D: (batch, length, channels)
    x1d = torch.randn(batch_size, length, input_channels)
    # Create a 1D FNO: dims=1, modes=16 (for example), width=32.
    model_1d = FNO(dims=1, modes=16, width=32, input_channels=input_channels, output_channels=1)
    out1d = model_1d(x1d)
    print("1D input shape :", x1d.shape)
    print("1D output shape:", out1d.shape)

    # Example 2: 2D FNO.
    height, width = 64, 64
    input_channels = 1
    # Input shape for 2D: (batch, height, width, channels)
    x2d = torch.randn(batch_size, height, width, input_channels)
    # Create a 2D FNO: dims=2, modes=12 (or tuple like (12, 12)), width=32.
    model_2d = FNO(dims=2, modes=(12, 12), width=32, input_channels=input_channels, output_channels=1)
    out2d = model_2d(x2d)
    print("2D input shape :", x2d.shape)
    print("2D output shape:", out2d.shape)

    # Example 3: 3D FNO.
    D, H, W = 16, 32, 32
    input_channels = 1
    # Input shape for 3D: (batch, D, H, W, channels)
    x3d = torch.randn(batch_size, D, H, W, input_channels)
    # Create a 3D FNO: dims=3, modes=(8,8,8) for example, width=32.
    model_3d = FNO(dims=3, modes=(8, 8, 8), width=32, input_channels=input_channels, output_channels=1)
    out3d = model_3d(x3d)
    print("3D input shape :", x3d.shape)
    print("3D output shape:", out3d.shape)
