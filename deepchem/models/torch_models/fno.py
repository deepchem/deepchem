import torch
import torch.nn as nn
import torch.nn.functional as F
from deepchem.models.torch_models.layers import SpectralConv
from deepchem.models.torch_models.torch_model import TorchModel
from typing import Union, Tuple, Optional, List


class FNOBlock(nn.Module):
    """A single Fourier Neural Operator block.

    This block combines spectral convolution in Fourier space with a standard
    convolution to learn both global and local features.

    Spectral convolution is a key component of Fourier Neural Operators (FNOs).
    It leverages the Fourier transform to perform convolution in the frequency
    domain, which allows it to capture global, long-range dependencies in the
    input data efficiently. The operation consists of three steps:
    1. Transform the input to the frequency domain using the Fast Fourier Transform (FFT).
    2. Apply a linear transformation to a truncated set of lower-frequency modes.
    3. Transform the result back to the spatial domain using the Inverse FFT.

    By operating in the frequency domain, spectral convolutions can learn global
    patterns without the large kernels and deep architectures required by
    traditional CNNs, in contrast to the local convolutions used in CNNs.

    This is because each Fourier mode represents a sinusoidal function that
    spans the entire spatial domain (meaning the entire input). Their coefficients in the frequency
    domain contain information about the overall structure of the input. By manipulating the
    coefficients of these modes in the frequency domain, the spectral convolution can model
    relationships between distant points in the input, effectively capturing global dependencies.

    The forward pass computes:
    FNO_block(x) = ReLU(SpectralConv(x) + Conv(x))

    Example
    -------------
    >>> import torch
    >>> from deepchem.models.torch_models.fno import FNOBlock
    >>> block = FNOBlock(width=128, modes=8, dims=2)
    >>> x = torch.randn(1, 128, 16, 16)
    >>> output = block(x)
    """

    def __init__(self, width: int, modes: Union[int, Tuple[int, ...]],
                 dims: int) -> None:
        """Initialize the FNO block.

        Parameters
        ----------
        width: int
            Number of channels/features in the block
        modes: int or tuple
            Number of Fourier modes to keep in spectral convolution
        dims: int
            Spatial dimensionality (1, 2, or 3)
        """
        super().__init__()
        self.spectral_conv = SpectralConv(width, width, modes, dims=dims)
        self.w: Union[nn.Conv1d, nn.Conv2d, nn.Conv3d]

        if dims == 1:
            self.w = nn.Conv1d(width, width, 1)
        elif dims == 2:
            self.w = nn.Conv2d(width, width, 1)
        elif dims == 3:
            self.w = nn.Conv3d(width, width, 1)
        else:
            raise ValueError(f"Invalid dimension: {dims}. Must be 1, 2, or 3.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the FNO block.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape (batch, width, *spatial_dims)

        Returns
        -------
        torch.Tensor
            Output tensor of same shape as input after spectral and local convolution
        """
        x1 = self.spectral_conv(x)
        x2 = self.w(x)
        return F.relu(x1 + x2)


class FNO(nn.Module):
    """Base implementation of Fourier Neural Operator, inheriting from the Torch nn.Module class.

    Fourier Neural Operator (FNO) is a neural network architecture for learning
    mappings between function spaces. It uses spectral convolutions in Fourier
    space to capture global dependencies efficiently, making it particularly
    effective for solving partial differential equations (PDEs).

    The architecture consists of:
    1. Lifting layer: Maps input to higher-dimensional representation
    2. Multiple FNO blocks: Perform spectral and local convolutions
    3. Projection layers: Map back to output space

    References
    ----------
    This technique was introduced in Li, Zongyi, et al. "Fourier neural operator for parametric partial differential equations." arXiv preprint arXiv:2010.08895 (2020).

    Example
    -------------
    >>> import torch
    >>> from deepchem.models.torch_models.fno import FNO
    >>> model = FNO(in_channels=1, out_channels=1, modes=8, width=32, dims=2)
    >>> x = torch.randn(1, 16, 16, 1)
    >>> output = model(x)
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 modes: Union[int, Tuple[int, ...]],
                 width: int,
                 dims: int,
                 depth: int = 4,
                 positional_encoding: bool = False) -> None:
        """Initialize the FNO base model.

        Parameters
        ----------
        in_channels: int
            Dimension of input features
        out_channels: int
            Dimension of output features
        modes: int or tuple
            Number of Fourier modes to keep in spectral convolution
        width: int
            Width of the hidden layers
        dims: int
            Spatial dimensionality (1, 2, or 3)
        depth: int, default 4
            Number of FNO blocks to stack
        positional_encoding: bool, default False
            When enabled, uses meshgrids as positional encodings. If custom positional encodings, must be set to False.
        """
        super().__init__()
        self.dims = dims
        self.in_channels = in_channels + dims if positional_encoding else in_channels
        self.out_channels = out_channels
        self.width = width
        self.positional_encoding = positional_encoding

        self.lifting = nn.Sequential(nn.Linear(self.in_channels, 2 * width),
                                     nn.GELU(), nn.Linear(2 * width, width))

        self.fno_blocks = nn.Sequential(
            *[FNOBlock(width, modes, dims=dims) for _ in range(depth)])

        self.projection = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(),
                                        nn.Linear(2 * width, out_channels))

    def _ensure_channel_first(self, x: torch.Tensor) -> torch.Tensor:
        """Ensure input tensor has channels in the correct position.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
            Input tensor with channels in the correct position
        """
        in_ch = self.in_channels - self.dims if self.positional_encoding else self.in_channels
        if x.shape[-1] == in_ch:
            perm = (0, -1) + tuple(range(1, self.dims + 1))
            return x.permute(*perm).contiguous()
        elif x.shape[1] == in_ch:
            return x
        else:
            raise ValueError(
                f"Expected either (batch, {in_ch}, *spatial_dims) or "
                f"(batch, *spatial_dims, {in_ch}), got {tuple(x.shape)}")

    def _generate_meshgrid(self, x: torch.Tensor) -> torch.Tensor:
        """Generate meshgrid of coordinates for positional encoding.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
            Meshgrid of coordinates for positional encoding
        """
        batch_size = x.shape[0]
        spatial = x.shape[2:]
        coords = [
            torch.linspace(0, 1, steps=s, device=x.device) for s in spatial
        ]
        mesh = torch.meshgrid(*coords, indexing='ij')
        grid = torch.stack(mesh, dim=0)  # shape (dims, *spatial)
        # Repeat for batch dimension and add spatial dimensions
        grid = grid.unsqueeze(0).expand(batch_size, -1, *spatial)
        return grid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the FNO model.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
            Output tensor
        """
        if x.ndim != self.dims + 2:
            raise ValueError(
                f"Expected tensor with {self.dims + 2} dims (batch, *spatial_dims, in_channels), got {x.ndim}"
            )

        x = self._ensure_channel_first(x)

        if self.positional_encoding:
            grid = self._generate_meshgrid(x)
            x = torch.cat([x, grid], dim=1)

        # Reshape for lifting layer: (batch, *spatial, channels) -> (batch * spatial, channels)
        x = x.permute(0, *range(2, x.ndim), 1).contiguous()
        x = self.lifting(x)

        # FNO blocks: permute channels to second dim
        perm_back = (0, x.ndim - 1) + tuple(range(1, x.ndim - 1))
        x = x.permute(*perm_back).contiguous()
        x = self.fno_blocks(x)

        # Reshape for projection layer: (batch, *spatial, channels) -> (batch * spatial, channels)
        x = x.permute(0, *range(2, x.ndim), 1).contiguous()
        x = self.projection(x)

        # Permute to channels-first for output
        x = x.permute(0, -1, *range(1, x.ndim - 1)).contiguous()

        return x


class FNOModel(TorchModel):
    """Fourier Neural Operator for learning mappings between function spaces.

    This is a TorchModel wrapper around the nn.Module FNO class that provides the DeepChem
    interface for training and prediction. FNO is particularly effective for
    solving partial differential equations (PDEs) and learning operators
    between infinite-dimensional function spaces.

    The model uses spectral convolutions in Fourier space to capture global
    dependencies efficiently, making it much more parameter-efficient than
    traditional convolutional neural networks for PDE solving tasks.

    References
    ----------
    This technique was introduced in Li, Zongyi, et al. "Fourier neural operator for parametric partial differential equations." arXiv preprint arXiv:2010.08895 (2020).

    Example
    -------------
    >>> import torch
    >>> import deepchem as dc
    >>> from deepchem.models.torch_models.fno import FNOModel
    >>> x = torch.randn(1, 16, 16, 1)
    >>> dataset = dc.data.NumpyDataset(X=x, y=x)
    >>> model = FNOModel(in_channels=1, out_channels=1, modes=8, width=32, dims=2)
    >>> loss = model.fit(dataset)
    >>> predictions = model.predict(dataset)
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 modes: Union[int, Tuple[int, ...]],
                 width: int,
                 dims: int,
                 depth: int = 4,
                 positional_encoding: bool = False,
                 loss: nn.Module = nn.MSELoss(),
                 **kwargs) -> None:
        """Initialize the FNO model.
        Parameters
        ----------
        in_channels: int
            Dimension of input features at each spatial location
        out_channels: int
            Dimension of output features at each spatial location
        modes: int or tuple
            Number of Fourier modes to keep in spectral convolution. Higher values
            capture more high-frequency information but increase computational cost
        width: int
            Width of the hidden layers in the FNO blocks. Controls model capacity
        dims: int
            Spatial dimensionality of the input data (1, 2, or 3)
        depth: int, default 4
            Number of FNO blocks to stack. More blocks can learn more complex mappings
        positional_encoding: bool, default False
            When enabled, uses meshgrids as positional encodings
        loss: Union[Loss, LossFn], default nn.MSELoss()
            Loss function to use for training
        **kwargs: dict
            Additional arguments passed to TorchModel constructor
        """
        self.loss_fn = loss
        model = FNO(in_channels, out_channels, modes, width, dims, depth,
                    positional_encoding)

        super(FNOModel, self).__init__(model=model,
                                       loss=self._loss_fn,
                                       **kwargs)

    def _loss_fn(self,
                 outputs: List[torch.Tensor],
                 labels: List[torch.Tensor],
                 weights: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        """Overrides the default loss function for training.

        Computes the loss for training.

        Parameters
        ----------
        outputs: List[torch.Tensor]
            List of output tensors from the model
        labels: List[torch.Tensor]
            List of label tensors
        weights: Optional[List[torch.Tensor]], default None
            List of weight tensors
        """
        labels_tensor: torch.Tensor = labels[0]
        outputs_tensor: torch.Tensor = outputs[0]
        loss = self.loss_fn(labels_tensor, outputs_tensor)
        return loss
