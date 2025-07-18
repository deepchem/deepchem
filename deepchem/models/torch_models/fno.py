import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from deepchem.models.torch_models.layers import SpectralConv
from typing import Union, Tuple, Optional, List
from deepchem.utils.fno_utils import GaussianNormalizer
from deepchem.models.torch_models.torch_model import TorchModel


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
                 positional_encoding: bool = False,
                 normalize_input: bool = True,
                 normalize_output: bool = True,
                 normalization_dims: Optional[List[int]] = None) -> None:
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
        normalize_input: bool, default True
            When enabled, normalizes input data
        normalize_output: bool, default True
            When enabled, normalizes output data
        normalization_dims: List[int], optional
            Dimensions to normalize over. If None, defaults to batch + spatial dimensions, preserving channels.
        """
        super().__init__()
        self.dims = dims
        self.in_channels = in_channels + dims if positional_encoding else in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.width = width
        self.dims = dims
        self.depth = depth
        self.positional_encoding = positional_encoding
        self.normalize_input = normalize_input
        self.normalize_output = normalize_output

        # Default normalization dimensions: batch + spatial dimensions, preserve channels
        # for channels-first format (batch, channels, *spatial_dims)
        if normalization_dims is None:
            self.normalization_dims = [0] + list(range(2, dims + 2))
        else:
            self.normalization_dims = normalization_dims

        self.input_normalizer = GaussianNormalizer(
            dim=self.normalization_dims) if normalize_input else None
        self.output_normalizer = GaussianNormalizer(
            dim=self.normalization_dims) if normalize_output else None

        self.lifting = nn.Sequential(nn.Linear(self.in_channels, 2 * width),
                                     nn.GELU(), nn.Linear(2 * width, width))

        self.fno_blocks = nn.Sequential(
            *[FNOBlock(self.width, self.modes, dims=self.dims) for _ in range(self.depth)])

        self.projection = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(),
                                        nn.Linear(2 * width, out_channels))

    def fit_normalizers(self,
                        x_train: torch.Tensor,
                        y_train: Optional[torch.Tensor] = None,
                        device: Optional[torch.device] = None) -> None:
        """Fit normalizers on training data. Called by the fit method of the FNOModel class."""
        x_train = self._ensure_channel_first(x_train)
        if y_train is not None:
            y_train = self._ensure_channel_first(y_train, is_output=True)

        if self.normalize_input and self.input_normalizer:
            self.input_normalizer.fit(x_train)
            if device is not None:
                self.input_normalizer.to(device)
        if self.normalize_output and self.output_normalizer and y_train is not None:
            self.output_normalizer.fit(y_train)
            if device is not None:
                self.output_normalizer.to(device)

    def _ensure_channel_first(self, x: torch.Tensor, is_output: bool = False) -> torch.Tensor:
        """Ensure input tensor has channels in the correct position."""
        if is_output:
            ch = self.out_channels
        else:
            ch = self.in_channels - self.dims if self.positional_encoding else self.in_channels
        
        if x.shape[-1] == ch:
            perm = (0, -1) + tuple(range(1, self.dims + 1))
            return x.permute(*perm).contiguous()
        elif x.shape[1] == ch:
            return x
        else:
            raise ValueError(
                f"Expected either (batch, {ch}, *spatial_dims) or "
                f"(batch, *spatial_dims, {ch}), got {tuple(x.shape)}")

    def _generate_meshgrid(self, x: torch.Tensor) -> torch.Tensor:
        """Generate meshgrid of coordinates for positional encoding."""
        batch_size = x.shape[0]
        spatial = x.shape[2:]
        coords = [
            torch.linspace(0, 1, steps=s, device=x.device) for s in spatial
        ]
        mesh = torch.meshgrid(*coords, indexing='ij')
        grid = torch.stack(mesh, dim=0)  # shape (dims, *spatial)
        grid = grid.unsqueeze(0).repeat(batch_size, 1, *([1] * self.dims))
        return grid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != self.dims + 2:
            raise ValueError(
                f"Expected tensor with {self.dims + 2} dims (batch, *spatial_dims, in_channels), got {x.ndim}"
            )

        x = self._ensure_channel_first(x)

        if self.positional_encoding:
            grid = self._generate_meshgrid(x)
            x = torch.cat([x, grid], dim=1)

        if self.normalize_input and self.input_normalizer and self.input_normalizer.fitted:
            if self.input_normalizer.mean is not None and self.input_normalizer.mean.device != x.device:
                self.input_normalizer.to(x.device)
            x = self.input_normalizer.transform(x)

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

        if self.normalize_output and self.output_normalizer and self.output_normalizer.fitted:
            if self.output_normalizer.mean is not None and self.output_normalizer.mean.device != x.device:
                self.output_normalizer.to(x.device)

            # Permute to channels-first for normalization
            x = x.permute(0, -1, *range(1, x.ndim - 1)).contiguous()

            if not self.training:
                # During inference, denormalize the model output to original scale
                x = self.output_normalizer.inverse_transform(x)
            # During training, keep the model output as-is (it should be in normalized space)
        else:
            # Permute to channels-first for label compatibility if not normalizing
            x = x.permute(0, -1, *range(1, x.ndim - 1)).contiguous()

        return x

    def state_dict(self):
        """Override state_dict to include normalizer statistics."""
        state = super().state_dict()

        if self.input_normalizer:
            state['input_normalizer_mean'] = self.input_normalizer.mean
            state['input_normalizer_std'] = self.input_normalizer.std
            state['input_normalizer_fitted'] = self.input_normalizer.fitted

        if self.output_normalizer:
            state['output_normalizer_mean'] = self.output_normalizer.mean
            state['output_normalizer_std'] = self.output_normalizer.std
            state['output_normalizer_fitted'] = self.output_normalizer.fitted

        return state

    def load_state_dict(self, state_dict, strict=True):
        """Override load_state_dict to restore normalizer statistics."""
        input_normalizer_mean = state_dict.pop('input_normalizer_mean', None)
        input_normalizer_std = state_dict.pop('input_normalizer_std', None)
        input_normalizer_fitted = state_dict.pop('input_normalizer_fitted',
                                                 False)

        output_normalizer_mean = state_dict.pop('output_normalizer_mean', None)
        output_normalizer_std = state_dict.pop('output_normalizer_std', None)
        output_normalizer_fitted = state_dict.pop('output_normalizer_fitted',
                                                  False)

        super().load_state_dict(state_dict, strict)

        if self.input_normalizer and input_normalizer_mean is not None:
            self.input_normalizer.mean = input_normalizer_mean
            self.input_normalizer.std = input_normalizer_std
            self.input_normalizer.fitted = input_normalizer_fitted

        if self.output_normalizer and output_normalizer_mean is not None:
            self.output_normalizer.mean = output_normalizer_mean
            self.output_normalizer.std = output_normalizer_std
            self.output_normalizer.fitted = output_normalizer_fitted


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
    >>> model.fit(dataset)
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
                 normalize_input: bool = True,
                 normalize_output: bool = True,
                 normalization_dims: Optional[List[int]] = None,
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
        normalize_input: bool, default True
            When enabled, normalizes input data
        normalize_output: bool, default True
            When enabled, normalizes output data
        normalization_dims: List[int], optional
            Dimensions to normalize over. If None, defaults to batch + spatial dimensions, preserving channels.
        **kwargs: dict
            Additional arguments passed to TorchModel constructor
        """

        model = FNO(in_channels, out_channels, modes, width, dims, depth,
                    positional_encoding, normalize_input, normalize_output,
                    normalization_dims)

        self._normalize_input = normalize_input
        self._normalize_output = normalize_output
        self._normalizers_fitted = False

        super(FNOModel, self).__init__(model=model,
                                       loss=self._loss_fn,
                                       **kwargs)

    def fit(self, dataset, nb_epoch=1, **kwargs):
        """Fit the model with automatic normalizer fitting."""
        if (self._normalize_input or
                self._normalize_output) and not self._normalizers_fitted:
            self._fit_normalizers_from_dataset(dataset)
            self._normalizers_fitted = True

        return super().fit(dataset, nb_epoch, **kwargs)

    def _fit_normalizers_from_dataset(self, dataset):
        """Extract data from dataset and fit normalizers."""

        X_all = torch.tensor(dataset.X, dtype=torch.float32)
        y_all = torch.tensor(dataset.y, dtype=torch.float32)
        self.model.fit_normalizers(X_all, y_all, device=self.device)

    def _loss_fn(self,
                 outputs: List[torch.Tensor],
                 labels: List[torch.Tensor],
                 weights: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        """Compute the loss for training."""
        labels_tensor: torch.Tensor = labels[0]
        outputs_tensor: torch.Tensor = outputs[0]

        if self._normalizers_fitted and self.model.output_normalizer is not None:
            # Ensure labels are channels-first before transforming
            labels_tensor = self.model._ensure_channel_first(
                labels_tensor, is_output=True)
            labels_tensor = self.model.output_normalizer.transform(labels_tensor)

        loss = nn.MSELoss()(outputs_tensor, labels_tensor)
        return loss
