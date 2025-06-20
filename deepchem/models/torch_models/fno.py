from deepchem.models.torch_models.layers import SpectralConv
from deepchem.models.torch_models import TorchModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple, Optional, List


class FNOBlock(nn.Module):
    """A single Fourier Neural Operator block.

    This block combines spectral convolution in Fourier space with a standard
    convolution to learn both global and local features. The spectral convolution
    operates on the Fourier coefficients of the input, while the standard convolution
    provides a residual connection.

    The forward pass computes:
    FNO_block(x) = ReLU(SpectralConv(x) + Conv(x))

    Usage Example
    -------------
    >>> import torch
    >>> from deepchem.models.torch_models.fno import FNOBlock
    >>> block = FNOBlock(width=128, modes=10, dims=2)
    >>> x = torch.randn(1, 128, 10, 10)
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
        if dims == 1:
            self.w = nn.Conv1d(width, width, 1)
        elif dims == 2:
            self.w = nn.Conv2d(width, width, 1)
        elif dims == 3:
            self.w = nn.Conv3d(width, width, 1)
        else:
            raise NotImplementedError(f"Invalid dimension: {dims}")

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


class FNOBase(nn.Module):
    """Base implementation of Fourier Neural Operator.

    Fourier Neural Operator (FNO) is a neural network architecture for learning
    mappings between function spaces. It uses spectral convolutions in Fourier
    space to capture global dependencies efficiently, making it particularly
    effective for solving partial differential equations (PDEs).

    The architecture consists of:
    1. Lifting layer (fc0): Maps input to higher-dimensional representation
    2. Multiple FNO blocks: Perform spectral and local convolutions
    3. Projection layers (fc1, fc2): Map back to output space

    References
    ----------
    This technique was introduced in [1]_

    .. [1] Li, Zongyi, et al. "Fourier neural operator for parametric partial differential equations." arXiv preprint arXiv:2010.08895 (2020).

    Usage Example
    -------------
    >>> import torch
    >>> from deepchem.models.torch_models.fno import FNOBase
    >>> model = FNOBase(input_dim=1, output_dim=1, modes=8, width=32, dims=2)
    >>> x = torch.randn(1, 16, 16, 1)
    >>> output = model(x)
    """

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 modes: Union[int, Tuple[int, ...]],
                 width: int,
                 dims: int,
                 depth: int = 4) -> None:
        """Initialize the FNO base model.

        Parameters
        ----------
        input_dim: int
            Dimension of input features
        output_dim: int
            Dimension of output features
        modes: int or tuple
            Number of Fourier modes to keep in spectral convolution
        width: int
            Width of the hidden layers
        dims: int
            Spatial dimensionality (1, 2, or 3)
        depth: int, default 4
            Number of FNO blocks to stack
        """
        super().__init__()
        self.dims = dims
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.width = width

        self.fc0 = nn.Linear(input_dim, width)
        self.fno_blocks = nn.Sequential(
            *[FNOBlock(width, modes, dims=dims) for _ in range(depth)])
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def _ensure_channel_first(self, x: torch.Tensor) -> torch.Tensor:
        """Ensure input tensor has channels in the correct position.

        Converts between (batch, *spatial_dims, input_dim) and
        (batch, input_dim, *spatial_dims) formats.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
            Tensor with channels in position 1: (batch, input_dim, *spatial_dims)
        """
        if x.shape[-1] == self.input_dim:
            perm_dims = range(1, self.dims + 1)
            return x.permute(0, -1, *perm_dims).contiguous()
        elif x.shape[1] == self.input_dim:
            return x
        else:
            raise ValueError(
                f"Expected either (batch, {self.input_dim}, *spatial_dims) or "
                f"(batch, *spatial_dims, {self.input_dim}), got {tuple(x.shape)}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the FNO model.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape (batch, *spatial_dims, input_dim)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, *spatial_dims, output_dim)
        """
        if x.ndim != self.dims + 2:
            raise ValueError(
                f"Expected tensor with {self.dims + 2} dims (batch, *spatial_dims, input_dim), got {x.ndim}"
            )

        x = self._ensure_channel_first(x)
        print("After _ensure_channel_first", x.shape)
        # Need to permute the channels to the last dimension for the linear layer fc0
        perm_fc0 = (0, *range(2, x.ndim), 1)
        x = x.permute(*perm_fc0).contiguous()
        x = self.fc0(x)

        # Need to permute the channels back to the first dimension (excluding batch dimension) for FNO blocks.
        # This is because they expect the spatial dimensions to be last.
        perm_back = (0, x.ndim - 1) + tuple(range(1, x.ndim - 1))
        x = x.permute(*perm_back).contiguous()
        x = self.fno_blocks(x)

        # Need to permute the channels back to the last dimension again for the linear layer fc1
        perm_proj = (0, *range(2, x.ndim), 1)
        x = x.permute(*perm_proj).contiguous()
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x


class FNOModel(TorchModel):
    """Fourier Neural Operator for learning mappings between function spaces.

    This is a TorchModel wrapper around FNOBase that provides the DeepChem
    interface for training and prediction. FNO is particularly effective for
    solving partial differential equations (PDEs) and learning operators
    between infinite-dimensional function spaces.

    The model uses spectral convolutions in Fourier space to capture global
    dependencies efficiently, making it much more parameter-efficient than
    traditional convolutional neural networks for PDE solving tasks.

    References
    ----------
    This technique was introduced in [1]_

    .. [1] Li, Zongyi, et al. "Fourier neural operator for parametric partial differential equations." arXiv preprint arXiv:2010.08895 (2020).

    Usage Example
    -------------
    >>> import torch
    >>> import deepchem as dc
    >>> from deepchem.models.torch_models.fno import FNOModel
    >>> x = torch.randn(1, 16, 16, 1)
    >>> dataset = dc.data.NumpyDataset(X=x, y=x)
    >>> model = FNOModel(input_dim=1, output_dim=1, modes=8, width=32, dims=2)
    >>> model.fit(dataset)
    >>> predictions = model.predict(dataset)
    """

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 modes: Union[int, Tuple[int, ...]],
                 width: int,
                 dims: int,
                 depth: int = 4,
                 **kwargs) -> None:
        """Initialize the FNO model.

        Parameters
        ----------
        input_dim: int
            Dimension of input features at each spatial location
        output_dim: int
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
        **kwargs: dict
            Additional arguments passed to TorchModel constructor
        """
        model = FNOBase(input_dim, output_dim, modes, width, dims, depth)
        super(FNOModel, self).__init__(model=model,
                                       loss=self._loss_fn,
                                       **kwargs)

    def _loss_fn(self,
                 outputs: List[torch.Tensor],
                 labels: List[torch.Tensor],
                 weights: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        """Compute the loss for training.

        Parameters
        ----------
        outputs: List[torch.Tensor]
            Model predictions
        labels: List[torch.Tensor]
            Ground truth labels
        weights: torch.Tensor, optional
            Sample weights (currently unused)

        Returns
        -------
        torch.Tensor
            Mean squared error loss between predictions and labels
        """
        labels = labels[0]
        outputs = outputs[0]
        return nn.MSELoss()(outputs, labels)
