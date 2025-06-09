from deepchem.models.torch_models.layers import SpectralConv
from deepchem.models.torch_models import TorchModel
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F


class FNOBlock(nn.Module):
    def __init__(self, width, modes, dims):
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

    def forward(self, x):
        x1 = self.spectral_conv(x)
        x2 = self.w(x)
        return F.relu(x1 + x2)

class FNOBase(nn.Module):
    def __init__(self, input_dim, output_dim, modes, width, dims, depth=4):
        super().__init__()
        self.dims = dims
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.width = width

        self.fc0 = nn.Linear(input_dim, width)
        self.fno_blocks = nn.Sequential(
            *[FNOBlock(width, modes, dims=dims) for _ in range(depth)]
        )
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def _ensure_channel_first(self, x):
        # If x is (batch, *spatial_dims, input_dim) -> permute -> (batch, input_dim, *spatial_dims)
        if x.shape[-1] == self.input_dim:
            perm = (0, x.ndim - 1) + tuple(range(1, x.ndim - 1))
            return x.permute(*perm).contiguous()
        # If x is already (batch, input_dim, *spatial_dims), leave it
        elif x.shape[1] == self.input_dim:
            return x
        else:
            raise ValueError(
                f"Expected either (batch, input_dim, *spatial_dims) or "
                f"(batch, *spatial_dims, input_dim), got {tuple(x.shape)}"
            )

    def forward(self, x):
        if x.ndim != self.dims + 2:
            raise ValueError(
                f"Expected tensor with {self.dims+2} dims (batch, *spatial_dims, input_dim), got {x.ndim}"
            )

        x = self._ensure_channel_first(x)

        perm_fc0 = (0, *range(2, x.ndim), 1)
        x = x.permute(*perm_fc0).contiguous()
        x = self.fc0(x)

        perm_back = (0, x.ndim - 1) + tuple(range(1, x.ndim - 1))
        x = x.permute(*perm_back).contiguous()
        x = self.fno_blocks(x)

        perm_proj = (0, *range(2, x.ndim), 1)
        x = x.permute(*perm_proj).contiguous()
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x


class FNO(TorchModel):
    def __init__(self, input_dim, output_dim, modes, width, dims, depth=4):
        model = FNOBase(input_dim, output_dim, modes, width, dims, depth)
        kwargs = {}
        super(FNO, self).__init__(model=model,
                                        loss=self._loss_fn,
                                        **kwargs)

    def _loss_fn(self, outputs, labels, weights=None):
        labels = labels[0]
        outputs = outputs[0]
        return nn.MSELoss()(outputs, labels)
        