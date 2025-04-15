import pytest
from deepchem.models.torch_models.layers import SpectralConv

try:
    import torch
    has_torch = True
except ModuleNotFoundError:
    has_torch = False
    pass


@pytest.mark.torch
def test_spectral_conv_output_shape():
    """Test if spectral convolution output shape is correct"""
    conv = SpectralConv(in_channels=4, out_channels=8, modes=12, dims=2)
    x = torch.randn(2, 4, 64, 64)  # batch=2, channels=4, H=64, W=64
    y = conv(x)
    assert y.shape == (2, 8, 64, 64)


@pytest.mark.torch
def test_spectral_conv_gradient_flow():
    """Test if gradients flow through spectral convolution"""
    conv = SpectralConv(3, 6, modes=10, dims=2)
    x = torch.randn(2, 3, 64, 64, requires_grad=True)
    y = conv(x)
    loss = y.mean()
    loss.backward()
    assert x.grad is not None


@pytest.mark.parametrize("dims,input_shape", [
    (1, (2, 4, 128)),
    (2, (2, 4, 64, 64)),
    (3, (2, 4, 32, 32, 32)),
])
def test_spectral_conv_dims(dims, input_shape):
    """Test if spectral convolution works with different dimensions 1D, 2D, 3D"""
    conv = SpectralConv(in_channels=4, out_channels=6, modes=8, dims=dims)
    x = torch.randn(*input_shape)
    y = conv(x)
    assert y.shape == (input_shape[0], 6, *input_shape[2:])
