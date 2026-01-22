import pytest
import numpy as np

try:
    import torch
    from deepchem.models.torch_models.layers import ActNorm, FourierBlock1D
    has_torch = True
except ModuleNotFoundError:
    has_torch = False
    pass


@pytest.mark.torch
def test_actnorm_identity():
    """ActNorm: identity behavior when scale=1, bias=0."""
    layer = ActNorm(dim=3)
    x = torch.randn(4, 5, 3)
    out = layer(x)
    assert torch.allclose(
        out, x), "ActNorm did not act as identity with default params"


@pytest.mark.torch
def test_actnorm_gradients():
    """ActNorm: scale & bias should receive gradients."""
    layer = ActNorm(dim=2)
    x = torch.randn(2, 3, 2, requires_grad=True)
    out = layer(x)
    out.sum().backward()
    assert layer.scale.grad is not None
    assert layer.bias.grad is not None


@pytest.mark.torch
def test_actnorm_custom_params():
    """ActNorm: applies custom scale & bias correctly."""
    layer = ActNorm(dim=1)
    with torch.no_grad():
        layer.scale.fill_(2.0)
        layer.bias.fill_(3.0)
    x = torch.tensor([[[1.0], [2.0]]])
    out = layer(x)
    expected = x * 2.0 + 3.0
    assert torch.allclose(out, expected)


@pytest.mark.torch
def test_actnorm_broadcasting():
    """ActNorm: broadcasts scale & bias over batch & length dims."""
    layer = ActNorm(dim=4)
    with torch.no_grad():
        layer.scale[:] = torch.arange(4).view(1, 1, 4)
        layer.bias[:] = 0.5
    for shape in [(1, 1, 4), (2, 3, 4), (5, 1, 4)]:
        x = torch.ones(*shape)
        out = layer(x)
        expected = torch.stack(
            [torch.ones(*shape[:-1]) * i + 0.5 for i in range(4)], dim=-1)
        assert torch.allclose(out, expected)


@pytest.mark.torch
def test_fourierblock_shape_and_grad():
    """FourierBlock1D: output shape should be (B, L, C), and gradients should flow."""
    B, C, L, modes = 2, 4, 16, 8
    block = FourierBlock1D(in_channels=C, out_channels=C, modes=modes)
    x = torch.randn(B, C, L, requires_grad=True)
    y = block(x)
    assert y.shape == (B, L, C)
    y.sum().backward()
    assert x.grad is not None
    assert block.weights.grad is not None


@pytest.mark.torch
def test_fourierblock_mode_truncation():
    """FourierBlock1D: modes beyond the specified cutoff should be zeroed."""
    block = FourierBlock1D(in_channels=1, out_channels=1, modes=4)
    L = 32
    grid = np.linspace(0, 1, L, endpoint=False)
    signal = np.sin(2 * np.pi * 5 * grid)  # freq > modes=4
    x = torch.tensor(signal, dtype=torch.float32).view(1, 1, L)
    y = block(x)
    assert torch.allclose(y, torch.zeros_like(y), atol=1e-5)


@pytest.mark.torch
def test_fourierblock_zero_weights_identity():
    """FourierBlock1D: zero weights & zero normalization yields zero output."""
    block = FourierBlock1D(in_channels=2, out_channels=2, modes=8)
    with torch.no_grad():
        block.weights.zero_()
        block.norm.scale.zero_()
        block.norm.bias.zero_()
    x = torch.randn(1, 2, 16)
    y = block(x)
    assert torch.allclose(y, torch.zeros_like(y))


@pytest.mark.torch
def test_fourierblock_spectral_operation_correctness():
    """FourierBlock1D: output matches NumPy FFT→multiply→IFFT reference pipeline."""
    block = FourierBlock1D(in_channels=1, out_channels=1, modes=8)
    with torch.no_grad():
        block.weights[:] = torch.tensor([[[[1.0, 0.0]]]])
    L = 8
    x_np = np.arange(L, dtype=np.float32)
    x = torch.tensor(x_np).view(1, 1, L)
    y_block = block(x).detach().view(L).numpy()
    xf = np.fft.rfft(x_np)
    y_ref = np.fft.irfft(xf, n=L)
    assert np.allclose(y_block, y_ref, atol=1e-5)
