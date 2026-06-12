"""Tests for the RFDiffusion input embedding layers."""

import pytest

from deepchem.models.torch_models.rfdiffusion import (
    PositionalEncoding,
    ResidueEmbedding,
    SinusoidalTimestepEmbedding,
)

try:
    import torch
    has_torch = True
except ImportError:
    has_torch = False


@pytest.mark.torch
@pytest.mark.skipif(not has_torch, reason="PyTorch not installed")
class TestSinusoidalTimestepEmbedding:

    def test_output_shape(self):
        emb = SinusoidalTimestepEmbedding(64)
        t = torch.tensor([0, 50, 100, 999])
        assert emb(t).shape == (4, 64)

    def test_different_timesteps_differ(self):
        emb = SinusoidalTimestepEmbedding(64)
        out = emb(torch.tensor([0, 500]))
        assert not torch.allclose(out[0], out[1])

    def test_not_all_zeros(self):
        emb = SinusoidalTimestepEmbedding(32)
        out = emb(torch.tensor([100]))
        assert not torch.allclose(out, torch.zeros_like(out))

    def test_gradient_flows(self):
        # timestep embedding has no parameters, but input grad should work
        emb = SinusoidalTimestepEmbedding(64)
        t = torch.tensor([50])
        out = emb(t)
        # just checking no error is raised
        assert out.shape == (1, 64)


@pytest.mark.torch
@pytest.mark.skipif(not has_torch, reason="PyTorch not installed")
class TestResidueEmbedding:

    def test_output_shape(self):
        emb = ResidueEmbedding(9, 128)
        x = torch.randn(2, 50, 9)
        assert emb(x).shape == (2, 50, 128)

    def test_gradient_flow(self):
        emb = ResidueEmbedding(9, 64)
        x = torch.randn(1, 10, 9, requires_grad=True)
        out = emb(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_zero_input_does_not_explode(self):
        emb = ResidueEmbedding(9, 64)
        x = torch.zeros(1, 5, 9)
        out = emb(x)
        assert torch.isfinite(out).all()

    def test_custom_coord_dim(self):
        emb = ResidueEmbedding(coord_dim=14, embed_dim=32)
        x = torch.randn(1, 8, 14)
        assert emb(x).shape == (1, 8, 32)


@pytest.mark.torch
@pytest.mark.skipif(not has_torch, reason="PyTorch not installed")
class TestPositionalEncoding:

    def test_output_shape(self):
        pe = PositionalEncoding(128, max_len=256)
        x = torch.randn(2, 50, 128)
        assert pe(x).shape == (2, 50, 128)

    def test_adds_to_input(self):
        pe = PositionalEncoding(64, max_len=100)
        x = torch.zeros(1, 10, 64)
        out = pe(x)
        assert not torch.allclose(out, x)

    def test_different_positions_differ(self):
        pe = PositionalEncoding(64)
        x = torch.zeros(1, 5, 64)
        out = pe(x)
        # each row should be different because positions differ
        for i in range(4):
            assert not torch.allclose(out[0, i], out[0, i + 1])

    def test_preserves_shape_for_short_seq(self):
        pe = PositionalEncoding(32, max_len=512)
        x = torch.randn(4, 1, 32)
        assert pe(x).shape == (4, 1, 32)
