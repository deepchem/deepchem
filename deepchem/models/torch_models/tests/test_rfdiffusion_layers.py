"""Tests for RFDiffusion diffusion layers.

Tests for the neural network layers and noise scheduling components
used by the RFDiffusion protein backbone generation model.
"""

import pytest

from deepchem.models.torch_models.rfdiffusion import (
    SinusoidalTimestepEmbedding,
    ResidueEmbedding,
    PositionalEncoding,
    DiffusionTransformerBlock,
    BackboneDiffusion,
    CosineSchedule,
)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    has_torch = True
except ImportError:
    has_torch = False


@pytest.mark.torch
@pytest.mark.skipif(not has_torch, reason="PyTorch not installed")
class TestSinusoidalTimestepEmbedding:
    """Tests for SinusoidalTimestepEmbedding."""

    def test_output_shape(self):
        """Test that output has correct shape."""
        emb = SinusoidalTimestepEmbedding(64)
        t = torch.tensor([0, 50, 100, 999])
        output = emb(t)
        assert output.shape == (4, 64)

    def test_different_timesteps_differ(self):
        """Test that different timesteps produce different embeddings."""
        emb = SinusoidalTimestepEmbedding(64)
        t = torch.tensor([0, 500])
        output = emb(t)
        assert not torch.allclose(output[0], output[1])

    def test_not_all_zeros(self):
        """Test that output is non-trivial."""
        emb = SinusoidalTimestepEmbedding(32)
        t = torch.tensor([100])
        output = emb(t)
        assert not torch.allclose(output, torch.zeros_like(output))

    def test_odd_dimension(self):
        """Test odd embedding dimensions are preserved."""
        emb = SinusoidalTimestepEmbedding(65)
        t = torch.tensor([0, 10])
        output = emb(t)
        assert output.shape == (2, 65)


@pytest.mark.torch
@pytest.mark.skipif(not has_torch, reason="PyTorch not installed")
class TestResidueEmbedding:
    """Tests for ResidueEmbedding."""

    def test_output_shape(self):
        """Test that output has correct shape."""
        emb = ResidueEmbedding(9, 128)
        x = torch.randn(2, 50, 9)
        output = emb(x)
        assert output.shape == (2, 50, 128)

    def test_gradient_flow(self):
        """Test that gradients flow through the embedding."""
        emb = ResidueEmbedding(9, 64)
        x = torch.randn(1, 10, 9, requires_grad=True)
        output = emb(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape


@pytest.mark.torch
@pytest.mark.skipif(not has_torch, reason="PyTorch not installed")
class TestPositionalEncoding:
    """Tests for PositionalEncoding."""

    def test_output_shape(self):
        """Test that output preserves input shape."""
        pe = PositionalEncoding(128, max_len=256)
        x = torch.randn(2, 50, 128)
        output = pe(x)
        assert output.shape == (2, 50, 128)

    def test_adds_to_input(self):
        """Test that positional encoding modifies the input."""
        pe = PositionalEncoding(64, max_len=100)
        x = torch.zeros(1, 10, 64)
        output = pe(x)
        assert not torch.allclose(output, x)

    def test_odd_dimension(self):
        """Test odd embedding dimensions are supported."""
        pe = PositionalEncoding(65, max_len=100)
        x = torch.zeros(1, 10, 65)
        output = pe(x)
        assert output.shape == (1, 10, 65)

    def test_over_max_len_raises(self):
        """Test long sequences fail with a clear error."""
        pe = PositionalEncoding(64, max_len=5)
        x = torch.zeros(1, 6, 64)
        with pytest.raises(ValueError, match="exceeds max_len"):
            pe(x)


@pytest.mark.torch
@pytest.mark.skipif(not has_torch, reason="PyTorch not installed")
class TestDiffusionTransformerBlock:
    """Tests for DiffusionTransformerBlock."""

    def test_output_shape(self):
        """Test that output matches input shape."""
        block = DiffusionTransformerBlock(128, num_heads=4)
        x = torch.randn(2, 50, 128)
        t_emb = torch.randn(2, 128)
        output = block(x, t_emb)
        assert output.shape == (2, 50, 128)

    def test_time_conditioning_matters(self):
        """Test that different timesteps produce different outputs."""
        block = DiffusionTransformerBlock(64, num_heads=4)
        block.eval()
        x = torch.randn(1, 10, 64)
        t1 = torch.randn(1, 64)
        t2 = torch.randn(1, 64) * 5
        out1 = block(x, t1)
        out2 = block(x, t2)
        assert not torch.allclose(out1, out2)

    def test_attention_mask_shape(self):
        """Test attention masking preserves output shape."""
        block = DiffusionTransformerBlock(64, num_heads=4)
        x = torch.randn(2, 5, 64)
        t_emb = torch.randn(2, 64)
        mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]],
                            dtype=torch.bool)
        output = block(x, t_emb, attention_mask=mask)
        assert output.shape == x.shape


@pytest.mark.torch
@pytest.mark.skipif(not has_torch, reason="PyTorch not installed")
class TestBackboneDiffusion:
    """Tests for BackboneDiffusion network."""

    def test_output_shape(self):
        """Test that output shape matches input coordinates."""
        model = BackboneDiffusion(coord_dim=9,
                                  embed_dim=64,
                                  num_layers=2,
                                  num_heads=4)
        x = torch.randn(2, 20, 9)
        t = torch.randint(0, 100, (2,))
        output = model([x, t])
        assert output.shape == (2, 20, 9)

    def test_different_timesteps(self):
        """Test that different timesteps give different predictions
        after one gradient step to break zero-init symmetry."""
        model = BackboneDiffusion(coord_dim=9,
                                  embed_dim=64,
                                  num_layers=2,
                                  num_heads=4)
        # One training step to break zero-init symmetry
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        x = torch.randn(1, 10, 9)
        t = torch.tensor([500])
        pred = model([x, t])
        loss = pred.sum()
        loss.backward()
        optimizer.step()

        model.eval()
        t_early = torch.tensor([10])
        t_late = torch.tensor([900])
        out_early = model([x, t_early])
        out_late = model([x, t_late])
        assert not torch.allclose(out_early, out_late)

    def test_zero_initialized_output(self):
        """Test that output starts near zero due to zero-init convention."""
        model = BackboneDiffusion(coord_dim=9,
                                  embed_dim=64,
                                  num_layers=2,
                                  num_heads=4)
        x = torch.randn(1, 10, 9)
        t = torch.tensor([500])
        output = model([x, t])
        # Output should be relatively small due to zero initialization
        assert output.abs().mean().item() < 1.0

    def test_variable_length(self):
        """Test that model handles different sequence lengths."""
        model = BackboneDiffusion(coord_dim=9,
                                  embed_dim=64,
                                  num_layers=2,
                                  num_heads=4)
        for length in [5, 20, 100]:
            x = torch.randn(1, length, 9)
            t = torch.tensor([100])
            output = model([x, t])
            assert output.shape == (1, length, 9)

    def test_gradient_flow(self):
        """Test that gradients flow through the entire model."""
        model = BackboneDiffusion(coord_dim=9,
                                  embed_dim=64,
                                  num_layers=2,
                                  num_heads=4)
        x = torch.randn(2, 10, 9, requires_grad=True)
        t = torch.randint(0, 100, (2,))
        output = model([x, t])
        loss = output.sum()
        loss.backward()
        assert x.grad is not None

    def test_over_max_seq_len_raises(self):
        """Test long sequences fail with a clear error."""
        model = BackboneDiffusion(coord_dim=9,
                                  embed_dim=64,
                                  num_layers=2,
                                  num_heads=4,
                                  max_seq_len=5)
        x = torch.randn(1, 6, 9)
        t = torch.tensor([100])
        with pytest.raises(ValueError, match="exceeds max_seq_len"):
            model([x, t])

    def test_masked_positions_zeroed(self):
        """Test masked residues are zeroed in the output."""
        model = BackboneDiffusion(coord_dim=9,
                                  embed_dim=64,
                                  num_layers=2,
                                  num_heads=4)
        nn.init.xavier_uniform_(model.output_proj[-1].weight)
        x = torch.randn(1, 5, 9)
        t = torch.tensor([100])
        mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.float32)
        output = model([x, t, mask])
        assert torch.allclose(output[:, 3:], torch.zeros_like(output[:, 3:]))

    def test_self_conditioning_output_shape(self):
        """Test output shape when self_conditioning is enabled."""
        model = BackboneDiffusion(coord_dim=9,
                                  embed_dim=64,
                                  num_layers=2,
                                  num_heads=4,
                                  self_conditioning=True)
        x = torch.randn(2, 10, 9)
        t = torch.randint(0, 100, (2,))
        x0_prev = torch.randn(2, 10, 9)
        output = model([x, t, x0_prev])
        assert output.shape == (2, 10, 9)

    def test_self_conditioning_differs_from_without(self):
        """Test that self-conditioning input changes the output."""
        model = BackboneDiffusion(coord_dim=9,
                                  embed_dim=64,
                                  num_layers=2,
                                  num_heads=4,
                                  self_conditioning=True)
        # Give the output projection non-zero weights so we can
        # observe the effect of self-conditioning on the output.
        nn.init.xavier_uniform_(model.output_proj[-1].weight)
        x = torch.randn(2, 10, 9)
        t = torch.randint(0, 100, (2,))
        out_no_cond = model([x, t])
        x0_prev = torch.randn(2, 10, 9)
        out_with_cond = model([x, t, x0_prev])
        assert not torch.allclose(out_no_cond, out_with_cond)

    def test_self_conditioning_gradient_flow(self):
        """Test gradients flow through self-conditioning input."""
        model = BackboneDiffusion(coord_dim=9,
                                  embed_dim=64,
                                  num_layers=2,
                                  num_heads=4,
                                  self_conditioning=True)
        x = torch.randn(2, 10, 9)
        t = torch.randint(0, 100, (2,))
        x0_prev = torch.randn(2, 10, 9, requires_grad=True)
        output = model([x, t, x0_prev])
        loss = output.sum()
        loss.backward()
        assert x0_prev.grad is not None


@pytest.mark.torch
@pytest.mark.skipif(not has_torch, reason="PyTorch not installed")
class TestCosineSchedule:
    """Tests for CosineSchedule."""

    def test_q_sample_shape(self):
        """Test that forward diffusion preserves shape."""
        schedule = CosineSchedule(num_timesteps=100)
        x0 = torch.randn(2, 10, 9)
        t = torch.tensor([0, 50])
        noisy_x, noise = schedule.q_sample(x0, t)
        assert noisy_x.shape == x0.shape
        assert noise.shape == x0.shape

    def test_no_noise_at_t0(self):
        """Test that t=0 adds almost no noise."""
        schedule = CosineSchedule(num_timesteps=1000)
        x0 = torch.randn(1, 10, 9)
        t = torch.tensor([0])
        noisy_x, noise = schedule.q_sample(x0, t)
        # At t=0, alpha_cumprod is close to 1, so noisy_x should be close to x0
        assert torch.allclose(noisy_x, x0, atol=0.05)

    def test_mostly_noise_at_tmax(self):
        """Test that t=T-1 produces mostly noise."""
        schedule = CosineSchedule(num_timesteps=1000)
        x0 = torch.ones(1, 10, 9)
        t = torch.tensor([999])
        noisy_x, noise = schedule.q_sample(x0, t)
        # At t=T, alpha_cumprod is close to 0, so noisy_x should be close to noise
        correlation = F.cosine_similarity(noisy_x.flatten().unsqueeze(0),
                                          noise.flatten().unsqueeze(0))
        assert correlation.item() > 0.9

    def test_alpha_cumprod_monotonic(self):
        """Test that alpha_cumprod decreases monotonically."""
        schedule = CosineSchedule(num_timesteps=1000)
        diffs = schedule.alpha_cumprod[1:] - schedule.alpha_cumprod[:-1]
        assert (diffs <= 0).all()

    def test_betas_bounded(self):
        """Test that betas are in valid range."""
        schedule = CosineSchedule(num_timesteps=1000)
        assert (schedule.betas >= 0).all()
        assert (schedule.betas <= 1).all()

    def test_out_of_range_timestep_raises(self):
        """Test invalid timesteps fail clearly."""
        schedule = CosineSchedule(num_timesteps=10)
        x0 = torch.randn(1, 10, 9)
        with pytest.raises(ValueError, match="out of range"):
            schedule.q_sample(x0, torch.tensor([10]))

    def test_p_sample_returns_tuple(self):
        """Test that p_sample returns (x_prev, x0_pred) tuple."""
        schedule = CosineSchedule(num_timesteps=50)
        model = BackboneDiffusion(coord_dim=9,
                                  embed_dim=64,
                                  num_layers=2,
                                  num_heads=4)
        x_t = torch.randn(2, 10, 9)
        t = torch.tensor([25, 25])
        x_prev, x0_pred = schedule.p_sample(model, x_t, t)
        assert x_prev.shape == (2, 10, 9)
        assert x0_pred.shape == (2, 10, 9)

    def test_sample_with_self_conditioning(self):
        """Test full reverse diffusion with self-conditioning."""
        schedule = CosineSchedule(num_timesteps=10)
        model = BackboneDiffusion(coord_dim=9,
                                  embed_dim=64,
                                  num_layers=2,
                                  num_heads=4,
                                  self_conditioning=True)
        shape = (2, 10, 9)
        samples = schedule.sample(model,
                                  shape,
                                  device=torch.device('cpu'),
                                  self_conditioning=True)
        assert samples.shape == shape
        assert torch.isfinite(samples).all()
