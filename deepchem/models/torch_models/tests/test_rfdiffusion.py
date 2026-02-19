"""Tests for RFDiffusion model.

These tests verify that the RFDiffusion model correctly implements
DeepChem's TorchModel interface and produces valid outputs for
protein backbone generation tasks.
"""

import numpy as np
import pytest
import tempfile
import os

import deepchem as dc
from deepchem.models.torch_models.rfdiffusion import (
    SinusoidalTimestepEmbedding,
    ResidueEmbedding,
    PositionalEncoding,
    DiffusionTransformerBlock,
    BackboneDiffusion,
    CosineSchedule,
    RFDiffusionModel,
)

try:
    import torch
    import torch.nn.functional as F
    has_torch = True
except ImportError:
    has_torch = False


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
        """Test that different timesteps produce different predictions after
        breaking zero-init symmetry with one gradient step."""
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
        """Test that output starts near zero (zero-init convention)."""
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
        # At t=0, alpha_cumprod is close to 1, so noisy_x ≈ x0
        assert torch.allclose(noisy_x, x0, atol=0.05)

    def test_mostly_noise_at_tmax(self):
        """Test that t=T-1 produces mostly noise."""
        schedule = CosineSchedule(num_timesteps=1000)
        x0 = torch.ones(1, 10, 9)
        t = torch.tensor([999])
        noisy_x, noise = schedule.q_sample(x0, t)
        # At t=T, alpha_cumprod is close to 0, so noisy_x ≈ noise
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


@pytest.mark.skipif(not has_torch, reason="PyTorch not installed")
class TestRFDiffusionModel:
    """Tests for RFDiffusionModel (TorchModel wrapper)."""

    def _make_dataset(self, n_samples=10, seq_len=20):
        """Create a small test dataset."""
        proteins = [
            np.random.randn(seq_len, 3, 3).astype(np.float32)
            for _ in range(n_samples)
        ]
        X = np.empty(n_samples, dtype=object)
        for i, p in enumerate(proteins):
            X[i] = p
        y = np.zeros((n_samples, 1), dtype=np.float32)
        return dc.data.NumpyDataset(X=X, y=y)

    def _make_variable_length_dataset(self, n_samples=10):
        """Create a dataset with variable-length proteins."""
        lengths = [10, 15, 20, 25, 30, 12, 18, 22, 8, 35]
        proteins = [
            np.random.randn(lengths[i % len(lengths)], 3,
                            3).astype(np.float32) for i in range(n_samples)
        ]
        X = np.empty(n_samples, dtype=object)
        for i, p in enumerate(proteins):
            X[i] = p
        y = np.zeros((n_samples, 1), dtype=np.float32)
        return dc.data.NumpyDataset(X=X, y=y)

    def test_model_creation(self):
        """Test that model can be instantiated."""
        model = RFDiffusionModel(embed_dim=64,
                                 num_layers=2,
                                 num_heads=4,
                                 num_diffusion_steps=100,
                                 batch_size=2)
        assert model is not None
        assert model.num_diffusion_steps == 100

    def test_model_parameter_count(self):
        """Test that model has reasonable number of parameters."""
        model = RFDiffusionModel(embed_dim=64,
                                 num_layers=2,
                                 num_heads=4,
                                 batch_size=2)
        n_params = sum(p.numel() for p in model.model.parameters())
        assert n_params > 0
        # Small model should have parameters in reasonable range
        assert n_params < 10_000_000

    def test_fit_returns_loss(self):
        """Test that fit returns a finite loss value."""
        dataset = self._make_dataset(n_samples=8, seq_len=15)
        model = RFDiffusionModel(embed_dim=64,
                                 num_layers=2,
                                 num_heads=4,
                                 num_diffusion_steps=50,
                                 batch_size=4)
        loss = model.fit(dataset, nb_epoch=1)
        assert np.isfinite(loss)
        assert loss > 0

    def test_fit_variable_length(self):
        """Test training with variable-length proteins."""
        dataset = self._make_variable_length_dataset(n_samples=8)
        model = RFDiffusionModel(embed_dim=64,
                                 num_layers=2,
                                 num_heads=4,
                                 num_diffusion_steps=50,
                                 batch_size=4)
        loss = model.fit(dataset, nb_epoch=1)
        assert np.isfinite(loss)

    def test_loss_decreases(self):
        """Test that loss decreases during training (memorization test)."""
        # Use a tiny dataset - model should memorize it
        dataset = self._make_dataset(n_samples=4, seq_len=10)
        model = RFDiffusionModel(embed_dim=64,
                                 num_layers=2,
                                 num_heads=4,
                                 num_diffusion_steps=50,
                                 batch_size=4,
                                 learning_rate=1e-3)

        # Collect losses over many epochs to smooth out noise
        losses = []
        for _ in range(50):
            loss = model.fit(dataset, nb_epoch=1)
            losses.append(loss)

        # Average of first 5 vs last 5 epochs
        avg_early = np.mean(losses[:5])
        avg_late = np.mean(losses[-5:])

        # Loss should decrease after more training
        assert avg_late < avg_early

    def test_generate_shape(self):
        """Test that generate produces correct output shape."""
        model = RFDiffusionModel(embed_dim=64,
                                 num_layers=2,
                                 num_heads=4,
                                 num_diffusion_steps=10,
                                 batch_size=2)
        samples = model.generate(num_samples=2, seq_length=20)
        assert samples.shape == (2, 20, 9)

    def test_generate_finite(self):
        """Test that generated samples are finite."""
        model = RFDiffusionModel(embed_dim=64,
                                 num_layers=2,
                                 num_heads=4,
                                 num_diffusion_steps=10,
                                 batch_size=2)
        samples = model.generate(num_samples=1, seq_length=15)
        assert np.isfinite(samples).all()

    def test_save_and_reload(self):
        """Test model checkpointing."""
        dataset = self._make_dataset(n_samples=4, seq_len=10)
        model = RFDiffusionModel(embed_dim=64,
                                 num_layers=2,
                                 num_heads=4,
                                 num_diffusion_steps=50,
                                 batch_size=4)
        model.fit(dataset, nb_epoch=1)

        # Save and reload
        model.save_checkpoint()

        model2 = RFDiffusionModel(embed_dim=64,
                                  num_layers=2,
                                  num_heads=4,
                                  num_diffusion_steps=50,
                                  batch_size=4,
                                  model_dir=model.model_dir)
        model2.restore()

        # Check that parameters match after restore
        for p1, p2 in zip(model.model.parameters(),
                          model2.model.parameters()):
            np.testing.assert_array_almost_equal(p1.detach().cpu().numpy(),
                                                 p2.detach().cpu().numpy(),
                                                 decimal=5)

    def test_normalize_coords(self):
        """Test coordinate normalization."""
        model = RFDiffusionModel(embed_dim=64,
                                 num_layers=2,
                                 num_heads=4,
                                 batch_size=2)
        coords = np.random.randn(20, 3, 3).astype(np.float32) * 10 + 50
        normalized = model._normalize_coords(coords)

        assert normalized.shape == (20, 9)
        # Should be roughly centered
        assert abs(normalized.mean()) < 1.0
        # Should be roughly unit variance
        assert abs(normalized.std() - 1.0) < 0.5

    def test_pad_coords(self):
        """Test coordinate padding."""
        model = RFDiffusionModel(embed_dim=64,
                                 num_layers=2,
                                 num_heads=4,
                                 batch_size=2)
        coords = np.random.randn(10, 9).astype(np.float32)

        # Pad to longer
        padded, orig_len = model._pad_coords(coords, 20)
        assert padded.shape == (20, 9)
        assert orig_len == 10
        np.testing.assert_array_equal(padded[:10], coords)
        np.testing.assert_array_equal(padded[10:], 0)

        # Truncate longer
        long_coords = np.random.randn(30, 9).astype(np.float32)
        truncated, orig_len = model._pad_coords(long_coords, 20)
        assert truncated.shape == (20, 9)
        assert orig_len == 20

    def test_default_generator_yields_correct_format(self):
        """Test that default_generator produces proper batch format."""
        dataset = self._make_dataset(n_samples=4, seq_len=10)
        model = RFDiffusionModel(embed_dim=64,
                                 num_layers=2,
                                 num_heads=4,
                                 num_diffusion_steps=50,
                                 batch_size=4)

        gen = model.default_generator(dataset, epochs=1)
        batch = next(gen)
        inputs, labels, weights = batch

        # inputs should be [noisy_coords, timesteps]
        assert len(inputs) == 2
        assert inputs[0].shape[0] == 4  # batch_size
        assert inputs[0].shape[2] == 9  # coord_dim
        assert inputs[1].shape[0] == 4  # batch_size

        # labels should be [noise]
        assert len(labels) == 1
        assert labels[0].shape == inputs[0].shape

        # weights should be [ones]
        assert len(weights) == 1
