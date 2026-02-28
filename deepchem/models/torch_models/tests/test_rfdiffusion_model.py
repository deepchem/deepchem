"""Tests for RFDiffusionModel (TorchModel wrapper).

These tests verify that the RFDiffusionModel correctly implements
DeepChem's TorchModel interface, handles coordinate normalization
and denormalization, and produces valid outputs for protein backbone
generation tasks.
"""

import numpy as np
import pytest

import deepchem as dc
from deepchem.models.torch_models.rfdiffusion import RFDiffusionModel

try:
    import torch  # noqa: F401
    has_torch = True
except ImportError:
    has_torch = False


@pytest.mark.torch
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
            np.random.randn(lengths[i % len(lengths)], 3, 3).astype(np.float32)
            for i in range(n_samples)
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
        for p1, p2 in zip(model.model.parameters(), model2.model.parameters()):
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

    def test_denormalization_after_training(self):
        """Test that generate applies denormalization after training."""
        # Create dataset with known statistics (coords around 50 Angstroms)
        proteins = [(np.random.randn(15, 3, 3).astype(np.float32) * 3 + 50)
                    for _ in range(8)]
        X = np.empty(8, dtype=object)
        for i, p in enumerate(proteins):
            X[i] = p
        y = np.zeros((8, 1), dtype=np.float32)
        dataset = dc.data.NumpyDataset(X=X, y=y)

        model = RFDiffusionModel(embed_dim=64,
                                 num_layers=2,
                                 num_heads=4,
                                 num_diffusion_steps=10,
                                 batch_size=4)
        model.fit(dataset, nb_epoch=1)

        # After training, model should have accumulated statistics
        assert model._train_std is not None
        assert model._train_mean is not None
        assert model._train_std > 0

    def test_generate_without_training(self):
        """Test that generate works even without training (no denorm)."""
        model = RFDiffusionModel(embed_dim=64,
                                 num_layers=2,
                                 num_heads=4,
                                 num_diffusion_steps=10,
                                 batch_size=2)
        # No fit called, so _train_std and _train_mean are None
        assert model._train_std is None
        samples = model.generate(num_samples=1, seq_length=10)
        assert samples.shape == (1, 10, 9)
        assert np.isfinite(samples).all()
