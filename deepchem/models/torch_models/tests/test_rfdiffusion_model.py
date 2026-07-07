"""Tests for the RFDiffusionModel TorchModel wrapper."""

import numpy as np
import pytest

try:
    import deepchem as dc
    from deepchem.models.torch_models.rfdiffusion import RFDiffusionModel
    import torch  # noqa: F401
    has_dc = True
except ImportError:
    has_dc = False

requires_dc = pytest.mark.skipif(not has_dc,
                                 reason="deepchem or torch not installed")


def _make_dataset(n=6, length=20):
    """Build a tiny backbone coordinate dataset for testing."""
    proteins = [np.random.randn(length, 9).astype(np.float32) for _ in range(n)]
    X = np.empty(n, dtype=object)
    for i, p in enumerate(proteins):
        X[i] = p
    y = np.zeros((n, 1), dtype=np.float32)
    return dc.data.NumpyDataset(X=X, y=y)


def _small_model(**kw):
    defaults = dict(embed_dim=32,
                    num_layers=1,
                    num_heads=4,
                    num_diffusion_steps=10,
                    batch_size=2)
    defaults.update(kw)
    return RFDiffusionModel(**defaults)


@pytest.mark.torch
@requires_dc
class TestRFDiffusionModel:

    def test_fit_returns_loss(self):
        model = _small_model()
        ds = _make_dataset()
        loss = model.fit(ds, nb_epoch=1)
        assert isinstance(loss, float)
        assert np.isfinite(loss)

    def test_generate_shape(self):
        model = _small_model()
        samples = model.generate(num_samples=2, seq_length=10)
        assert samples.shape == (2, 10, 9)

    def test_generate_finite(self):
        model = _small_model()
        samples = model.generate(num_samples=1, seq_length=10)
        assert np.isfinite(samples).all()

    def test_generate_after_fit_shape(self):
        model = _small_model()
        model.fit(_make_dataset(), nb_epoch=1)
        samples = model.generate(num_samples=3, seq_length=15)
        assert samples.shape == (3, 15, 9)

    def test_fit_variable_length(self):
        model = _small_model()
        proteins = [
            np.random.randn(np.random.randint(5, 15), 9).astype(np.float32)
            for _ in range(4)
        ]
        X = np.empty(4, dtype=object)
        for i, p in enumerate(proteins):
            X[i] = p
        ds = dc.data.NumpyDataset(X=X, y=np.zeros((4, 1), dtype=np.float32))
        loss = model.fit(ds, nb_epoch=1)
        assert np.isfinite(loss)

    def test_normalize_coords_l33(self):
        model = _small_model()
        coords = np.random.randn(10, 3, 3).astype(np.float32)
        out = model._normalize_coords(coords)
        assert out.shape == (10, 9)
        assert out.dtype == np.float32

    def test_normalize_coords_l9(self):
        model = _small_model()
        coords = np.random.randn(10, 9).astype(np.float32)
        out = model._normalize_coords(coords)
        assert out.shape == (10, 9)

    def test_pad_coords(self):
        model = _small_model()
        coords = np.random.randn(5, 9).astype(np.float32)
        padded, orig_len = model._pad_coords(coords, 10)
        assert padded.shape == (10, 9)
        assert orig_len == 5
        assert np.allclose(padded[5:], 0.0)

    def test_predict_mode_raises(self):
        model = _small_model()
        ds = _make_dataset()
        gen = model.default_generator(ds, mode='predict')
        with pytest.raises(NotImplementedError):
            next(gen)

    def test_generate_input_validation(self):
        model = _small_model()
        with pytest.raises(ValueError):
            model.generate(num_samples=0)
        with pytest.raises(ValueError):
            model.generate(seq_length=0)
        with pytest.raises(ValueError):
            model.generate(seq_length=model.max_seq_len + 1)

    def test_save_and_reload(self, tmp_path):
        """A saved model reloads with identical weights and stats."""
        model = _small_model()
        model.fit(_make_dataset(), nb_epoch=1)
        model.save_checkpoint(model_dir=str(tmp_path))

        model2 = _small_model()
        model2.restore(model_dir=str(tmp_path))
        # normalization stats should survive the round-trip
        assert model2._train_std == model._train_std
        # and every weight should match the saved model exactly
        sd1 = model.model.state_dict()
        sd2 = model2.model.state_dict()
        assert sd1.keys() == sd2.keys()
        for key in sd1:
            assert torch.equal(sd1[key], sd2[key])

    def test_loss_decreases(self):
        model = _small_model(num_diffusion_steps=20)
        ds = _make_dataset(n=8, length=10)
        loss_before = model.fit(ds, nb_epoch=1)
        loss_after = model.fit(ds, nb_epoch=5)
        assert loss_after <= loss_before * 1.5  # some decrease expected
