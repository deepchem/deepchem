"""Tests for the RFDiffusion All-Atom ligand-conditioned model."""

import numpy as np
import pytest

try:
    import deepchem as dc
    import torch
    from deepchem.models.torch_models.rfdiffusion_aa import (
        LigandContextEmbedding, RFDiffusionAA, RFDiffusionAADenoiser,
        _ligand_protein_clash)
    from deepchem.utils.rfdiffusion_ligand import LigandPointCloud
    has_dc = True
except ImportError:
    has_dc = False

requires_dc = pytest.mark.skipif(not has_dc,
                                 reason='deepchem or torch not installed')


def _make_dataset(n=4, length=15):
    proteins = [np.random.randn(length, 9).astype(np.float32) for _ in range(n)]
    X = np.empty(n, dtype=object)
    for i, p in enumerate(proteins):
        X[i] = p
    y = np.zeros((n, 1), dtype=np.float32)
    return dc.data.NumpyDataset(X=X, y=y)


def _make_ligand(num_atoms=4, seed=0):
    rng = np.random.RandomState(seed)
    return LigandPointCloud(coords=rng.randn(num_atoms, 3).astype(np.float32),
                            atom_types=rng.randint(0, 47, size=num_atoms),
                            atomic_numbers=rng.randint(1, 30, size=num_atoms),
                            bond_features=np.zeros((num_atoms, num_atoms),
                                                   dtype=np.int64))


def _small_model(**kw):
    defaults = dict(ligand=_make_ligand(),
                    embed_dim=16,
                    pair_dim=8,
                    num_blocks=1,
                    num_heads=2,
                    pair_num_heads=2,
                    ligand_num_heads=2,
                    num_diffusion_steps=10,
                    batch_size=2)
    defaults.update(kw)
    return RFDiffusionAA(**defaults)


@pytest.mark.torch
@requires_dc
class TestLigandContextEmbedding:

    def test_output_shape(self):
        embed = LigandContextEmbedding(embed_dim=16)
        atom_types = torch.zeros(2, 5, dtype=torch.long)
        coords = torch.randn(2, 5, 3)
        out = embed(atom_types, coords)
        assert out.shape == (2, 5, 16)

    def test_invalid_embed_dim_raises(self):
        with pytest.raises(ValueError):
            LigandContextEmbedding(embed_dim=0)


@pytest.mark.torch
@requires_dc
class TestRFDiffusionAADenoiser:

    def _denoiser(self, **kw):
        defaults = dict(embed_dim=16,
                        pair_dim=8,
                        num_blocks=1,
                        num_heads=2,
                        pair_num_heads=2,
                        ligand_num_heads=2)
        defaults.update(kw)
        return RFDiffusionAADenoiser(**defaults)

    def test_output_shapes_with_ligand(self):
        denoiser = self._denoiser()
        noisy_coords = torch.randn(2, 6, 9)
        R = torch.eye(3).expand(2, 6, 3, 3).contiguous()
        t = torch.zeros(2, 6, 3)
        timesteps = torch.tensor([0, 5])
        mask = torch.ones(2, 6)
        ligand_types = torch.zeros(2, 4, dtype=torch.long)
        ligand_coords = torch.randn(2, 4, 3)
        pred_R, pred_t = denoiser([
            noisy_coords, R, t, timesteps, mask, ligand_types, ligand_coords,
            None
        ])
        assert pred_R.shape == (2, 6, 3, 3)
        assert pred_t.shape == (2, 6, 3)

    def test_works_without_ligand(self):
        denoiser = self._denoiser()
        noisy_coords = torch.randn(1, 5, 9)
        R = torch.eye(3).expand(1, 5, 3, 3).contiguous()
        t = torch.zeros(1, 5, 3)
        timesteps = torch.tensor([2])
        mask = torch.ones(1, 5)
        pred_R, pred_t = denoiser(
            [noisy_coords, R, t, timesteps, mask, None, None, None])
        assert pred_R.shape == (1, 5, 3, 3)
        assert pred_t.shape == (1, 5, 3)

    def test_finite_output(self):
        denoiser = self._denoiser()
        noisy_coords = torch.randn(1, 4, 9)
        R = torch.eye(3).expand(1, 4, 3, 3).contiguous()
        t = torch.zeros(1, 4, 3)
        timesteps = torch.tensor([1])
        mask = torch.ones(1, 4)
        ligand_types = torch.zeros(1, 3, dtype=torch.long)
        ligand_coords = torch.randn(1, 3, 3)
        pred_R, pred_t = denoiser([
            noisy_coords, R, t, timesteps, mask, ligand_types, ligand_coords,
            None
        ])
        assert torch.isfinite(pred_R).all()
        assert torch.isfinite(pred_t).all()

    def test_ligand_mask_respected(self):
        torch.manual_seed(0)
        denoiser = self._denoiser()
        denoiser.eval()
        noisy_coords = torch.randn(1, 4, 9)
        R = torch.eye(3).expand(1, 4, 3, 3).contiguous()
        t = torch.zeros(1, 4, 3)
        timesteps = torch.tensor([1])
        mask = torch.ones(1, 4)
        ligand_types = torch.zeros(1, 3, dtype=torch.long)
        ligand_coords = torch.randn(1, 3, 3)

        drop_last_mask = torch.tensor([[1.0, 1.0, 0.0]])
        perturbed_coords = ligand_coords.clone()
        perturbed_coords[0, -1] += 50.0

        out1, _ = denoiser([
            noisy_coords, R, t, timesteps, mask, ligand_types, ligand_coords,
            drop_last_mask
        ])
        out2, _ = denoiser([
            noisy_coords, R, t, timesteps, mask, ligand_types, perturbed_coords,
            drop_last_mask
        ])
        assert torch.allclose(out1, out2, atol=1e-4)


@pytest.mark.torch
@requires_dc
class TestLigandProteinClash:

    def test_zero_when_far_apart(self):
        protein = torch.tensor([[[0.0, 0, 0]]])  # (1, 1, 3)
        ligand = torch.tensor([[[100.0, 0, 0]]])  # (1, 1, 3)
        protein_radii = torch.tensor([1.7])
        ligand_radii = torch.tensor([1.7])
        mask = torch.ones(1, 1)
        clash = _ligand_protein_clash(protein, protein_radii, ligand,
                                      ligand_radii, mask)
        assert round(float(clash), 6) == 0.0

    def test_positive_when_overlapping(self):
        protein = torch.tensor([[[0.0, 0, 0]]])
        ligand = torch.tensor([[[0.5, 0, 0]]])
        protein_radii = torch.tensor([1.7])
        ligand_radii = torch.tensor([1.7])
        mask = torch.ones(1, 1)
        clash = _ligand_protein_clash(protein, protein_radii, ligand,
                                      ligand_radii, mask)
        assert float(clash) > 0.0

    def test_masked_protein_atom_excluded(self):
        protein = torch.tensor([[[100.0, 0, 0], [0.5, 0, 0]]])
        ligand = torch.tensor([[[0.5, 0, 0]]])
        protein_radii = torch.tensor([1.7, 1.7])
        ligand_radii = torch.tensor([1.7])
        mask = torch.tensor([[1.0, 0.0]])  # clashing atom masked out
        clash = _ligand_protein_clash(protein, protein_radii, ligand,
                                      ligand_radii, mask)
        assert round(float(clash), 6) == 0.0


@pytest.mark.torch
@requires_dc
class TestRFDiffusionAA:

    def test_fit_returns_loss(self):
        model = _small_model()
        loss = model.fit(_make_dataset(), nb_epoch=1)
        assert isinstance(loss, float)
        assert np.isfinite(loss)

    def test_generate_shape_and_finite(self):
        model = _small_model()
        samples = model.generate(num_samples=2, seq_length=10)
        assert samples.shape == (2, 10, 9)
        assert np.isfinite(samples).all()

    def test_generate_after_fit(self):
        model = _small_model()
        model.fit(_make_dataset(), nb_epoch=1)
        samples = model.generate(num_samples=2, seq_length=12)
        assert samples.shape == (2, 12, 9)
        assert np.isfinite(samples).all()

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

    def test_generate_input_validation(self):
        model = _small_model()
        with pytest.raises(ValueError):
            model.generate(num_samples=0)
        with pytest.raises(ValueError):
            model.generate(seq_length=0)
        with pytest.raises(ValueError):
            model.generate(seq_length=model.max_seq_len + 1)

    def test_invalid_embed_dim_raises(self):
        with pytest.raises(ValueError):
            _small_model(embed_dim=0)

    def test_invalid_num_diffusion_steps_raises(self):
        with pytest.raises(ValueError):
            _small_model(num_diffusion_steps=0)

    def test_save_and_reload(self, tmp_path):
        model = _small_model()
        model.fit(_make_dataset(), nb_epoch=1)
        model.save_checkpoint(model_dir=str(tmp_path))

        model2 = _small_model()
        model2.restore(model_dir=str(tmp_path))
        assert model2._train_std == model._train_std

    def test_ligand_identity_changes_generation_after_training(self):
        # BackboneUpdate's output head is zero-initialized (a standard
        # "stable training start" pattern -- see rfdiffusion_sequence_track's
        # BackboneUpdate), so an *untrained* model's frame predictions are
        # exactly the identity update regardless of the input, including
        # the ligand. Conditioning can only show up once training has
        # moved those weights away from zero, so that's what this test
        # checks, with everything else (weight init, training data order,
        # generation noise) held fixed via matched seeds.
        # Genuinely different shapes -- not a translation/rotation of one
        # another, since centering (see RFDiffusionAA.__init__) removes
        # translation differences before the ligand ever reaches the
        # model.
        ligand_a = LigandPointCloud(coords=np.array(
            [[0.0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
                                    atom_types=np.array([4, 12, 12]),
                                    atomic_numbers=np.array([8, 6, 6]),
                                    bond_features=np.zeros((3, 3),
                                                           dtype=np.int64))
        ligand_b = LigandPointCloud(coords=np.array(
            [[0.0, 0, 0], [5, 0, 0], [0, 5, 5]], dtype=np.float32),
                                    atom_types=np.array([8, 20, 30]),
                                    atomic_numbers=np.array([8, 6, 6]),
                                    bond_features=np.zeros((3, 3),
                                                           dtype=np.int64))

        def build_and_train(ligand):
            torch.manual_seed(42)
            model = _small_model(ligand=ligand)
            torch.manual_seed(7)
            model.fit(_make_dataset(n=6, length=10), nb_epoch=15)
            return model

        model_a = build_and_train(ligand_a)
        model_b = build_and_train(ligand_b)

        torch.manual_seed(123)
        out_a = model_a.generate(num_samples=1, seq_length=8)
        torch.manual_seed(123)
        out_b = model_b.generate(num_samples=1, seq_length=8)
        assert not np.allclose(out_a, out_b, atol=1e-4)

    def test_clash_weight_zero_runs(self):
        # clash_weight=0 should still run (loss reduces to the frame loss).
        model = _small_model(clash_weight=0.0)
        loss = model.fit(_make_dataset(), nb_epoch=1)
        assert np.isfinite(loss)
