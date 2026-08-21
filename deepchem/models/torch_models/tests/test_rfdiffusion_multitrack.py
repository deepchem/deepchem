"""Tests for the RFDiffusion multi-track denoiser integration."""

import pytest

try:
    import torch
    from deepchem.models.torch_models.layers import CosineSchedule
    from deepchem.models.torch_models.rfdiffusion_frames import (
        build_backbone_frames, make_identity_rigid)
    from deepchem.models.torch_models.rfdiffusion_multitrack import (
        RFDiffusionMultiTrackDenoiser, backbone_coords_from_frames,
        multitrack_frame_loss, sample_noisy_frames, so3_x0_reverse_step,
        translation_posterior_step)
    from deepchem.models.torch_models.rfdiffusion_so3 import (IGSO3,
                                                              log_beta_schedule)
    has_torch = True
except ImportError:
    has_torch = False

requires_torch = pytest.mark.skipif(not has_torch,
                                    reason='PyTorch not installed')


def _schedule_and_igso3(num_steps=10):
    schedule = CosineSchedule(num_timesteps=num_steps)
    igso3 = IGSO3(log_beta_schedule(num_steps), num_omega=256)
    return schedule, igso3


@pytest.mark.torch
@requires_torch
class TestBackboneCoordsFromFrames:

    def test_identity_frame_gives_ideal_geometry(self):
        R, t = make_identity_rigid((2, 3))
        coords = backbone_coords_from_frames(R, t)
        assert coords.shape == (2, 3, 3, 3)
        # CA (middle atom) should sit exactly at the translation (origin).
        assert torch.allclose(coords[:, :, 1, :], t)

    def test_roundtrip_with_build_backbone_frames(self):
        torch.manual_seed(0)
        R, t = make_identity_rigid((4,))
        # Perturb with a random valid rotation via so3-consistent frames
        # built from random backbone coordinates, then reconstruct.
        backbone = torch.randn(4, 3, 3)
        R2, t2 = build_backbone_frames(backbone)
        coords = backbone_coords_from_frames(R2, t2)
        R3, t3 = build_backbone_frames(coords)
        assert torch.allclose(R2, R3, atol=1e-4)
        assert torch.allclose(t2, t3, atol=1e-4)

    def test_output_shape(self):
        R, t = make_identity_rigid((2, 5))
        coords = backbone_coords_from_frames(R, t)
        assert coords.shape == (2, 5, 3, 3)


@pytest.mark.torch
@requires_torch
class TestSampleNoisyFrames:

    def test_shapes_and_finite(self):
        schedule, igso3 = _schedule_and_igso3()
        R0, T0 = make_identity_rigid((2, 6))
        t = torch.tensor([1, 5])
        Rt, Tt = sample_noisy_frames(igso3, schedule, R0, T0, t)
        assert Rt.shape == (2, 6, 3, 3)
        assert Tt.shape == (2, 6, 3)
        assert torch.isfinite(Rt).all()
        assert torch.isfinite(Tt).all()

    def test_rotations_stay_valid(self):
        schedule, igso3 = _schedule_and_igso3()
        R0, T0 = make_identity_rigid((3, 4))
        t = torch.tensor([0, 4, 9])
        Rt, _ = sample_noisy_frames(igso3, schedule, R0, T0, t)
        det = torch.linalg.det(Rt)
        assert torch.allclose(det, torch.ones_like(det), atol=1e-3)
        orth = torch.matmul(Rt, Rt.transpose(-1, -2))
        assert torch.allclose(orth, torch.eye(3).expand_as(orth), atol=1e-3)

    def test_fixed_mask_holds_positions_clean(self):
        schedule, igso3 = _schedule_and_igso3()
        torch.manual_seed(0)
        backbone = torch.randn(2, 5, 3, 3)
        R0, T0 = build_backbone_frames(backbone)
        t = torch.tensor([9, 9])
        fixed = torch.zeros(2, 5, dtype=torch.bool)
        fixed[:, :2] = True
        Rt, Tt = sample_noisy_frames(igso3,
                                     schedule,
                                     R0,
                                     T0,
                                     t,
                                     fixed_mask=fixed)
        assert torch.allclose(Rt[:, :2], R0[:, :2])
        assert torch.allclose(Tt[:, :2], T0[:, :2])

    def test_batch_size_mismatch_raises(self):
        schedule, igso3 = _schedule_and_igso3()
        R0, T0 = make_identity_rigid((2, 4))
        t = torch.tensor([1, 2, 3])
        with pytest.raises(ValueError):
            sample_noisy_frames(igso3, schedule, R0, T0, t)


@pytest.mark.torch
@requires_torch
class TestReverseSteps:

    def test_translation_posterior_step_shape(self):
        schedule, _ = _schedule_and_igso3()
        x0 = torch.zeros(2, 4, 3)
        xt = torch.randn(2, 4, 3)
        out = translation_posterior_step(schedule, x0, xt, t=5)
        assert out.shape == (2, 4, 3)
        assert torch.isfinite(out).all()

    def test_translation_posterior_step_t0_is_deterministic(self):
        schedule, _ = _schedule_and_igso3()
        x0 = torch.randn(2, 4, 3)
        xt = torch.randn(2, 4, 3)
        out1 = translation_posterior_step(schedule, x0, xt, t=0)
        out2 = translation_posterior_step(schedule, x0, xt, t=0)
        assert torch.allclose(out1, out2)

    def test_translation_posterior_step_negative_t_raises(self):
        schedule, _ = _schedule_and_igso3()
        x0 = torch.zeros(1, 2, 3)
        xt = torch.zeros(1, 2, 3)
        with pytest.raises(ValueError):
            translation_posterior_step(schedule, x0, xt, t=-1)

    def test_so3_reverse_step_output_is_valid_rotation(self):
        _, igso3 = _schedule_and_igso3()
        Rt, _ = make_identity_rigid((2, 3))
        R0, _ = make_identity_rigid((2, 3))
        out = so3_x0_reverse_step(igso3, Rt, R0, t=5)
        det = torch.linalg.det(out)
        assert torch.allclose(det, torch.ones_like(det), atol=1e-3)

    def test_so3_reverse_step_t0_raises(self):
        _, igso3 = _schedule_and_igso3()
        Rt, _ = make_identity_rigid((1, 2))
        R0, _ = make_identity_rigid((1, 2))
        with pytest.raises(ValueError):
            so3_x0_reverse_step(igso3, Rt, R0, t=0)


@pytest.mark.torch
@requires_torch
class TestMultitrackFrameLoss:

    def test_zero_loss_for_perfect_prediction(self):
        R, t = make_identity_rigid((2, 4))
        mask = torch.ones(2, 4)
        loss = multitrack_frame_loss([R, t], [R, t], [mask])
        assert round(float(loss), 5) == 0.0

    def test_masked_positions_do_not_contribute(self):
        torch.manual_seed(0)
        R0, T0 = make_identity_rigid((1, 3))
        R_pred = torch.randn(1, 3, 3, 3)  # garbage prediction
        T_pred = torch.randn(1, 3, 3)
        mask = torch.zeros(1, 3)  # everything masked out
        loss = multitrack_frame_loss([R_pred, T_pred], [R0, T0], [mask])
        assert round(float(loss), 5) == 0.0

    def test_rotation_weight_scales_rotation_term(self):
        torch.manual_seed(1)
        backbone = torch.randn(1, 3, 3, 3)
        R0, T0 = build_backbone_frames(backbone)
        R_pred, _ = make_identity_rigid((1, 3))
        mask = torch.ones(1, 3)
        loss_low = multitrack_frame_loss([R_pred, T0], [R0, T0], [mask],
                                         rotation_weight=0.1)
        loss_high = multitrack_frame_loss([R_pred, T0], [R0, T0], [mask],
                                          rotation_weight=10.0)
        assert loss_high > loss_low

    def test_gradients_do_not_explode_near_identity(self):
        # Regression test for the so3_log_map sqrt(0)-gradient bug: a
        # denoiser output that has converged very close to the target
        # should not blow up into NaN gradients.
        torch.manual_seed(0)
        denoiser = RFDiffusionMultiTrackDenoiser(embed_dim=16,
                                                 pair_dim=8,
                                                 num_blocks=1,
                                                 num_heads=2,
                                                 pair_num_heads=2)
        R0, T0 = make_identity_rigid((2, 4))
        mask = torch.ones(2, 4)
        noisy_coords = backbone_coords_from_frames(R0, T0).reshape(2, 4, 9)
        t = torch.tensor([1, 1])
        pred_R, pred_T = denoiser([noisy_coords, R0, T0, t, mask])
        loss = multitrack_frame_loss([pred_R, pred_T], [R0, T0], [mask])
        loss.backward()
        for p in denoiser.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all()


@pytest.mark.torch
@requires_torch
class TestRFDiffusionMultiTrackDenoiser:

    def _denoiser(self, **kw):
        defaults = dict(embed_dim=16,
                        pair_dim=8,
                        num_blocks=1,
                        num_heads=2,
                        pair_num_heads=2)
        defaults.update(kw)
        return RFDiffusionMultiTrackDenoiser(**defaults)

    def test_output_shapes(self):
        denoiser = self._denoiser()
        R, t_trans = make_identity_rigid((2, 5))
        noisy_coords = backbone_coords_from_frames(R, t_trans).reshape(2, 5, 9)
        t = torch.tensor([0, 3])
        mask = torch.ones(2, 5)
        pred_R, pred_T = denoiser([noisy_coords, R, t_trans, t, mask])
        assert pred_R.shape == (2, 5, 3, 3)
        assert pred_T.shape == (2, 5, 3)

    def test_finite_without_mask(self):
        denoiser = self._denoiser()
        R, t_trans = make_identity_rigid((1, 4))
        noisy_coords = backbone_coords_from_frames(R, t_trans).reshape(1, 4, 9)
        t = torch.tensor([2])
        pred_R, pred_T = denoiser([noisy_coords, R, t_trans, t, None])
        assert torch.isfinite(pred_R).all()
        assert torch.isfinite(pred_T).all()

    def test_invalid_num_blocks_raises(self):
        with pytest.raises(ValueError):
            self._denoiser(num_blocks=0)
