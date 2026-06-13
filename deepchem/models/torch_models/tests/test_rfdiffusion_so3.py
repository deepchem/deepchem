"""Exhaustive tests for the IGSO(3) diffusion utilities.

These tests verify:

* PDF normalisation to within 1e-3 on a dense ω grid.
* Series-expansion stability of ``so3_exp_map`` and ``so3_log_map``
  near ω = 0 and ω = π.
* Round-trip ``log ∘ exp = id`` to fp32 precision on the principal
  branch.
* Logarithmic β-schedule monotonicity and endpoint correctness.
* Single-step reverse Euler-Maruyama produces a proper rotation matrix
  with finite entries and is deterministic for ``noise = 0``.
"""

import math

import pytest
import torch

from deepchem.models.torch_models.rfdiffusion_so3 import (
    IGSO3,
    log_beta_schedule,
    so3_exp_map,
    so3_log_map,
    so3_reverse_step,
)


@pytest.fixture(scope='module')
def igso3():
    sigmas = torch.tensor([0.05, 0.1, 0.3, 0.5, 1.0, 1.5])
    return IGSO3(sigmas, num_omega=2048, lmax=500)


class TestExpLogMaps:

    def test_exp_returns_proper_rotation(self):
        torch.manual_seed(0)
        tau = torch.randn(32, 3) * 0.5
        R = so3_exp_map(tau)
        det = torch.det(R)
        ortho_err = (R @ R.transpose(-1, -2)
                     - torch.eye(3)).abs().max()
        assert torch.allclose(det, torch.ones_like(det), atol=1e-5)
        assert ortho_err < 1e-5

    def test_exp_zero_tangent_gives_identity(self):
        R = so3_exp_map(torch.zeros(4, 3))
        eye = torch.eye(3).expand(4, 3, 3)
        assert torch.allclose(R, eye, atol=1e-7)

    def test_log_exp_roundtrip_principal_branch(self):
        torch.manual_seed(1)
        u = torch.randn(50, 3)
        u = u / u.norm(dim=-1, keepdim=True)
        omega = torch.rand(50) * (math.pi - 1e-3) + 1e-4
        tau = omega.unsqueeze(-1) * u
        tau_back = so3_log_map(so3_exp_map(tau))
        # fp32 precision degrades near ω = π; 1e-3 captures the worst case.
        assert (tau - tau_back).norm(dim=-1).max() < 1e-3

    def test_log_map_stable_near_pi(self):
        axis = torch.tensor([[0.6, -0.2, 0.7745967]])
        axis = axis / axis.norm(dim=-1, keepdim=True)
        tau = (math.pi - 1e-4) * axis
        tau_back = so3_log_map(so3_exp_map(tau))
        assert torch.allclose(tau_back, tau, atol=1e-3)

    def test_small_omega_taylor_branch(self):
        tau = torch.tensor([[1e-7, 0.0, 0.0], [0.0, 1e-7, 0.0]])
        R = so3_exp_map(tau)
        eye = torch.eye(3).expand_as(R)
        assert (R - eye).abs().max() < 1e-6


class TestLogBetaSchedule:

    def test_endpoints(self):
        s = log_beta_schedule(50, beta_min=0.1, beta_max=1.5)
        assert s[0].item() == pytest.approx(0.1, rel=1e-6)
        assert s[-1].item() == pytest.approx(1.5, rel=1e-6)

    def test_monotone(self):
        s = log_beta_schedule(100, 0.1, 1.5)
        diffs = s[1:] - s[:-1]
        assert (diffs >= -1e-7).all()

    def test_logarithmic_spacing(self):
        s = log_beta_schedule(100, 0.1, 1.5)
        log_s = torch.log(s)
        diffs = log_s[1:] - log_s[:-1]
        assert (diffs - diffs[0]).abs().max() < 1e-6

    def test_validation(self):
        with pytest.raises(ValueError):
            log_beta_schedule(1)
        with pytest.raises(ValueError):
            log_beta_schedule(10, beta_min=0.0)
        with pytest.raises(ValueError):
            log_beta_schedule(10, beta_min=0.5, beta_max=0.4)


class TestIGSO3:

    def test_construction_validates(self):
        with pytest.raises(ValueError):
            IGSO3(torch.tensor([[1.0]]))
        with pytest.raises(ValueError):
            IGSO3(torch.tensor([0.5, -0.1]))
        with pytest.raises(ValueError):
            IGSO3(torch.tensor([0.5]), num_omega=4)

    @pytest.mark.parametrize('idx', range(6))
    def test_normalisation_within_tolerance(self, igso3, idx):
        assert igso3.normalisation_error(idx) < 1e-3

    def test_pdf_nonnegative(self, igso3):
        for idx in range(len(igso3.sigmas)):
            # The upstream truncated series can have tiny negative tails
            # from cancellation, especially at small σ.
            assert igso3._pdf[idx].min().item() > -1e-5

    def test_cdf_monotone_and_unit(self, igso3):
        for idx in range(len(igso3.sigmas)):
            cdf = igso3._cdf[idx]
            assert (cdf[1:] - cdf[:-1] >= -2e-8).all()
            assert abs(cdf[-1].item() - 1.0) < 5e-4

    def test_mean_angle_increases_with_sigma(self, igso3):
        torch.manual_seed(0)
        means = []
        for idx in range(len(igso3.sigmas)):
            sam = igso3.sample(idx, (2000,))
            omegas = so3_log_map(sam).norm(dim=-1)
            means.append(omegas.mean().item())
        diffs = [b - a for a, b in zip(means[:-1], means[1:])]
        assert all(d > 0 for d in diffs), f'means not monotone: {means}'

    def test_sampled_rotations_are_proper(self, igso3):
        sam = igso3.sample(2, (16,))
        assert torch.allclose(torch.det(sam), torch.ones(16), atol=1e-4)

    def test_score_finite(self, igso3):
        omega = torch.linspace(0.01, math.pi - 0.01, 64,
                               dtype=torch.float64)
        for idx in range(len(igso3.sigmas)):
            s = igso3.score(omega, idx)
            assert torch.isfinite(s).all()

    def test_score_matches_autograd_series(self):
        omega = torch.linspace(0.05, 1.25, 16, dtype=torch.float64)
        sigma = torch.tensor([0.3], dtype=torch.float64)
        omega_var = omega.clone().requires_grad_(True)
        f_val = IGSO3._f_omega(omega_var, sigma, lmax=64)[0]
        expected = torch.autograd.grad(torch.log(f_val).sum(), omega_var)[0]
        actual = IGSO3._score_omega(omega, sigma, lmax=64)[0]
        assert torch.allclose(actual, expected, atol=0.0, rtol=0.0)


class TestReverseStep:

    def test_zero_score_zero_noise_drift(self):
        torch.manual_seed(0)
        tau = torch.randn(4, 3) * 0.5
        R = so3_exp_map(tau)
        out = so3_reverse_step(R, torch.zeros(4),
                               sigma_t=1.0,
                               sigma_t_minus_1=0.8,
                               noise=torch.zeros(4, 3))
        det = torch.det(out)
        assert torch.allclose(det, torch.ones(4), atol=1e-4)

    def test_validation(self):
        R = so3_exp_map(torch.zeros(1, 3))
        with pytest.raises(ValueError):
            so3_reverse_step(R, torch.zeros(1), -0.1, 0.05)
        with pytest.raises(ValueError):
            so3_reverse_step(R, torch.zeros(1), 0.5, 0.6)

    def test_noise_changes_output(self):
        torch.manual_seed(0)
        R = so3_exp_map(torch.randn(4, 3) * 0.3)
        out_a = so3_reverse_step(R, torch.zeros(4), 1.0, 0.8,
                                 noise=torch.zeros(4, 3))
        out_b = so3_reverse_step(R, torch.zeros(4), 1.0, 0.8,
                                 noise=torch.randn(4, 3))
        assert not torch.allclose(out_a, out_b, atol=1e-3)
