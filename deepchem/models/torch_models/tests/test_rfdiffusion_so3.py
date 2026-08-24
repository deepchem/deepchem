"""Tests for the RFDiffusion SO(3) diffusion utilities."""

import math

import pytest

try:
    import torch
    from deepchem.models.torch_models.rfdiffusion_so3 import (
        IGSO3,
        log_beta_schedule,
        so3_exp_map,
        so3_log_map,
        so3_reverse_step,
    )
    has_torch = True
except ImportError:
    has_torch = False


@pytest.mark.torch
@pytest.mark.skipif(not has_torch, reason="PyTorch not installed")
class TestSO3Maps:
    """Exponential and logarithm maps between so(3) and SO(3)."""

    def test_exp_zero_is_identity(self):
        """A zero tangent vector maps to the identity rotation."""
        rotations = so3_exp_map(torch.zeros(4, 3))
        assert torch.allclose(rotations,
                              torch.eye(3).expand(4, 3, 3),
                              atol=1e-6)

    def test_exp_is_valid_rotation(self):
        """Exp-map outputs are orthogonal with determinant +1."""
        torch.manual_seed(0)
        rotations = so3_exp_map(torch.randn(16, 3))
        eye = torch.eye(3).expand(16, 3, 3)
        assert torch.allclose(rotations @ rotations.transpose(-1, -2),
                              eye,
                              atol=1e-5)
        assert torch.allclose(torch.det(rotations), torch.ones(16), atol=1e-5)

    def test_exp_known_rotation_about_z(self):
        """A pi/2 rotation about z matches the closed-form matrix."""
        rotation = so3_exp_map(torch.tensor([[0.0, 0.0, math.pi / 2]]))[0]
        expected = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0],
                                 [0.0, 0.0, 1.0]])
        assert torch.allclose(rotation, expected, atol=1e-6)

    def test_log_is_inverse_of_exp(self):
        """log(exp(tau)) recovers tau for angles across (0, pi)."""
        torch.manual_seed(1)
        axis = torch.randn(32, 3)
        axis = axis / axis.norm(dim=-1, keepdim=True)
        for angle in [0.05, 1.0, 2.5, math.pi - 0.05]:
            tangent = angle * axis
            recovered = so3_log_map(so3_exp_map(tangent))
            assert torch.allclose(recovered, tangent, atol=1e-3)

    def test_log_near_pi_is_stable(self):
        """The log map stays accurate for rotations very close to pi.

        The quaternion route keeps the error small where reading the angle
        from the trace would lose precision (``sin(omega) -> 0``).
        """
        axis = torch.tensor([[0.3, -0.5, 0.8]])
        axis = axis / axis.norm(dim=-1, keepdim=True)
        rotation = so3_exp_map((math.pi - 1e-3) * axis)
        # Compare rotations, avoiding the axis-sign ambiguity at pi.
        recovered = so3_exp_map(so3_log_map(rotation))
        assert torch.allclose(rotation, recovered, atol=1e-3)

    def test_log_identity_is_zero(self):
        """The log of the identity rotation is the zero tangent vector."""
        identity = torch.eye(3).unsqueeze(0)
        assert torch.allclose(so3_log_map(identity),
                              torch.zeros(1, 3),
                              atol=1e-6)


@pytest.mark.torch
@pytest.mark.skipif(not has_torch, reason="PyTorch not installed")
class TestLogBetaSchedule:
    """Logarithmic sigma schedule for rotational diffusion."""

    def test_shape_and_endpoints(self):
        """The schedule has the requested length and hits both endpoints."""
        sigmas = log_beta_schedule(10, beta_min=0.1, beta_max=1.5)
        assert sigmas.shape == (10,)
        assert math.isclose(sigmas[0].item(), 0.1, rel_tol=1e-6)
        assert math.isclose(sigmas[-1].item(), 1.5, rel_tol=1e-6)

    def test_monotonic_increasing(self):
        """Sigma increases monotonically with the step index."""
        sigmas = log_beta_schedule(50)
        assert bool((sigmas[1:] > sigmas[:-1]).all())

    def test_linear_in_log_space(self):
        """Successive log-sigma differences are constant."""
        sigmas = log_beta_schedule(7, beta_min=0.2, beta_max=2.0)
        diffs = torch.log(sigmas)[1:] - torch.log(sigmas)[:-1]
        assert torch.allclose(diffs, diffs[0].expand_as(diffs), atol=1e-6)

    def test_invalid_args_raise(self):
        """Bad step counts and beta ranges raise ValueError."""
        with pytest.raises(ValueError):
            log_beta_schedule(1)
        with pytest.raises(ValueError):
            log_beta_schedule(10, beta_min=1.0, beta_max=0.5)


@pytest.mark.torch
@pytest.mark.skipif(not has_torch, reason="PyTorch not installed")
class TestIGSO3:
    """IGSO(3) density, score, and sampling."""

    def test_pdf_normalised(self):
        """The discretised marginal density integrates to approximately 1."""
        dist = IGSO3(torch.tensor([0.3, 0.8, 1.5]), num_omega=512)
        for index in range(3):
            assert dist.normalisation_error(index) < 1e-2

    def test_cdf_monotonic_and_bounded(self):
        """The CDF increases monotonically and reaches ~1 at pi."""
        dist = IGSO3(torch.tensor([0.7]), num_omega=512)
        cdf = dist._cdf[0]
        assert bool((cdf[1:] >= cdf[:-1] - 1e-9).all())
        assert abs(cdf[-1].item() - 1.0) < 1e-2

    def test_score_matches_finite_difference(self):
        """The autograd score equals a finite-difference of log f(omega)."""
        sigma = torch.tensor([0.9]).double()
        lmax = 500
        omega = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64)
        analytic = IGSO3._score_omega(omega, sigma, lmax)[0]
        eps = 1e-4
        fp = torch.log(IGSO3._f_omega(omega + eps, sigma, lmax))[0]
        fm = torch.log(IGSO3._f_omega(omega - eps, sigma, lmax))[0]
        finite_diff = (fp - fm) / (2 * eps)
        assert torch.allclose(analytic, finite_diff, atol=1e-3)

    def test_sample_shape_and_valid(self):
        """Samples have the requested batch shape and are valid rotations."""
        dist = IGSO3(torch.tensor([0.6]), num_omega=256)
        generator = torch.Generator().manual_seed(0)
        rotations = dist.sample(0, (5, 4), generator=generator)
        assert rotations.shape == (5, 4, 3, 3)
        eye = torch.eye(3).expand(5, 4, 3, 3)
        assert torch.allclose(rotations @ rotations.transpose(-1, -2),
                              eye,
                              atol=1e-4)

    def test_sample_angle_in_range(self):
        """Sampled rotation angles lie within (0, pi]."""
        dist = IGSO3(torch.tensor([1.0]), num_omega=256)
        generator = torch.Generator().manual_seed(0)
        angles = dist.sample_angle(0, (1000,), generator=generator)
        assert bool((angles > 0).all())
        assert bool((angles <= math.pi + 1e-6).all())

    def test_larger_sigma_gives_larger_mean_angle(self):
        """A larger sigma produces a larger mean sampled angle."""
        dist = IGSO3(torch.tensor([0.3, 1.5]), num_omega=512)
        generator = torch.Generator().manual_seed(0)
        small = dist.sample_angle(0, (4000,), generator=generator).mean()
        large = dist.sample_angle(1, (4000,), generator=generator).mean()
        assert large > small

    def test_invalid_construction_raises(self):
        """Invalid sigmas or grid sizes raise ValueError."""
        with pytest.raises(ValueError):
            IGSO3(torch.tensor([[0.5]]))
        with pytest.raises(ValueError):
            IGSO3(torch.tensor([-0.5]))
        with pytest.raises(ValueError):
            IGSO3(torch.tensor([0.5]), num_omega=8)


@pytest.mark.torch
@pytest.mark.skipif(not has_torch, reason="PyTorch not installed")
class TestSO3ReverseStep:
    """Reverse-time Euler-Maruyama integrator on SO(3)."""

    def test_output_shape_and_valid(self):
        """A reverse step returns valid rotations of the input shape."""
        torch.manual_seed(0)
        rotations = so3_exp_map(torch.randn(6, 3))
        score = torch.full((6,), -0.5)
        updated = so3_reverse_step(rotations,
                                   score,
                                   1.0,
                                   0.9,
                                   noise=torch.zeros(6, 3))
        assert updated.shape == (6, 3, 3)
        eye = torch.eye(3).expand(6, 3, 3)
        assert torch.allclose(updated @ updated.transpose(-1, -2),
                              eye,
                              atol=1e-5)

    def test_deterministic_without_noise(self):
        """With zero noise the reverse step is deterministic."""
        rotations = so3_exp_map(torch.tensor([[0.0, 0.0, 2.0]]))
        score = torch.tensor([-1.0])
        first = so3_reverse_step(rotations,
                                 score,
                                 1.0,
                                 0.8,
                                 noise=torch.zeros(1, 3))
        second = so3_reverse_step(rotations,
                                  score,
                                  1.0,
                                  0.8,
                                  noise=torch.zeros(1, 3))
        assert torch.allclose(first, second)

    def test_invalid_sigmas_raise(self):
        """Non-decreasing or non-positive sigma values raise ValueError."""
        rotations = so3_exp_map(torch.zeros(1, 3))
        score = torch.tensor([0.0])
        with pytest.raises(ValueError):
            so3_reverse_step(rotations, score, 1.0, 1.0)
        with pytest.raises(ValueError):
            so3_reverse_step(rotations, score, -1.0, 0.5)
