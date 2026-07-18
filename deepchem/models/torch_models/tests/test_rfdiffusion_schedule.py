"""Tests for the RFDiffusion cosine noise schedule."""

import pytest

try:
    import torch
    import torch.nn.functional as F
    from deepchem.models.torch_models.rfdiffusion import CosineSchedule
    has_torch = True
except ImportError:
    has_torch = False


@pytest.mark.torch
@pytest.mark.skipif(not has_torch, reason="PyTorch not installed")
class TestCosineSchedule:

    def test_q_sample_shape(self):
        """q_sample returns noisy data and noise shaped like the input."""
        schedule = CosineSchedule(num_timesteps=100)
        x0 = torch.randn(2, 10, 9)
        t = torch.tensor([0, 50])
        noisy_x, noise = schedule.q_sample(x0, t)
        assert noisy_x.shape == x0.shape
        assert noise.shape == x0.shape

    def test_no_noise_at_t0(self):
        """At t=0 almost no noise is added, so the output stays near x0."""
        schedule = CosineSchedule(num_timesteps=1000)
        x0 = torch.randn(1, 10, 9)
        t = torch.tensor([0])
        noisy_x, _ = schedule.q_sample(x0, t)
        # alpha_cumprod[0] is close to 1 so noisy should be close to clean
        assert torch.allclose(noisy_x, x0, atol=0.05)

    def test_mostly_noise_at_tmax(self):
        """At the final timestep the output is dominated by the noise."""
        schedule = CosineSchedule(num_timesteps=1000)
        x0 = torch.ones(1, 10, 9)
        t = torch.tensor([999])
        noisy_x, noise = schedule.q_sample(x0, t)
        # alpha_cumprod is near 0 at t=T so output should track the noise
        corr = F.cosine_similarity(noisy_x.flatten().unsqueeze(0),
                                   noise.flatten().unsqueeze(0))
        assert corr.item() > 0.9

    def test_alpha_cumprod_monotonic(self):
        """alpha_cumprod decreases monotonically from ~1 toward 0."""
        schedule = CosineSchedule(num_timesteps=1000)
        diffs = schedule.alpha_cumprod[1:] - schedule.alpha_cumprod[:-1]
        assert (diffs <= 0).all()

    def test_betas_bounded(self):
        """Every beta stays within [0, 1]."""
        schedule = CosineSchedule(num_timesteps=1000)
        assert (schedule.betas >= 0).all()
        assert (schedule.betas <= 1).all()

    def test_q_sample_uses_provided_noise(self):
        """q_sample uses a caller-supplied noise tensor when given one."""
        schedule = CosineSchedule(num_timesteps=100)
        x0 = torch.randn(2, 5, 9)
        t = torch.tensor([50, 50])
        fixed_noise = torch.ones_like(x0)
        noisy_x, returned_noise = schedule.q_sample(x0, t, noise=fixed_noise)
        assert torch.equal(returned_noise, fixed_noise)

    def test_schedule_tensors_available_as_fields(self):
        """The schedule tensors are exposed as plain module attributes."""
        schedule = CosineSchedule(num_timesteps=100)
        assert isinstance(schedule.betas, torch.Tensor)
        assert isinstance(schedule.alpha_cumprod, torch.Tensor)
        assert isinstance(schedule.sqrt_alpha_cumprod, torch.Tensor)

    def test_p_sample_shape(self):
        """One reverse step returns a tensor matching the input shape."""

        # Use a trivial model that always predicts zero noise
        class ZeroModel(torch.nn.Module):

            def forward(self, inputs):
                return torch.zeros_like(inputs[0])

        schedule = CosineSchedule(num_timesteps=10)
        x_t = torch.randn(2, 5, 9)
        t = torch.tensor([5, 5])
        x_prev = schedule.p_sample(ZeroModel(), x_t, t)
        assert x_prev.shape == x_t.shape

    def test_schedule_produces_finite_values(self):
        """All schedule tensors are finite (no NaN or inf from the math)."""
        schedule = CosineSchedule(num_timesteps=1000)
        assert torch.isfinite(schedule.alpha_cumprod).all()
        assert torch.isfinite(schedule.betas).all()
        assert torch.isfinite(schedule.posterior_variance).all()

    def test_matches_improved_ddpm_reference(self):
        """betas match the Improved DDPM cosine formula.

        The schedule depends only on the timestep, not on the data, so this
        checks our values against the reference formula from
        openai/improved-diffusion rather than against any dataset.
        """
        import math
        num_timesteps = 1000
        schedule = CosineSchedule(num_timesteps=num_timesteps)

        def alpha_bar(step):
            return math.cos((step + 0.008) / 1.008 * math.pi / 2)**2

        ref_betas = []
        for i in range(num_timesteps):
            t1 = i / num_timesteps
            t2 = (i + 1) / num_timesteps
            ref_betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), 0.999))
        expected = torch.tensor(ref_betas, dtype=schedule.betas.dtype)
        assert torch.allclose(schedule.betas, expected, atol=1e-4)
