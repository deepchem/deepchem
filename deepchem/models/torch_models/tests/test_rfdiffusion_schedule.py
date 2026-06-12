"""Tests for the RFDiffusion cosine noise schedule."""

import pytest

from deepchem.models.torch_models.rfdiffusion import CosineSchedule

try:
    import torch
    import torch.nn.functional as F
    has_torch = True
except ImportError:
    has_torch = False


@pytest.mark.torch
@pytest.mark.skipif(not has_torch, reason="PyTorch not installed")
class TestCosineSchedule:

    def test_q_sample_shape(self):
        schedule = CosineSchedule(num_timesteps=100)
        x0 = torch.randn(2, 10, 9)
        t = torch.tensor([0, 50])
        noisy_x, noise = schedule.q_sample(x0, t)
        assert noisy_x.shape == x0.shape
        assert noise.shape == x0.shape

    def test_no_noise_at_t0(self):
        schedule = CosineSchedule(num_timesteps=1000)
        x0 = torch.randn(1, 10, 9)
        t = torch.tensor([0])
        noisy_x, _ = schedule.q_sample(x0, t)
        # alpha_cumprod[0] is close to 1 so noisy should be close to clean
        assert torch.allclose(noisy_x, x0, atol=0.05)

    def test_mostly_noise_at_tmax(self):
        schedule = CosineSchedule(num_timesteps=1000)
        x0 = torch.ones(1, 10, 9)
        t = torch.tensor([999])
        noisy_x, noise = schedule.q_sample(x0, t)
        # alpha_cumprod is near 0 at t=T so output should track the noise
        corr = F.cosine_similarity(noisy_x.flatten().unsqueeze(0),
                                   noise.flatten().unsqueeze(0))
        assert corr.item() > 0.9

    def test_alpha_cumprod_monotonic(self):
        schedule = CosineSchedule(num_timesteps=1000)
        diffs = schedule.alpha_cumprod[1:] - schedule.alpha_cumprod[:-1]
        assert (diffs <= 0).all()

    def test_betas_bounded(self):
        schedule = CosineSchedule(num_timesteps=1000)
        assert (schedule.betas >= 0).all()
        assert (schedule.betas <= 1).all()

    def test_q_sample_uses_provided_noise(self):
        schedule = CosineSchedule(num_timesteps=100)
        x0 = torch.randn(2, 5, 9)
        t = torch.tensor([50, 50])
        fixed_noise = torch.ones_like(x0)
        noisy_x, returned_noise = schedule.q_sample(x0, t, noise=fixed_noise)
        assert torch.equal(returned_noise, fixed_noise)

    def test_buffers_included_in_state_dict(self):
        schedule = CosineSchedule(num_timesteps=100)
        sd = schedule.state_dict()
        assert 'betas' in sd
        assert 'alpha_cumprod' in sd
        assert 'sqrt_alpha_cumprod' in sd

    def test_p_sample_shape(self):
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
        schedule = CosineSchedule(num_timesteps=1000)
        assert torch.isfinite(schedule.alpha_cumprod).all()
        assert torch.isfinite(schedule.betas).all()
        assert torch.isfinite(schedule.posterior_variance).all()
