"""Tests for the RFDiffusion backbone denoiser."""

import pytest

try:
    import torch
    from deepchem.models.torch_models.rfdiffusion import (
        BackboneDiffusion,
        DiffusionTransformerBlock,
    )
    has_torch = True
except ImportError:
    has_torch = False


@pytest.mark.torch
@pytest.mark.skipif(not has_torch, reason="PyTorch not installed")
class TestDiffusionTransformerBlock:

    def test_output_shape(self):
        block = DiffusionTransformerBlock(128, num_heads=4)
        x = torch.randn(2, 50, 128)
        t_emb = torch.randn(2, 128)
        assert block(x, t_emb).shape == (2, 50, 128)

    def test_time_conditioning_changes_output(self):
        block = DiffusionTransformerBlock(64, num_heads=4)
        block.eval()
        x = torch.randn(1, 10, 64)
        out1 = block(x, torch.randn(1, 64))
        out2 = block(x, torch.randn(1, 64) * 5)
        assert not torch.allclose(out1, out2)

    def test_gradient_flows(self):
        block = DiffusionTransformerBlock(32, num_heads=4)
        x = torch.randn(1, 5, 32, requires_grad=True)
        t_emb = torch.randn(1, 32)
        block(x, t_emb).sum().backward()
        assert x.grad is not None


@pytest.mark.torch
@pytest.mark.skipif(not has_torch, reason="PyTorch not installed")
class TestBackboneDiffusion:

    def test_output_shape(self):
        model = BackboneDiffusion(coord_dim=9,
                                  embed_dim=64,
                                  num_layers=2,
                                  num_heads=4)
        x = torch.randn(2, 20, 9)
        t = torch.randint(0, 100, (2,))
        assert model([x, t]).shape == (2, 20, 9)

    def test_variable_lengths(self):
        model = BackboneDiffusion(coord_dim=9,
                                  embed_dim=64,
                                  num_layers=2,
                                  num_heads=4)
        for length in [5, 20, 100]:
            x = torch.randn(1, length, 9)
            t = torch.tensor([100])
            assert model([x, t]).shape == (1, length, 9)

    def test_zero_initialized_output(self):
        model = BackboneDiffusion(coord_dim=9,
                                  embed_dim=64,
                                  num_layers=2,
                                  num_heads=4)
        x = torch.randn(1, 10, 9)
        t = torch.tensor([500])
        # output should be small right after init due to zero-init convention
        assert model([x, t]).abs().mean().item() < 1.0

    def test_gradient_flows_through_model(self):
        model = BackboneDiffusion(coord_dim=9,
                                  embed_dim=64,
                                  num_layers=2,
                                  num_heads=4)
        x = torch.randn(2, 10, 9, requires_grad=True)
        t = torch.randint(0, 100, (2,))
        model([x, t]).sum().backward()
        assert x.grad is not None

    def test_different_timesteps_give_different_output(self):
        model = BackboneDiffusion(coord_dim=9,
                                  embed_dim=64,
                                  num_layers=2,
                                  num_heads=4)
        # one gradient step breaks zero-init so outputs differ by timestep
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        x = torch.randn(1, 10, 9)
        pred = model([x, torch.tensor([500])])
        pred.sum().backward()
        opt.step()
        model.eval()
        out_early = model([x, torch.tensor([10])])
        out_late = model([x, torch.tensor([900])])
        assert not torch.allclose(out_early, out_late)
