"""Tests for RFDiffusion All-Atom conditioning modules."""

import pytest

try:
    import torch
    from deepchem.models.torch_models.rfdiffusion_conditioning import (
        BinderCrossAttention, LengthConditioning)
    has_torch = True
except ImportError:
    has_torch = False

requires_torch = pytest.mark.skipif(not has_torch,
                                    reason='PyTorch not installed')


@pytest.mark.torch
@requires_torch
class TestLengthConditioning:

    def test_output_shape(self):
        cond = LengthConditioning(embed_dim=16, max_length=200)
        lengths = torch.tensor([10, 50, 200])
        out = cond(lengths)
        assert out.shape == (3, 16)

    def test_different_lengths_give_different_embeddings(self):
        cond = LengthConditioning(embed_dim=16, max_length=200)
        out = cond(torch.tensor([10, 190]))
        assert not torch.allclose(out[0], out[1])

    def test_same_length_gives_same_embedding(self):
        cond = LengthConditioning(embed_dim=16, max_length=200)
        out = cond(torch.tensor([50, 50]))
        assert torch.allclose(out[0], out[1])

    def test_accepts_float_lengths(self):
        cond = LengthConditioning(embed_dim=8, max_length=100)
        out = cond(torch.tensor([25.0, 75.0]))
        assert out.shape == (2, 8)

    def test_invalid_embed_dim_raises(self):
        with pytest.raises(ValueError):
            LengthConditioning(embed_dim=0)

    def test_invalid_max_length_raises(self):
        with pytest.raises(ValueError):
            LengthConditioning(embed_dim=8, max_length=0)

    def test_gradient_flows(self):
        cond = LengthConditioning(embed_dim=8, max_length=100)
        out = cond(torch.tensor([30.0]))
        out.sum().backward()
        grads = [p.grad for p in cond.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in grads)


@pytest.mark.torch
@requires_torch
class TestBinderCrossAttention:

    def test_output_shape(self):
        attn = BinderCrossAttention(embed_dim=32, num_heads=4)
        query = torch.randn(2, 10, 32)
        partner = torch.randn(2, 15, 32)
        out = attn(query, partner)
        assert out.shape == (2, 10, 32)

    def test_finite_output(self):
        attn = BinderCrossAttention(embed_dim=16, num_heads=2)
        query = torch.randn(1, 5, 16)
        partner = torch.randn(1, 8, 16)
        out = attn(query, partner)
        assert torch.isfinite(out).all()

    def test_masked_partner_positions_are_ignored(self):
        torch.manual_seed(0)
        attn = BinderCrossAttention(embed_dim=16, num_heads=2)
        attn.eval()
        query = torch.randn(1, 3, 16)
        partner = torch.randn(1, 4, 16)
        mask_all_valid = torch.ones(1, 4, dtype=torch.bool)

        # Change a masked-out position: output should not change.
        mask_drop_last = mask_all_valid.clone()
        mask_drop_last[0, -1] = False

        partner_changed = partner.clone()
        partner_changed[0, -1] += 100.0  # perturb only the masked position

        out_before = attn(query, partner, partner_mask=mask_drop_last)
        out_after = attn(query, partner_changed, partner_mask=mask_drop_last)
        assert torch.allclose(out_before, out_after, atol=1e-5)

    def test_embed_dim_not_divisible_by_heads_raises(self):
        with pytest.raises(ValueError):
            BinderCrossAttention(embed_dim=10, num_heads=3)

    def test_invalid_embed_dim_raises(self):
        with pytest.raises(ValueError):
            BinderCrossAttention(embed_dim=0, num_heads=1)

    def test_invalid_num_heads_raises(self):
        with pytest.raises(ValueError):
            BinderCrossAttention(embed_dim=8, num_heads=0)

    def test_gradient_flows(self):
        attn = BinderCrossAttention(embed_dim=16, num_heads=4)
        query = torch.randn(1, 3, 16, requires_grad=True)
        partner = torch.randn(1, 5, 16)
        out = attn(query, partner)
        out.sum().backward()
        assert query.grad is not None
        assert torch.isfinite(query.grad).all()
