"""Tests for binder cross-attention and length conditioning."""

import math

import pytest
import torch

from deepchem.models.torch_models.rfdiffusion_conditioning import (
    BinderCrossAttention,
    LengthConditioning,
    sinusoidal_length_embedding,
)


class TestLengthEmbedding:

    def test_shape_and_dtype(self):
        lengths = torch.tensor([10, 25, 64, 128])
        emb = sinusoidal_length_embedding(lengths, embed_dim=32)
        assert emb.shape == (4, 32)
        assert emb.dtype == torch.float32

    def test_distinct_lengths_distinct_embeddings(self):
        lengths = torch.tensor([10, 11])
        emb = sinusoidal_length_embedding(lengths, embed_dim=64)
        assert not torch.allclose(emb[0], emb[1])

    def test_odd_embed_dim_rejected(self):
        with pytest.raises(ValueError):
            sinusoidal_length_embedding(torch.tensor([1]), embed_dim=31)


class TestLengthConditioning:

    def test_output_shape(self):
        m = LengthConditioning(time_dim=64, embed_dim=32)
        out = m(torch.tensor([10, 20, 30]))
        assert out.shape == (3, 64)

    def test_distinct_lengths_distinct_outputs(self):
        torch.manual_seed(0)
        m = LengthConditioning(time_dim=32)
        out = m(torch.tensor([5, 15, 25]))
        assert not torch.allclose(out[0], out[1])
        assert not torch.allclose(out[1], out[2])

    def test_gradient_flow(self):
        m = LengthConditioning(time_dim=32)
        out = m(torch.tensor([5]))
        out.sum().backward()
        for p in m.parameters():
            assert p.grad is not None


class TestBinderCrossAttention:

    def _make(self, embed_dim=32, num_heads=4):
        torch.manual_seed(0)
        return BinderCrossAttention(embed_dim, num_heads=num_heads)

    def test_construction_validation(self):
        with pytest.raises(ValueError):
            BinderCrossAttention(33, num_heads=4)

    def test_zero_init_is_identity(self):
        block = self._make()
        q = torch.randn(2, 5, 32)
        t = torch.randn(2, 4, 32)
        out = block(q, t)
        assert torch.allclose(out, q, atol=1e-6)

    def test_target_kv_frozen(self):
        block = self._make()
        for p in block.k_proj.parameters():
            assert not p.requires_grad
        for p in block.v_proj.parameters():
            assert not p.requires_grad

    def test_no_grad_into_frozen_kv(self):
        block = self._make()
        # Randomise output projection so gradient is non-zero.
        torch.nn.init.normal_(block.out_proj.weight, std=0.05)
        q = torch.randn(1, 4, 32, requires_grad=True)
        t = torch.randn(1, 3, 32)
        out = block(q, t)
        out.sum().backward()
        for p in block.k_proj.parameters():
            assert p.grad is None
        for p in block.v_proj.parameters():
            assert p.grad is None
        # Query path still receives gradient.
        assert q.grad is not None and torch.isfinite(q.grad).all()

    def test_mask_blocks_invalid_target_positions(self):
        block = self._make()
        torch.nn.init.normal_(block.out_proj.weight, std=0.1)
        q = torch.randn(1, 2, 32)
        t = torch.randn(1, 4, 32)
        full_mask = torch.ones(1, 4, dtype=torch.bool)
        partial_mask = torch.tensor([[True, False, False, False]])
        out_full = block(q, t, target_mask=full_mask)
        out_partial = block(q, t, target_mask=partial_mask)
        # The two outputs must differ because attention sees a different
        # set of valid target keys.
        assert not torch.allclose(out_full, out_partial, atol=1e-5)

    def test_shape_validation(self):
        block = self._make()
        with pytest.raises(ValueError):
            block(torch.zeros(2, 5), torch.zeros(2, 4, 32))
        with pytest.raises(ValueError):
            block(torch.zeros(2, 5, 16), torch.zeros(2, 4, 32))
