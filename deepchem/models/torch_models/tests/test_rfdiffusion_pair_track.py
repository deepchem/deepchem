"""Tests for the RFDiffusion pair-track blocks (PR 7 of the RFDiffusion
integration)."""

import pytest

try:
    import torch
    has_torch = True
except ImportError:
    has_torch = False

if has_torch:
    from deepchem.models.torch_models.rfdiffusion_pair_track import (
        RelativePositionEmbedding,
        OuterProductMean,
        TriangleMultiplicativeUpdate,
        TriangleAttention,
        PairTransition,
    )


@pytest.mark.torch
@pytest.mark.skipif(not has_torch, reason='PyTorch not installed')
class TestRelativePositionEmbedding:
    """Symmetric clipped relative-position embedding."""

    @pytest.mark.parametrize('kwargs', [
        dict(pair_dim=0),
        dict(pair_dim=16, max_relative_position=0),
    ])
    def test_bad_construction_raises(self, kwargs):
        with pytest.raises(ValueError):
            RelativePositionEmbedding(**kwargs)

    def test_output_shape(self):
        rel = RelativePositionEmbedding(pair_dim=16, max_relative_position=8)
        assert rel(10).shape == (10, 10, 16)

    def test_symmetric(self):
        rel = RelativePositionEmbedding(pair_dim=12, max_relative_position=8)
        emb = rel(9)
        assert torch.allclose(emb, emb.transpose(0, 1))

    def test_clip_radius(self):
        rel = RelativePositionEmbedding(pair_dim=8, max_relative_position=3)
        emb = rel(12)
        # Offsets beyond the clip radius collapse to the same bucket.
        assert torch.allclose(emb[0, 4], emb[0, 11])
        assert torch.allclose(emb[0, 5], emb[0, 4])
        # Offsets within the radius use distinct buckets.
        assert not torch.allclose(emb[0, 1], emb[0, 2])

    def test_diagonal_is_constant(self):
        rel = RelativePositionEmbedding(pair_dim=8, max_relative_position=4)
        emb = rel(6)
        diag = torch.stack([emb[i, i] for i in range(6)])
        assert torch.allclose(diag, diag[:1].expand_as(diag))


@pytest.mark.torch
@pytest.mark.skipif(not has_torch, reason='PyTorch not installed')
class TestOuterProductMean:
    """Symmetrised 1D -> 2D outer-product injection."""

    @pytest.mark.parametrize('kwargs', [
        dict(embed_dim=0, pair_dim=16),
        dict(embed_dim=12, pair_dim=0),
        dict(embed_dim=12, pair_dim=16, hidden_dim=0),
    ])
    def test_bad_construction_raises(self, kwargs):
        with pytest.raises(ValueError):
            OuterProductMean(**kwargs)

    def test_output_shape(self):
        opm = OuterProductMean(embed_dim=12, pair_dim=16, hidden_dim=8)
        out = opm(torch.randn(2, 7, 12))
        assert out.shape == (2, 7, 7, 16)

    def test_symmetric(self):
        torch.manual_seed(0)
        opm = OuterProductMean(embed_dim=10, pair_dim=14, hidden_dim=6)
        out = opm(torch.randn(3, 5, 10))
        assert torch.allclose(out, out.transpose(1, 2), atol=1e-6)

    def test_gradient_flow(self):
        opm = OuterProductMean(embed_dim=8, pair_dim=8, hidden_dim=4)
        single = torch.randn(2, 4, 8, requires_grad=True)
        opm(single).sum().backward()
        assert single.grad is not None
        assert torch.isfinite(single.grad).all()


@pytest.mark.torch
@pytest.mark.skipif(not has_torch, reason='PyTorch not installed')
class TestTriangleMultiplicativeUpdate:
    """Outgoing / incoming triangle multiplicative update."""

    @pytest.mark.parametrize('kwargs', [
        dict(pair_dim=0),
        dict(pair_dim=16, hidden_dim=0),
    ])
    def test_bad_construction_raises(self, kwargs):
        with pytest.raises(ValueError):
            TriangleMultiplicativeUpdate(**kwargs)

    @pytest.mark.parametrize('outgoing', [True, False])
    def test_output_shape(self, outgoing):
        tmu = TriangleMultiplicativeUpdate(pair_dim=16,
                                           hidden_dim=32,
                                           outgoing=outgoing)
        assert tmu(torch.randn(2, 6, 6, 16)).shape == (2, 6, 6, 16)

    def test_requires_4d(self):
        tmu = TriangleMultiplicativeUpdate(pair_dim=16)
        with pytest.raises(ValueError):
            tmu(torch.randn(6, 6, 16))

    @pytest.mark.parametrize('outgoing', [True, False])
    def test_chunked_matches_dense(self, outgoing):
        torch.manual_seed(0)
        tmu = TriangleMultiplicativeUpdate(pair_dim=12,
                                           hidden_dim=16,
                                           outgoing=outgoing)
        tmu.eval()
        pair = torch.randn(2, 9, 9, 12)
        with torch.no_grad():
            dense = tmu(pair)
            chunked = tmu(pair, chunk_size=3)
        assert torch.allclose(dense, chunked, atol=1e-5)

    def test_outgoing_differs_from_incoming(self):
        torch.manual_seed(0)
        pair = torch.randn(1, 5, 5, 8)
        out_mod = TriangleMultiplicativeUpdate(pair_dim=8,
                                               hidden_dim=8,
                                               outgoing=True)
        in_mod = TriangleMultiplicativeUpdate(pair_dim=8,
                                              hidden_dim=8,
                                              outgoing=False)
        # Share weights so only the contraction direction differs.
        in_mod.load_state_dict(out_mod.state_dict())
        with torch.no_grad():
            assert not torch.allclose(out_mod(pair), in_mod(pair))

    def test_gradient_flow(self):
        tmu = TriangleMultiplicativeUpdate(pair_dim=8, hidden_dim=8)
        pair = torch.randn(2, 5, 5, 8, requires_grad=True)
        tmu(pair).sum().backward()
        assert pair.grad is not None
        assert torch.isfinite(pair.grad).all()


@pytest.mark.torch
@pytest.mark.skipif(not has_torch, reason='PyTorch not installed')
class TestTriangleAttention:
    """Starting / ending-node triangle self-attention."""

    @pytest.mark.parametrize('kwargs', [
        dict(pair_dim=0),
        dict(pair_dim=16, num_heads=0),
        dict(pair_dim=16, num_heads=4, head_dim=0),
    ])
    def test_bad_construction_raises(self, kwargs):
        with pytest.raises(ValueError):
            TriangleAttention(**kwargs)

    @pytest.mark.parametrize('starting_node', [True, False])
    def test_output_shape(self, starting_node):
        ta = TriangleAttention(pair_dim=16,
                               num_heads=4,
                               head_dim=8,
                               starting_node=starting_node)
        assert ta(torch.randn(2, 6, 6, 16)).shape == (2, 6, 6, 16)

    def test_requires_4d(self):
        ta = TriangleAttention(pair_dim=16)
        with pytest.raises(ValueError):
            ta(torch.randn(6, 6, 16))

    @pytest.mark.parametrize('starting_node', [True, False])
    def test_chunked_matches_dense(self, starting_node):
        torch.manual_seed(0)
        ta = TriangleAttention(pair_dim=12,
                               num_heads=3,
                               head_dim=8,
                               starting_node=starting_node)
        ta.eval()
        pair = torch.randn(2, 8, 8, 12)
        with torch.no_grad():
            dense = ta(pair)
            chunked = ta(pair, chunk_size=2)
        assert torch.allclose(dense, chunked, atol=1e-5)

    def test_gradient_flow(self):
        ta = TriangleAttention(pair_dim=8, num_heads=2, head_dim=4)
        pair = torch.randn(2, 5, 5, 8, requires_grad=True)
        ta(pair).sum().backward()
        assert pair.grad is not None
        assert torch.isfinite(pair.grad).all()


@pytest.mark.torch
@pytest.mark.skipif(not has_torch, reason='PyTorch not installed')
class TestPairTransition:
    """Position-wise pair feed-forward."""

    @pytest.mark.parametrize('kwargs', [
        dict(pair_dim=0),
        dict(pair_dim=16, expansion=0),
    ])
    def test_bad_construction_raises(self, kwargs):
        with pytest.raises(ValueError):
            PairTransition(**kwargs)

    def test_output_shape(self):
        pt = PairTransition(pair_dim=16, expansion=2)
        assert pt(torch.randn(2, 6, 6, 16)).shape == (2, 6, 6, 16)

    def test_gradient_flow(self):
        pt = PairTransition(pair_dim=8)
        pair = torch.randn(2, 4, 4, 8, requires_grad=True)
        pt(pair).sum().backward()
        assert pair.grad is not None
        assert torch.isfinite(pair.grad).all()
