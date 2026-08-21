"""Tests for the RFDiffusion sequence/structure track (PR 8 of the
RFDiffusion integration)."""

import pytest

try:
    import torch
    from deepchem.models.torch_models.rfdiffusion_sequence_track import (
        PairBiasedSingleAttention,
        SingleTransition,
        quaternion_to_rotation_matrix,
        BackboneUpdate,
        RFDiffusionTrackBlock,
        RFDiffusionMultiTrackStack,
    )
    has_torch = True
except ImportError:
    has_torch = False


def _rot(seed=0):
    """Return a random proper rotation matrix via QR."""
    torch.manual_seed(seed)
    q, r = torch.linalg.qr(torch.randn(3, 3))
    q = q * torch.sign(torch.diagonal(r))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


@pytest.mark.torch
@pytest.mark.skipif(not has_torch, reason='PyTorch not installed')
class TestPairBiasedSingleAttention:

    @pytest.mark.parametrize('kwargs', [
        dict(embed_dim=0, pair_dim=8),
        dict(embed_dim=16, pair_dim=0),
        dict(embed_dim=16, pair_dim=8, num_heads=0),
        dict(embed_dim=15, pair_dim=8, num_heads=4),
        dict(embed_dim=16, pair_dim=8, dropout=1.0),
    ])
    def test_bad_construction_raises(self, kwargs):
        with pytest.raises(ValueError):
            PairBiasedSingleAttention(**kwargs)

    def test_output_shape(self):
        attn = PairBiasedSingleAttention(16, 8, num_heads=4)
        single = torch.randn(2, 6, 16)
        pair = torch.randn(2, 6, 6, 8)
        assert attn(single, pair).shape == (2, 6, 16)

    def test_mask_zeros_padded_queries(self):
        attn = PairBiasedSingleAttention(16, 8, num_heads=4)
        single = torch.randn(1, 5, 16)
        pair = torch.randn(1, 5, 5, 8)
        mask = torch.tensor([[True, True, True, False, False]])
        out = attn(single, pair, mask=mask)
        assert torch.allclose(out[0, 3:], torch.zeros(2, 16), atol=1e-6)


@pytest.mark.torch
@pytest.mark.skipif(not has_torch, reason='PyTorch not installed')
class TestSingleTransition:

    @pytest.mark.parametrize('kwargs', [
        dict(embed_dim=0),
        dict(embed_dim=16, expansion=0),
        dict(embed_dim=16, dropout=1.0),
    ])
    def test_bad_construction_raises(self, kwargs):
        with pytest.raises(ValueError):
            SingleTransition(**kwargs)

    def test_output_shape(self):
        tr = SingleTransition(16)
        assert tr(torch.randn(2, 6, 16), torch.randn(2, 16)).shape == (2, 6, 16)

    def test_time_conditioning_matters(self):
        torch.manual_seed(0)
        tr = SingleTransition(16)
        single = torch.randn(2, 6, 16)
        out_a = tr(single, torch.zeros(2, 16))
        out_b = tr(single, torch.ones(2, 16))
        assert not torch.allclose(out_a, out_b)


@pytest.mark.torch
@pytest.mark.skipif(not has_torch, reason='PyTorch not installed')
class TestQuaternionToRotationMatrix:

    def test_identity_quaternion(self):
        rot = quaternion_to_rotation_matrix(torch.tensor([1.0, 0.0, 0.0, 0.0]))
        assert torch.allclose(rot, torch.eye(3), atol=1e-6)

    def test_orthogonal_and_det_one(self):
        torch.manual_seed(0)
        quat = torch.randn(5, 4)
        rot = quaternion_to_rotation_matrix(quat)
        eye = torch.eye(3).expand(5, 3, 3)
        assert torch.allclose(rot @ rot.transpose(-1, -2), eye, atol=1e-5)
        assert torch.allclose(torch.linalg.det(rot), torch.ones(5), atol=1e-5)


@pytest.mark.torch
@pytest.mark.skipif(not has_torch, reason='PyTorch not installed')
class TestBackboneUpdate:

    def test_bad_construction_raises(self):
        with pytest.raises(ValueError):
            BackboneUpdate(0)

    def test_output_shapes(self):
        upd = BackboneUpdate(16)
        single = torch.randn(2, 5, 16)
        rot = _rot().view(1, 1, 3, 3).expand(2, 5, 3, 3).contiguous()
        trans = torch.randn(2, 5, 3)
        new_rot, new_trans = upd(single, rot, trans)
        assert new_rot.shape == (2, 5, 3, 3)
        assert new_trans.shape == (2, 5, 3)

    def test_zero_init_is_identity_update(self):
        # Zero-initialised head -> update is the identity transform.
        upd = BackboneUpdate(16)
        single = torch.randn(2, 5, 16)
        rot = torch.randn(2, 5, 3, 3)
        trans = torch.randn(2, 5, 3)
        new_rot, new_trans = upd(single, rot, trans)
        assert torch.allclose(new_rot, rot, atol=1e-6)
        assert torch.allclose(new_trans, trans, atol=1e-6)


def _small_stack(seed=0, **over):
    torch.manual_seed(seed)
    cfg = dict(embed_dim=24,
               pair_dim=12,
               num_blocks=2,
               num_heads=3,
               pair_num_heads=2,
               triangle_hidden_dim=16,
               triangle_head_dim=8,
               opm_hidden_dim=8)
    cfg.update(over)
    return RFDiffusionMultiTrackStack(**cfg)


@pytest.mark.torch
@pytest.mark.skipif(not has_torch, reason='PyTorch not installed')
class TestRFDiffusionTrackBlock:

    def test_output_shapes(self):
        block = RFDiffusionTrackBlock(24,
                                      12,
                                      num_heads=3,
                                      pair_num_heads=2,
                                      triangle_hidden_dim=16,
                                      triangle_head_dim=8,
                                      opm_hidden_dim=8)
        single = torch.randn(2, 5, 24)
        pair = torch.randn(2, 5, 5, 12)
        rot = _rot().view(1, 1, 3, 3).expand(2, 5, 3, 3).contiguous()
        trans = torch.randn(2, 5, 3)
        t_emb = torch.randn(2, 24)
        s, p, r, t = block(single, pair, rot, trans, t_emb)
        assert s.shape == (2, 5, 24)
        assert p.shape == (2, 5, 5, 12)
        assert r.shape == (2, 5, 3, 3)
        assert t.shape == (2, 5, 3)

    def test_pair_stays_symmetric(self):
        block = RFDiffusionTrackBlock(24,
                                      12,
                                      num_heads=3,
                                      pair_num_heads=2,
                                      triangle_hidden_dim=16,
                                      triangle_head_dim=8,
                                      opm_hidden_dim=8)
        block.eval()
        single = torch.randn(2, 5, 24)
        pair = torch.randn(2, 5, 5, 12)
        pair = 0.5 * (pair + pair.transpose(-2, -3))
        rot = _rot().view(1, 1, 3, 3).expand(2, 5, 3, 3).contiguous()
        trans = torch.randn(2, 5, 3)
        with torch.no_grad():
            _, p, _, _ = block(single, pair, rot, trans, torch.randn(2, 24))
        assert torch.allclose(p, p.transpose(-2, -3), atol=1e-6)


@pytest.mark.torch
@pytest.mark.skipif(not has_torch, reason='PyTorch not installed')
class TestRFDiffusionMultiTrackStack:

    def test_bad_construction_raises(self):
        with pytest.raises(ValueError):
            _small_stack(num_blocks=0)

    def test_forward_shape(self):
        stack = _small_stack()
        out = stack(torch.randn(2, 6, 24), torch.randn(2, 24))
        assert out.shape == (2, 6, 24)

    def test_forward_tracks_shapes(self):
        stack = _small_stack()
        s, p, r, t = stack.forward_tracks(torch.randn(2, 6, 24),
                                          torch.randn(2, 24))
        assert s.shape == (2, 6, 24)
        assert p.shape == (2, 6, 6, 12)
        assert r.shape == (2, 6, 3, 3)
        assert t.shape == (2, 6, 3)

    def test_mask_zeros_padding(self):
        stack = _small_stack()
        stack.eval()
        mask = torch.tensor([[True, True, True, True, False, False]])
        with torch.no_grad():
            out = stack(torch.randn(1, 6, 24),
                        torch.randn(1, 24),
                        attention_mask=mask)
        assert torch.allclose(out[0, 4:], torch.zeros(2, 24), atol=1e-6)

    def test_global_rotation_leaves_single_invariant(self):
        stack = _small_stack()
        stack.eval()
        single = torch.randn(2, 5, 24)
        t_emb = torch.randn(2, 24)
        rot = _rot(1).view(1, 1, 3, 3).expand(2, 5, 3, 3).contiguous()
        trans = torch.zeros(2, 5, 3)
        with torch.no_grad():
            s1, _, _, _ = stack.forward_tracks(single, t_emb)
            s2, _, _, _ = stack.forward_tracks(single,
                                               t_emb,
                                               rotations=rot,
                                               translations=trans)
        assert torch.allclose(s1, s2, atol=1e-5)

    def test_chunked_matches_dense(self):
        stack = _small_stack()
        stack.eval()
        single = torch.randn(2, 8, 24)
        t_emb = torch.randn(2, 24)
        with torch.no_grad():
            dense = stack.forward_tracks(single, t_emb, chunk_size=None)
            chunked = stack.forward_tracks(single, t_emb, chunk_size=3)
        for a, b in zip(dense, chunked):
            assert torch.allclose(a, b, atol=1e-5)
