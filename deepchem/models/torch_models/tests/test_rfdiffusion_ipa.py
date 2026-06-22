"""Tests for InvariantPointAttention (PR 6 of the RFDiffusion integration)."""

import math

import pytest

try:
    import torch
    has_torch = True
except ImportError:
    has_torch = False

if has_torch:
    from deepchem.models.torch_models.rfdiffusion_ipa import (
        InvariantPointAttention,)
    from deepchem.models.torch_models.rfdiffusion_frames import (
        apply_rigid,
        build_backbone_frames,
    )


def _random_rotation(dtype=None, seed=None):
    """Return a random proper rotation matrix via QR decomposition."""
    if seed is not None:
        torch.manual_seed(seed)
    dtype = dtype or torch.float32
    q, r = torch.linalg.qr(torch.randn(3, 3, dtype=dtype))
    # Ensure det = +1.
    sign = torch.sign(torch.det(q @ torch.diag(torch.sign(torch.diagonal(r)))))
    return q * sign


def _make_backbone(batch=2, seq=6, seed=0):
    """Build a small synthetic backbone tensor for frame construction."""
    torch.manual_seed(seed)
    base = torch.tensor([[-1.2, 1.1, 0.0], [0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    return base.view(1, 1, 3, 3) + torch.randn(batch, seq, 1, 3)


@pytest.mark.torch
@pytest.mark.skipif(not has_torch, reason='PyTorch not installed')
class TestIPAConstruction:
    """Tests for argument validation in __init__."""

    @pytest.mark.parametrize(
        'kwargs',
        [
            dict(embed_dim=0),
            dict(embed_dim=16, num_heads=0),
            dict(embed_dim=15, num_heads=4),  # not divisible
            dict(embed_dim=16, num_heads=4, num_qk_points=0),
            dict(embed_dim=16, num_heads=4, num_v_points=0),
            dict(embed_dim=16, num_heads=4, pair_dim=0),
            dict(embed_dim=16, num_heads=4, dropout=-0.1),
            dict(embed_dim=16, num_heads=4, dropout=1.0),
            dict(embed_dim=16, num_heads=4, eps=0.0),
        ])
    def test_bad_construction_raises(self, kwargs):
        with pytest.raises(ValueError):
            InvariantPointAttention(**kwargs)

    def test_basic_construction(self):
        layer = InvariantPointAttention(embed_dim=32, num_heads=4)
        assert layer.embed_dim == 32
        assert layer.num_heads == 4
        assert layer.head_dim == 8

    def test_pair_dim_construction(self):
        layer = InvariantPointAttention(embed_dim=32, num_heads=4, pair_dim=16)
        assert layer.pair_dim == 16
        assert layer.pair_bias is not None


@pytest.mark.torch
@pytest.mark.skipif(not has_torch, reason='PyTorch not installed')
class TestIPAOutputShape:
    """Tests that the output shape is always ``(B, L, embed_dim)``."""

    def test_no_pair_no_mask(self):
        backbone = _make_backbone(batch=2, seq=6)
        R, t = build_backbone_frames(backbone)
        layer = InvariantPointAttention(embed_dim=32, num_heads=4)
        out = layer(torch.randn(2, 6, 32), R, t)
        assert out.shape == (2, 6, 32)

    def test_with_pair_repr(self):
        backbone = _make_backbone(batch=2, seq=6)
        R, t = build_backbone_frames(backbone)
        layer = InvariantPointAttention(embed_dim=32, num_heads=4, pair_dim=8)
        pair = torch.randn(2, 6, 6, 8)
        out = layer(torch.randn(2, 6, 32), R, t, pair_repr=pair)
        assert out.shape == (2, 6, 32)

    def test_with_mask(self):
        backbone = _make_backbone(batch=2, seq=6)
        R, t = build_backbone_frames(backbone)
        layer = InvariantPointAttention(embed_dim=32, num_heads=4)
        mask = torch.tensor([[1, 1, 1, 1, 0, 0],
                             [1, 1, 1, 0, 0, 0]], dtype=torch.bool)
        out = layer(torch.randn(2, 6, 32), R, t, mask=mask)
        assert out.shape == (2, 6, 32)


@pytest.mark.torch
@pytest.mark.skipif(not has_torch, reason='PyTorch not installed')
class TestIPAMasking:
    """Tests for the residue mask behaviour."""

    def test_masked_outputs_are_zero(self):
        backbone = _make_backbone(batch=1, seq=5)
        R, t = build_backbone_frames(backbone)
        layer = InvariantPointAttention(embed_dim=32, num_heads=4)
        mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.bool)
        out = layer(torch.randn(1, 5, 32), R, t, mask=mask)
        assert torch.allclose(out[:, 3:], torch.zeros_like(out[:, 3:]),
                              atol=0.0)

    def test_masked_keys_do_not_affect_unmasked_queries(self):
        torch.manual_seed(7)
        backbone = _make_backbone(batch=1, seq=5)
        R, t = build_backbone_frames(backbone)
        layer = InvariantPointAttention(embed_dim=32, num_heads=4).eval()
        single = torch.randn(1, 5, 32)
        mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.bool)

        out_a = layer(single, R, t, mask=mask)
        # Replace masked rows with large random values.
        perturbed = single.clone()
        perturbed[:, 3:] = torch.randn(1, 2, 32) * 1000.0
        out_b = layer(perturbed, R, t, mask=mask)
        assert torch.allclose(out_a[:, :3], out_b[:, :3], atol=1e-4)

    def test_wrong_mask_shape_raises(self):
        backbone = _make_backbone(batch=1, seq=5)
        R, t = build_backbone_frames(backbone)
        layer = InvariantPointAttention(embed_dim=16, num_heads=4)
        with pytest.raises(ValueError, match='mask'):
            layer(torch.randn(1, 5, 16), R, t,
                  mask=torch.ones(1, 5, 1, dtype=torch.bool))


@pytest.mark.torch
@pytest.mark.skipif(not has_torch, reason='PyTorch not installed')
class TestIPAInvariance:
    """SE(3) invariance checks — the core correctness property."""

    @pytest.mark.parametrize('seed', [0, 1, 42, 2026])
    def test_global_rotation_invariance(self, seed):
        torch.manual_seed(seed)
        backbone = _make_backbone(batch=2, seq=6, seed=seed)
        R, t = build_backbone_frames(backbone)
        layer = InvariantPointAttention(
            embed_dim=32, num_heads=4, num_qk_points=3,
            num_v_points=4).eval()
        single = torch.randn(2, 6, 32)

        out = layer(single, R, t)

        # Apply a random global rigid transform to all frames.
        g_R = _random_rotation(seed=seed)
        g_t = torch.randn(3)
        R_moved = torch.einsum('ij,...jk->...ik', g_R, R)
        t_moved = apply_rigid(g_R, g_t, t)
        out_moved = layer(single, R_moved, t_moved)

        assert torch.allclose(out, out_moved, atol=1e-4)

    def test_translation_only_invariance(self):
        backbone = _make_backbone(batch=2, seq=6, seed=5)
        R, t = build_backbone_frames(backbone)
        layer = InvariantPointAttention(embed_dim=32, num_heads=4).eval()
        single = torch.randn(2, 6, 32)

        out = layer(single, R, t)
        out_shifted = layer(single, R, t + torch.randn(3) * 100.0)
        assert torch.allclose(out, out_shifted, atol=1e-4)

    def test_pair_bias_does_not_break_invariance(self):
        torch.manual_seed(3)
        backbone = _make_backbone(batch=2, seq=6, seed=3)
        R, t = build_backbone_frames(backbone)
        layer = InvariantPointAttention(embed_dim=32, num_heads=4,
                                        pair_dim=8).eval()
        single = torch.randn(2, 6, 32)
        pair = torch.randn(2, 6, 6, 8)

        out = layer(single, R, t, pair_repr=pair)

        g_R = _random_rotation(seed=3)
        g_t = torch.randn(3)
        R_moved = torch.einsum('ij,...jk->...ik', g_R, R)
        t_moved = apply_rigid(g_R, g_t, t)
        out_moved = layer(single, R_moved, t_moved, pair_repr=pair)

        assert torch.allclose(out, out_moved, atol=1e-4)

    def test_invariance_in_fp64(self):
        backbone = _make_backbone(batch=1, seq=4, seed=9).double()
        R, t = build_backbone_frames(backbone)
        layer = InvariantPointAttention(embed_dim=16, num_heads=4).eval()
        layer = layer.double()
        single = torch.randn(1, 4, 16, dtype=torch.float64)

        out = layer(single, R, t)
        g_R = _random_rotation(dtype=torch.float64, seed=9)
        g_t = torch.randn(3, dtype=torch.float64)
        R_moved = torch.einsum('ij,...jk->...ik', g_R, R)
        t_moved = apply_rigid(g_R, g_t, t)
        out_moved = layer(single, R_moved, t_moved)

        assert torch.allclose(out, out_moved, atol=1e-7)


@pytest.mark.torch
@pytest.mark.skipif(not has_torch, reason='PyTorch not installed')
class TestIPAValidation:
    """Tests that bad input shapes raise ValueError."""

    def test_pair_repr_missing_raises(self):
        backbone = _make_backbone(batch=1, seq=4)
        R, t = build_backbone_frames(backbone)
        layer = InvariantPointAttention(embed_dim=16, num_heads=4, pair_dim=8)
        with pytest.raises(ValueError):
            layer(torch.randn(1, 4, 16), R, t)  # pair_repr missing

    def test_extra_pair_repr_raises(self):
        backbone = _make_backbone(batch=1, seq=4)
        R, t = build_backbone_frames(backbone)
        layer = InvariantPointAttention(embed_dim=16, num_heads=4)  # no pair
        with pytest.raises(ValueError):
            layer(torch.randn(1, 4, 16), R, t,
                  pair_repr=torch.randn(1, 4, 4, 8))

    def test_bad_single_repr_dims_raises(self):
        backbone = _make_backbone(batch=1, seq=4)
        R, t = build_backbone_frames(backbone)
        layer = InvariantPointAttention(embed_dim=16, num_heads=4)
        with pytest.raises(ValueError):
            layer(torch.randn(1, 4), R, t)  # 2-D instead of 3-D

    def test_bad_rotation_shape_raises(self):
        layer = InvariantPointAttention(embed_dim=16, num_heads=4)
        with pytest.raises(ValueError):
            layer(torch.randn(1, 4, 16),
                  torch.randn(1, 4, 3),  # wrong rotation shape
                  torch.zeros(1, 4, 3))


@pytest.mark.torch
@pytest.mark.skipif(not has_torch, reason='PyTorch not installed')
class TestIPAGradients:
    """Gradient-flow and parameter tests."""

    def test_gradients_flow_to_inputs(self):
        backbone = _make_backbone(batch=2, seq=5)
        R, t = build_backbone_frames(backbone)
        layer = InvariantPointAttention(embed_dim=32, num_heads=4)
        single = torch.randn(2, 5, 32, requires_grad=True)
        layer(single, R, t).sum().backward()
        assert single.grad is not None

    def test_point_weights_are_positive_after_softplus(self):
        layer = InvariantPointAttention(embed_dim=32, num_heads=4)
        gamma = torch.nn.functional.softplus(layer.point_weights)
        assert (gamma > 0).all()

    def test_weighting_constants_have_expected_values(self):
        layer = InvariantPointAttention(embed_dim=32, num_heads=4,
                                        num_qk_points=4)
        expected_wc = math.sqrt(2.0 / (9.0 * 4))
        expected_wl = math.sqrt(1.0 / 2.0)  # no pair → L=2
        assert abs(float(layer.w_c) - expected_wc) < 1e-6
        assert abs(float(layer.w_l) - expected_wl) < 1e-6

    def test_dropout_eval_is_deterministic(self):
        backbone = _make_backbone(batch=2, seq=5)
        R, t = build_backbone_frames(backbone)
        layer = InvariantPointAttention(embed_dim=32, num_heads=4,
                                        dropout=0.5).eval()
        single = torch.randn(2, 5, 32)
        out_a = layer(single, R, t)
        out_b = layer(single, R, t)
        assert torch.allclose(out_a, out_b)
