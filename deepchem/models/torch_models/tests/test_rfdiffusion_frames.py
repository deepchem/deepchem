"""Tests for RFDiffusion rigid-frame math utilities."""

import math

import pytest

try:
    import torch
    has_torch = True
except ImportError:
    has_torch = False

if has_torch:
    from deepchem.models.torch_models.rfdiffusion_frames import (
        apply_inverse_rigid,
        apply_rigid,
        build_backbone_frames,
        compose_rigids,
        invert_rigid,
        make_identity_rigid,
        so3_exp_map,
        so3_log_map,
    )


def _random_rotation(device=None, dtype=torch.float32, generator=None):
    matrix = torch.randn(3, 3, device=device, dtype=dtype, generator=generator)
    q_matrix, r_matrix = torch.linalg.qr(matrix)
    signs = torch.sign(torch.diagonal(r_matrix))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    q_matrix = q_matrix @ torch.diag(signs)
    if torch.det(q_matrix) < 0:
        q_matrix[:, -1] = -q_matrix[:, -1]
    return q_matrix


def _make_backbone(batch_size=2, seq_len=5, dtype=torch.float32, seed=0):
    generator = torch.Generator().manual_seed(seed)
    base = torch.tensor([[-1.2, 1.1, 0.0], [0.0, 0.0, 0.0], [1.5, 0.0, 0.0]],
                         dtype=dtype)
    offsets = torch.randn(batch_size,
                          seq_len,
                          1,
                          3,
                          dtype=dtype,
                          generator=generator)
    return base.view(1, 1, 3, 3) + offsets


@pytest.mark.torch
@pytest.mark.skipif(not has_torch, reason="PyTorch not installed")
class TestRigidFrameUtilities:

    def test_frames_are_orthonormal_and_right_handed(self):
        backbone = _make_backbone(batch_size=3, seq_len=7)
        rotations, translations = build_backbone_frames(backbone)
        assert rotations.shape == (3, 7, 3, 3)
        assert translations.shape == (3, 7, 3)
        gram = rotations.transpose(-1, -2) @ rotations
        identity = torch.eye(3).expand_as(gram)
        assert torch.allclose(gram, identity, atol=1e-5)
        det = torch.det(rotations)
        assert torch.allclose(det, torch.ones_like(det), atol=1e-5)

    def test_translation_is_alpha_carbon(self):
        backbone = _make_backbone()
        _, translations = build_backbone_frames(backbone)
        assert torch.allclose(translations, backbone[..., 1, :], atol=0.0)

    def test_frames_match_canonical_residue(self):
        backbone = torch.tensor([[[-1.0, 1.0, 0.0], [0.0, 0.0, 0.0],
                                   [1.0, 0.0, 0.0]]])
        rotations, translations = build_backbone_frames(backbone)
        assert torch.allclose(rotations, torch.eye(3).unsqueeze(0), atol=1e-6)
        assert torch.allclose(translations, torch.zeros(1, 3), atol=0.0)

    def test_frames_match_upstream_epsilon_normalization(self):
        backbone = torch.tensor(
            [[[[-1.1, 0.9, 0.2], [0.2, -0.3, 0.4], [1.4, 0.1, -0.2]]]],
            dtype=torch.float64)
        rotations, translations = build_backbone_frames(backbone, eps=1e-8)
        n_atom = backbone[..., 0, :]
        ca_atom = backbone[..., 1, :]
        c_atom = backbone[..., 2, :]
        e1 = c_atom - ca_atom
        e1 = e1 / (torch.linalg.norm(e1, dim=-1, keepdim=True) + 1e-8)
        v2 = n_atom - ca_atom
        u2 = v2 - (e1 * v2).sum(dim=-1, keepdim=True) * e1
        e2 = u2 / (torch.linalg.norm(u2, dim=-1, keepdim=True) + 1e-8)
        e3 = torch.cross(e1, e2, dim=-1)
        expected = torch.stack([e1, e2, e3], dim=-1)
        assert torch.allclose(rotations, expected, atol=0.0, rtol=0.0)
        assert torch.allclose(translations, ca_atom, atol=0.0, rtol=0.0)

    def test_frames_handle_small_n_ca_c_angle(self):
        backbone = torch.tensor([[[1.0, 1e-3, 0.0], [0.0, 0.0, 0.0],
                                   [1.0, 0.0, 0.0]]])
        rotations, _ = build_backbone_frames(backbone)
        gram = rotations.transpose(-1, -2) @ rotations
        assert torch.isfinite(rotations).all()
        assert torch.allclose(gram, torch.eye(3).unsqueeze(0), atol=3e-5)

    def test_frames_se3_equivariance(self):
        backbone = _make_backbone(seed=11)
        rotations, translations = build_backbone_frames(backbone)
        global_rotation = _random_rotation()
        global_translation = torch.randn(3)
        moved = apply_rigid(global_rotation, global_translation, backbone)
        rotations_moved, translations_moved = build_backbone_frames(moved)
        assert torch.allclose(rotations_moved,
                              global_rotation @ rotations,
                              atol=1e-5)
        expected_t = apply_rigid(global_rotation, global_translation,
                                 translations)
        assert torch.allclose(translations_moved, expected_t, atol=1e-5)

    def test_frames_raise_on_bad_shape(self):
        with pytest.raises(ValueError, match=r"\(\.\.\., 3, 3\)"):
            build_backbone_frames(torch.zeros(2, 5, 3))

    def test_frames_raise_on_nonpositive_eps(self):
        backbone = _make_backbone()
        with pytest.raises(ValueError, match="eps"):
            build_backbone_frames(backbone, eps=0.0)

    def test_make_identity_rigid_shape_and_values(self):
        rotations, translations = make_identity_rigid((4, 3))
        assert rotations.shape == (4, 3, 3, 3)
        assert translations.shape == (4, 3, 3)
        identity = torch.eye(3).expand_as(rotations)
        assert torch.equal(rotations, identity)
        assert torch.equal(translations, torch.zeros_like(translations))

    def test_make_identity_rigid_returns_writable_tensor(self):
        rotations, _ = make_identity_rigid((4, 3))
        rotations[0, 0, 0, 0] = 99.0
        assert rotations[0, 0, 0, 0].item() == 99.0
        assert rotations[0, 1, 0, 0].item() == 1.0

    def test_apply_rigid_inverse_roundtrip(self):
        backbone = _make_backbone()
        rotations, translations = build_backbone_frames(backbone)
        local = apply_inverse_rigid(rotations, translations, backbone)
        recovered = apply_rigid(rotations, translations, local)
        assert torch.allclose(recovered, backbone, atol=1e-5)

    def test_apply_rigid_broadcasts_over_extra_dims(self):
        rotations, translations = make_identity_rigid((2, 3))
        rotations = rotations.clone()
        for batch_idx in range(2):
            for seq_idx in range(3):
                rotations[batch_idx, seq_idx] = _random_rotation()
        translations = torch.randn(2, 3, 3)
        points = torch.randn(2, 3, 4, 5, 3)
        moved = apply_rigid(rotations, translations, points)
        expected = torch.einsum("blij,blkmj->blkmi", rotations, points)
        expected = expected + translations.view(2, 3, 1, 1, 3)
        assert moved.shape == points.shape
        assert torch.allclose(moved, expected, atol=1e-5)

    def test_apply_rigid_translation_only(self):
        rotations, _ = make_identity_rigid((2, 5))
        translations = torch.randn(2, 5, 3)
        points = torch.randn(2, 5, 3)
        assert torch.allclose(apply_rigid(rotations, translations, points),
                              points + translations,
                              atol=1e-6)

    def test_apply_rigid_raises_on_too_few_point_dims(self):
        rotations, translations = make_identity_rigid((2, 3))
        bad = torch.randn(3)
        with pytest.raises(ValueError, match="at least"):
            apply_rigid(rotations, translations, bad)

    def test_invert_rigid_is_left_and_right_inverse(self):
        backbone = _make_backbone()
        rotations, translations = build_backbone_frames(backbone)
        inv_rotations, inv_translations = invert_rigid(rotations, translations)
        left_r, left_t = compose_rigids(inv_rotations, inv_translations,
                                        rotations, translations)
        right_r, right_t = compose_rigids(rotations, translations,
                                          inv_rotations, inv_translations)
        assert torch.allclose(left_r, torch.eye(3).expand_as(left_r), atol=1e-5)
        assert torch.allclose(left_t, torch.zeros_like(left_t), atol=1e-5)
        assert torch.allclose(right_r,
                              torch.eye(3).expand_as(right_r),
                              atol=1e-5)
        assert torch.allclose(right_t, torch.zeros_like(right_t), atol=1e-5)

    def test_compose_matches_sequential_application(self):
        rotations_a = torch.stack([_random_rotation() for _ in range(6)])
        rotations_a = rotations_a.view(2, 3, 3, 3)
        rotations_b = torch.stack([_random_rotation() for _ in range(6)])
        rotations_b = rotations_b.view(2, 3, 3, 3)
        translations_a = torch.randn(2, 3, 3)
        translations_b = torch.randn(2, 3, 3)
        points = torch.randn(2, 3, 7, 3)

        sequential = apply_rigid(
            rotations_a, translations_a,
            apply_rigid(rotations_b, translations_b, points))
        rotations, translations = compose_rigids(rotations_a, translations_a,
                                                 rotations_b, translations_b)
        composed = apply_rigid(rotations, translations, points)
        assert torch.allclose(composed, sequential, atol=1e-5)

    def test_compose_is_associative(self):
        rotations = [_random_rotation() for _ in range(3)]
        translations = [torch.randn(3) for _ in range(3)]
        rotation_a, rotation_b, rotation_c = rotations
        translation_a, translation_b, translation_c = translations
        left_r, left_t = compose_rigids(
            *compose_rigids(rotation_a, translation_a, rotation_b,
                            translation_b), rotation_c, translation_c)
        right_r, right_t = compose_rigids(
            rotation_a, translation_a,
            *compose_rigids(rotation_b, translation_b, rotation_c,
                            translation_c))
        assert torch.allclose(left_r, right_r, atol=1e-5)
        assert torch.allclose(left_t, right_t, atol=1e-5)

    def test_apply_rigid_supports_fp64_precision(self):
        backbone = _make_backbone(dtype=torch.float64, seed=3)
        rotations, translations = build_backbone_frames(backbone)
        local = apply_inverse_rigid(rotations, translations, backbone)
        recovered = apply_rigid(rotations, translations, local)
        assert torch.allclose(recovered, backbone, atol=1e-12)


@pytest.mark.torch
@pytest.mark.skipif(not has_torch, reason="PyTorch not installed")
class TestSO3Maps:

    def test_exp_returns_proper_rotation(self):
        torch.manual_seed(0)
        tangent = torch.randn(32, 3) * 0.5
        rotations = so3_exp_map(tangent)
        det = torch.det(rotations)
        ortho_err = (rotations @ rotations.transpose(-1, -2) -
                     torch.eye(3)).abs().max()
        assert torch.allclose(det, torch.ones_like(det), atol=1e-5)
        assert ortho_err < 1e-5

    def test_exp_zero_tangent_gives_identity(self):
        rotations = so3_exp_map(torch.zeros(4, 3))
        identity = torch.eye(3).expand(4, 3, 3)
        assert torch.allclose(rotations, identity, atol=1e-7)

    def test_log_exp_roundtrip_principal_branch(self):
        torch.manual_seed(1)
        axis = torch.randn(50, 3)
        axis = axis / axis.norm(dim=-1, keepdim=True)
        omega = torch.rand(50) * (math.pi - 1e-3) + 1e-4
        tangent = omega.unsqueeze(-1) * axis
        recovered = so3_log_map(so3_exp_map(tangent))
        assert (tangent - recovered).norm(dim=-1).max() < 1e-3

    def test_log_map_stable_near_pi(self):
        axis = torch.tensor([[0.6, -0.2, 0.7745967]])
        axis = axis / axis.norm(dim=-1, keepdim=True)
        tangent = (math.pi - 1e-4) * axis
        recovered = so3_log_map(so3_exp_map(tangent))
        assert torch.allclose(recovered, tangent, atol=1e-3)

    def test_small_omega_taylor_branch(self):
        tangent = torch.tensor([[1e-7, 0.0, 0.0], [0.0, 1e-7, 0.0]])
        rotations = so3_exp_map(tangent)
        identity = torch.eye(3).expand_as(rotations)
        assert (rotations - identity).abs().max() < 1e-6
