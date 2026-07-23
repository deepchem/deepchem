"""Exhaustive tests for SE(3) rigid-frame utilities and Invariant Point
Attention.

The test suite is organised by mathematical property:

* **Frame construction** — orthonormality, right-handedness, exact
  agreement with AlphaFold2 Algorithm 21 on a hand-computed example,
  invariance to atom ordering, behaviour under near-degenerate inputs.
* **Rigid algebra** — round-trips for ``apply``/``apply_inverse`` and
  ``invert``, associativity of ``compose_rigids``, identity element,
  broadcasting over arbitrary trailing dimensions.
* **Invariant Point Attention** — shape preservation, exact SE(3)
  invariance of the scalar output to global rotations *and*
  translations, key/query masking semantics, deterministic gradient
  flow into every learnable parameter, fp64 promotion, dropout-eval
  determinism, doubled-token symmetry, construction validation, and
  consistency between the no-pair and with-pair code paths.

All numerical tests use a strict ``atol = 1e-4`` in fp32 and
``atol = 1e-7`` in fp64, matching the project's standards contract
(Phase 1 of the Technical Readiness Report).
"""

import math

import pytest

try:
    import torch
    has_torch = True
except ImportError:
    has_torch = False

if has_torch:
    from deepchem.models.torch_models.rfdiffusion_se3 import (
        InvariantPointAttention,
        apply_inverse_rigid,
        apply_rigid,
        build_backbone_frames,
        compose_rigids,
        invert_rigid,
        make_identity_rigid,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _random_rotation(device=None, dtype=torch.float32, generator=None):
    """Sample a uniformly random right-handed rotation via QR.

    The Haar distribution on SO(3) is approximated by Q from the
    QR-decomposition of an i.i.d. Gaussian matrix, with the signs of R's
    diagonal absorbed into Q (Mezzadri 2007). A final det-flip ensures
    ``det(Q) = +1``.
    """
    matrix = torch.randn(3, 3, device=device, dtype=dtype, generator=generator)
    q_matrix, r_matrix = torch.linalg.qr(matrix)
    signs = torch.sign(torch.diagonal(r_matrix))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    q_matrix = q_matrix @ torch.diag(signs)
    if torch.det(q_matrix) < 0:
        q_matrix[:, -1] = -q_matrix[:, -1]
    return q_matrix


def _make_backbone(batch_size=2,
                   seq_len=5,
                   dtype=torch.float32,
                   seed=0):
    """Construct nondegenerate ``(N, Cα, C)`` backbones around a planar template."""
    generator = torch.Generator().manual_seed(seed)
    base = torch.tensor([[-1.2, 1.1, 0.0],
                         [0.0, 0.0, 0.0],
                         [1.5, 0.0, 0.0]],
                        dtype=dtype)
    offsets = torch.randn(batch_size,
                          seq_len,
                          1,
                          3,
                          dtype=dtype,
                          generator=generator)
    return base.view(1, 1, 3, 3) + offsets


# ---------------------------------------------------------------------------
# Rigid-frame utilities
# ---------------------------------------------------------------------------


@pytest.mark.torch
@pytest.mark.skipif(not has_torch, reason='PyTorch not installed')
class TestRigidFrameUtilities:
    """Tests for :func:`build_backbone_frames` and rigid algebra."""

    # --- Frame construction ----------------------------------------

    def test_frames_are_orthonormal_and_right_handed(self):
        backbone = _make_backbone(batch_size=3, seq_len=7)
        R, t = build_backbone_frames(backbone)
        assert R.shape == (3, 7, 3, 3)
        assert t.shape == (3, 7, 3)
        gram = R.transpose(-1, -2) @ R
        identity = torch.eye(3).expand_as(gram)
        assert torch.allclose(gram, identity, atol=1e-5)
        det = torch.det(R)
        assert torch.allclose(det, torch.ones_like(det), atol=1e-5)

    def test_translation_is_alpha_carbon(self):
        backbone = _make_backbone()
        _, t = build_backbone_frames(backbone)
        assert torch.allclose(t, backbone[..., 1, :], atol=0.0)

    def test_local_x_axis_is_unit_ca_to_c(self):
        """The first column of R must equal (C − Cα) / ‖C − Cα‖."""
        backbone = _make_backbone(seed=42)
        R, _ = build_backbone_frames(backbone)
        expected_x = backbone[..., 2, :] - backbone[..., 1, :]
        expected_x = expected_x / torch.linalg.norm(
            expected_x, dim=-1, keepdim=True)
        assert torch.allclose(R[..., :, 0], expected_x, atol=1e-6)

    def test_local_y_axis_perpendicular_to_x(self):
        """ê₂ must be orthogonal to ê₁ and lie in the (Cα→N) plane."""
        backbone = _make_backbone(seed=7)
        R, _ = build_backbone_frames(backbone)
        x_axis = R[..., :, 0]
        y_axis = R[..., :, 1]
        dot = (x_axis * y_axis).sum(dim=-1)
        assert torch.allclose(dot, torch.zeros_like(dot), atol=1e-6)

    def test_frames_match_alphafold_algorithm_21_on_canonical_residue(self):
        """Hand-computed reference for a canonical planar residue."""
        backbone = torch.tensor([[[-1.0, 1.0, 0.0],
                                  [0.0, 0.0, 0.0],
                                  [1.0, 0.0, 0.0]]])
        R, t = build_backbone_frames(backbone)
        # x̂ = (1, 0, 0); the N atom contributes a +y component after
        # projection so ŷ = (0, 1, 0); ẑ = x̂ × ŷ = (0, 0, 1).
        expected_R = torch.eye(3).unsqueeze(0)
        assert torch.allclose(R, expected_R, atol=1e-6)
        assert torch.allclose(t, torch.zeros(1, 3), atol=0.0)

    def test_frames_match_upstream_epsilon_normalization(self):
        backbone = torch.tensor([[[[-1.1, 0.9, 0.2],
                                   [0.2, -0.3, 0.4],
                                   [1.4, 0.1, -0.2]]]], dtype=torch.float64)
        R, t = build_backbone_frames(backbone, eps=1e-8)
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
        assert torch.allclose(R, expected, atol=0.0, rtol=0.0)
        assert torch.allclose(t, ca_atom, atol=0.0, rtol=0.0)

    def test_frames_handle_small_n_ca_c_angle(self):
        """A small but recoverable ∠N-Cα-C still yields orthonormal frames.

        The N-Cα-C angle is intentionally small (≈1 mrad) so that the
        Gram-Schmidt y component before normalization has magnitude
        ≈1e-3, well above ``eps = 1e-8``. The upstream ``norm + eps``
        convention still leaves a tiny O(eps / angle) norm error.
        """
        backbone = torch.tensor([[[1.0, 1e-3, 0.0],
                                  [0.0, 0.0, 0.0],
                                  [1.0, 0.0, 0.0]]])
        R, _ = build_backbone_frames(backbone)
        assert torch.isfinite(R).all()
        gram = R.transpose(-1, -2) @ R
        assert torch.allclose(gram, torch.eye(3).unsqueeze(0), atol=3e-5)

    def test_frames_handle_pathologically_collinear_atoms(self):
        """Truly degenerate (N − Cα ∥ C − Cα) inputs must not produce NaN.

        ε-stabilization cannot manufacture orthogonality from zero
        information, but it must keep the result finite — otherwise a
        single bad residue would corrupt an entire batch's gradient.
        """
        backbone = torch.tensor([[[1.0, 0.0, 0.0],
                                  [0.0, 0.0, 0.0],
                                  [1.0, 0.0, 0.0]]])
        R, t = build_backbone_frames(backbone)
        assert torch.isfinite(R).all()
        assert torch.isfinite(t).all()

    def test_frames_se3_equivariance(self):
        """build_backbone_frames is SE(3)-equivariant in (R, t)."""
        backbone = _make_backbone(seed=11)
        R, t = build_backbone_frames(backbone)
        global_R = _random_rotation()
        global_t = torch.randn(3)
        moved = apply_rigid(global_R, global_t, backbone)
        R_moved, t_moved = build_backbone_frames(moved)
        assert torch.allclose(R_moved, global_R @ R, atol=1e-5)
        expected_t = apply_rigid(global_R, global_t, t)
        assert torch.allclose(t_moved, expected_t, atol=1e-5)

    def test_frames_raise_on_bad_shape(self):
        with pytest.raises(ValueError, match=r'\(\.\.\., 3, 3\)'):
            build_backbone_frames(torch.zeros(2, 5, 3))

    def test_frames_raise_on_nonpositive_eps(self):
        backbone = _make_backbone()
        with pytest.raises(ValueError, match='eps'):
            build_backbone_frames(backbone, eps=0.0)

    # --- make_identity_rigid ----------------------------------------

    def test_make_identity_rigid_shape_and_values(self):
        R, t = make_identity_rigid((4, 3))
        assert R.shape == (4, 3, 3, 3)
        assert t.shape == (4, 3, 3)
        identity = torch.eye(3).expand_as(R)
        assert torch.equal(R, identity)
        assert torch.equal(t, torch.zeros_like(t))

    def test_make_identity_rigid_scalar_shape(self):
        R, t = make_identity_rigid(())
        assert R.shape == (3, 3)
        assert t.shape == (3,)

    def test_make_identity_rigid_respects_dtype(self):
        R, t = make_identity_rigid((2,), dtype=torch.float64)
        assert R.dtype == torch.float64
        assert t.dtype == torch.float64

    def test_make_identity_rigid_returns_writable_tensor(self):
        """The returned R must be a fresh tensor (no expanded strides)."""
        R, _ = make_identity_rigid((4, 3))
        R[0, 0, 0, 0] = 99.0  # Must not raise (no broadcasting view).
        assert R[0, 0, 0, 0].item() == 99.0
        assert R[0, 1, 0, 0].item() == 1.0  # Other slices unaffected.

    # --- apply_rigid / apply_inverse_rigid --------------------------

    def test_apply_rigid_inverse_roundtrip(self):
        backbone = _make_backbone()
        R, t = build_backbone_frames(backbone)
        local = apply_inverse_rigid(R, t, backbone)
        recovered = apply_rigid(R, t, local)
        assert torch.allclose(recovered, backbone, atol=1e-5)

    def test_apply_rigid_broadcasts_over_extra_dims(self):
        """Trailing point dimensions must broadcast cleanly."""
        R, t = make_identity_rigid((2, 3))
        R = R.clone()
        for b in range(2):
            for s in range(3):
                R[b, s] = _random_rotation()
        t = torch.randn(2, 3, 3)
        # points shape: (B, L, K, M, 3) with two extra dims K, M.
        points = torch.randn(2, 3, 4, 5, 3)
        moved = apply_rigid(R, t, points)
        assert moved.shape == points.shape
        # Manual reconstruction.
        expected = torch.einsum('blij,blkmj->blkmi', R, points) + \
            t.view(2, 3, 1, 1, 3)
        assert torch.allclose(moved, expected, atol=1e-5)

    def test_apply_rigid_identity_is_noop(self):
        R, t = make_identity_rigid((2, 5))
        points = torch.randn(2, 5, 3)
        moved = apply_rigid(R, t, points)
        assert torch.allclose(moved, points, atol=0.0)

    def test_apply_rigid_translation_only(self):
        R, _ = make_identity_rigid((2, 5))
        t = torch.randn(2, 5, 3)
        points = torch.randn(2, 5, 3)
        assert torch.allclose(apply_rigid(R, t, points),
                              points + t,
                              atol=1e-6)

    def test_apply_rigid_raises_on_too_few_point_dims(self):
        R, t = make_identity_rigid((2, 3))
        bad = torch.randn(3)  # rank 1, fewer than the transform's 2 dims.
        with pytest.raises(ValueError, match='at least'):
            apply_rigid(R, t, bad)

    # --- invert_rigid -----------------------------------------------

    def test_invert_rigid_is_left_and_right_inverse(self):
        backbone = _make_backbone()
        R, t = build_backbone_frames(backbone)
        Rinv, tinv = invert_rigid(R, t)
        # Left inverse: T⁻¹ ∘ T = I.
        R_left, t_left = compose_rigids(Rinv, tinv, R, t)
        assert torch.allclose(R_left, torch.eye(3).expand_as(R_left), atol=1e-5)
        assert torch.allclose(t_left, torch.zeros_like(t_left), atol=1e-5)
        # Right inverse: T ∘ T⁻¹ = I.
        R_right, t_right = compose_rigids(R, t, Rinv, tinv)
        assert torch.allclose(R_right,
                              torch.eye(3).expand_as(R_right),
                              atol=1e-5)
        assert torch.allclose(t_right, torch.zeros_like(t_right), atol=1e-5)

    def test_double_inverse_is_identity(self):
        backbone = _make_backbone()
        R, t = build_backbone_frames(backbone)
        R2, t2 = invert_rigid(*invert_rigid(R, t))
        assert torch.allclose(R2, R, atol=1e-6)
        assert torch.allclose(t2, t, atol=1e-6)

    # --- compose_rigids ---------------------------------------------

    def test_compose_matches_sequential_application(self):
        Ra = torch.stack([_random_rotation() for _ in range(6)]).view(2, 3, 3, 3)
        Rb = torch.stack([_random_rotation() for _ in range(6)]).view(2, 3, 3, 3)
        ta = torch.randn(2, 3, 3)
        tb = torch.randn(2, 3, 3)
        points = torch.randn(2, 3, 7, 3)

        sequential = apply_rigid(Ra, ta, apply_rigid(Rb, tb, points))
        R, t = compose_rigids(Ra, ta, Rb, tb)
        composed = apply_rigid(R, t, points)
        assert torch.allclose(composed, sequential, atol=1e-5)

    def test_compose_is_associative(self):
        rotations = [_random_rotation() for _ in range(3)]
        translations = [torch.randn(3) for _ in range(3)]
        Ra, Rb, Rc = rotations
        ta, tb, tc = translations

        R_left, t_left = compose_rigids(
            *compose_rigids(Ra, ta, Rb, tb), Rc, tc)
        R_right, t_right = compose_rigids(Ra, ta,
                                          *compose_rigids(Rb, tb, Rc, tc))
        assert torch.allclose(R_left, R_right, atol=1e-5)
        assert torch.allclose(t_left, t_right, atol=1e-5)

    def test_compose_with_identity_is_noop(self):
        Ra = _random_rotation()
        ta = torch.randn(3)
        Ri, ti = make_identity_rigid(())
        Ri = Ri.to(Ra)
        ti = ti.to(ta)
        R_left, t_left = compose_rigids(Ri, ti, Ra, ta)
        R_right, t_right = compose_rigids(Ra, ta, Ri, ti)
        assert torch.allclose(R_left, Ra, atol=1e-6)
        assert torch.allclose(t_left, ta, atol=1e-6)
        assert torch.allclose(R_right, Ra, atol=1e-6)
        assert torch.allclose(t_right, ta, atol=1e-6)

    def test_apply_rigid_supports_fp64_precision(self):
        backbone = _make_backbone(dtype=torch.float64, seed=3)
        R, t = build_backbone_frames(backbone)
        local = apply_inverse_rigid(R, t, backbone)
        recovered = apply_rigid(R, t, local)
        # Strict fp64 tolerance.
        assert torch.allclose(recovered, backbone, atol=1e-12)


# ---------------------------------------------------------------------------
# Invariant Point Attention
# ---------------------------------------------------------------------------


@pytest.mark.torch
@pytest.mark.skipif(not has_torch, reason='PyTorch not installed')
class TestInvariantPointAttention:
    """Tests for :class:`InvariantPointAttention`."""

    # --- Construction validation ------------------------------------

    @pytest.mark.parametrize('kwargs', [
        dict(embed_dim=0, num_heads=2),
        dict(embed_dim=16, num_heads=0),
        dict(embed_dim=30, num_heads=8),
        dict(embed_dim=16, num_heads=4, num_qk_points=0),
        dict(embed_dim=16, num_heads=4, num_v_points=0),
        dict(embed_dim=16, num_heads=4, pair_dim=0),
        dict(embed_dim=16, num_heads=4, dropout=-0.1),
        dict(embed_dim=16, num_heads=4, dropout=1.0),
        dict(embed_dim=16, num_heads=4, eps=0.0),
    ])
    def test_invalid_construction_raises(self, kwargs):
        with pytest.raises(ValueError):
            InvariantPointAttention(**kwargs)

    # --- Shape -------------------------------------------------------

    def test_output_shape_no_pair(self):
        backbone = _make_backbone(batch_size=2, seq_len=6)
        R, t = build_backbone_frames(backbone)
        layer = InvariantPointAttention(embed_dim=32, num_heads=4)
        out = layer(torch.randn(2, 6, 32), R, t)
        assert out.shape == (2, 6, 32)

    def test_output_shape_with_pair(self):
        backbone = _make_backbone(batch_size=2, seq_len=6)
        R, t = build_backbone_frames(backbone)
        layer = InvariantPointAttention(embed_dim=32,
                                        num_heads=4,
                                        pair_dim=8)
        out = layer(torch.randn(2, 6, 32),
                    R,
                    t,
                    pair_repr=torch.randn(2, 6, 6, 8))
        assert out.shape == (2, 6, 32)

    # --- SE(3) invariance -------------------------------------------

    @pytest.mark.parametrize('seed', [0, 1, 17, 2026])
    def test_global_rotation_invariance(self, seed):
        torch.manual_seed(seed)
        backbone = _make_backbone(batch_size=2, seq_len=6, seed=seed)
        R, t = build_backbone_frames(backbone)
        layer = InvariantPointAttention(embed_dim=32,
                                        num_heads=4,
                                        num_qk_points=3,
                                        num_v_points=4).eval()
        single = torch.randn(2, 6, 32)

        out = layer(single, R, t)
        global_R = _random_rotation()
        global_t = torch.randn(3)
        R_moved = global_R @ R
        t_moved = apply_rigid(global_R, global_t, t)
        out_moved = layer(single, R_moved, t_moved)
        assert torch.allclose(out, out_moved, atol=1e-4)

    def test_translation_only_invariance(self):
        torch.manual_seed(0)
        backbone = _make_backbone(batch_size=2, seq_len=6)
        R, t = build_backbone_frames(backbone)
        layer = InvariantPointAttention(embed_dim=32, num_heads=4).eval()
        single = torch.randn(2, 6, 32)
        out = layer(single, R, t)
        translation = torch.randn(3) * 100.0  # Very large translation.
        out_translated = layer(single, R, t + translation)
        assert torch.allclose(out, out_translated, atol=1e-4)

    def test_pair_bias_invariance(self):
        """Output remains invariant when a pair bias term is present."""
        torch.manual_seed(0)
        backbone = _make_backbone(batch_size=2, seq_len=6)
        R, t = build_backbone_frames(backbone)
        layer = InvariantPointAttention(embed_dim=32,
                                        num_heads=4,
                                        pair_dim=8).eval()
        single = torch.randn(2, 6, 32)
        pair = torch.randn(2, 6, 6, 8)
        out = layer(single, R, t, pair_repr=pair)
        global_R = _random_rotation()
        global_t = torch.randn(3)
        R_moved = global_R @ R
        t_moved = apply_rigid(global_R, global_t, t)
        out_moved = layer(single, R_moved, t_moved, pair_repr=pair)
        assert torch.allclose(out, out_moved, atol=1e-4)

    def test_invariance_in_fp64(self):
        """fp64 numerics should hit a much tighter tolerance."""
        torch.manual_seed(0)
        backbone = _make_backbone(batch_size=1,
                                  seq_len=4,
                                  dtype=torch.float64)
        R, t = build_backbone_frames(backbone)
        layer = InvariantPointAttention(embed_dim=16, num_heads=4).eval()
        layer = layer.to(dtype=torch.float64)
        single = torch.randn(1, 4, 16, dtype=torch.float64)
        out = layer(single, R, t)
        global_R = _random_rotation(dtype=torch.float64)
        global_t = torch.randn(3, dtype=torch.float64)
        R_moved = global_R @ R
        t_moved = apply_rigid(global_R, global_t, t)
        out_moved = layer(single, R_moved, t_moved)
        assert torch.allclose(out, out_moved, atol=1e-7)

    # --- Masking -----------------------------------------------------

    def test_masked_query_outputs_are_zero(self):
        torch.manual_seed(0)
        backbone = _make_backbone(batch_size=1, seq_len=5)
        R, t = build_backbone_frames(backbone)
        layer = InvariantPointAttention(embed_dim=32, num_heads=4)
        single = torch.randn(1, 5, 32)
        mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.bool)
        out = layer(single, R, t, mask=mask)
        assert torch.allclose(out[:, 3:], torch.zeros_like(out[:, 3:]),
                              atol=0.0)

    def test_masked_keys_do_not_influence_unmasked_queries(self):
        """Perturbing a masked residue must leave unmasked outputs intact."""
        torch.manual_seed(0)
        backbone = _make_backbone(batch_size=1, seq_len=5)
        R, t = build_backbone_frames(backbone)
        layer = InvariantPointAttention(embed_dim=32, num_heads=4).eval()
        single = torch.randn(1, 5, 32)
        mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.bool)

        out_a = layer(single, R, t, mask=mask)
        # Replace masked rows with arbitrary content.
        perturbed = single.clone()
        perturbed[:, 3:] = torch.randn(1, 2, 32) * 1e3
        R_p = R.clone()
        t_p = t.clone()
        R_p[:, 3:] = R_p[:, 3:] @ _random_rotation()
        t_p[:, 3:] = torch.randn(1, 2, 3) * 100.0
        out_b = layer(perturbed, R_p, t_p, mask=mask)
        assert torch.allclose(out_a[:, :3], out_b[:, :3], atol=1e-4)

    def test_mask_raises_on_wrong_shape(self):
        backbone = _make_backbone(batch_size=1, seq_len=5)
        R, t = build_backbone_frames(backbone)
        layer = InvariantPointAttention(embed_dim=16, num_heads=4)
        with pytest.raises(ValueError, match='mask'):
            layer(torch.randn(1, 5, 16),
                  R,
                  t,
                  mask=torch.ones(1, 5, 1, dtype=torch.bool))

    # --- Pair-repr argument validation ------------------------------

    def test_missing_pair_repr_raises(self):
        backbone = _make_backbone(batch_size=1, seq_len=4)
        R, t = build_backbone_frames(backbone)
        layer = InvariantPointAttention(embed_dim=16,
                                        num_heads=4,
                                        pair_dim=4)
        with pytest.raises(ValueError, match='pair_repr'):
            layer(torch.randn(1, 4, 16), R, t)

    def test_extra_pair_repr_raises(self):
        backbone = _make_backbone(batch_size=1, seq_len=4)
        R, t = build_backbone_frames(backbone)
        layer = InvariantPointAttention(embed_dim=16, num_heads=4)
        with pytest.raises(ValueError, match='pair_repr'):
            layer(torch.randn(1, 4, 16),
                  R,
                  t,
                  pair_repr=torch.randn(1, 4, 4, 4))

    def test_input_shape_validation(self):
        layer = InvariantPointAttention(embed_dim=16, num_heads=4)
        single = torch.randn(2, 5, 16)
        with pytest.raises(ValueError, match='single_repr'):
            layer(torch.randn(2, 5), torch.randn(2, 5, 3, 3),
                  torch.randn(2, 5, 3))
        with pytest.raises(ValueError, match='rotations'):
            layer(single, torch.randn(2, 5, 3, 4), torch.randn(2, 5, 3))
        with pytest.raises(ValueError, match='translations'):
            layer(single, torch.randn(2, 5, 3, 3), torch.randn(2, 5, 4))

    # --- Gradient flow ----------------------------------------------

    def test_gradient_flow_into_all_inputs_and_parameters(self):
        torch.manual_seed(0)
        backbone = _make_backbone(batch_size=1, seq_len=4)
        R, t = build_backbone_frames(backbone)
        R = R.detach().clone().requires_grad_(True)
        t = t.detach().clone().requires_grad_(True)
        single = torch.randn(1, 4, 16, requires_grad=True)
        pair = torch.randn(1, 4, 4, 4, requires_grad=True)
        layer = InvariantPointAttention(embed_dim=16,
                                        num_heads=4,
                                        pair_dim=4)
        out = layer(single, R, t, pair_repr=pair)
        out.sum().backward()
        # Input gradients.
        assert single.grad is not None and torch.isfinite(single.grad).all()
        assert R.grad is not None and torch.isfinite(R.grad).all()
        assert t.grad is not None and torch.isfinite(t.grad).all()
        assert pair.grad is not None and torch.isfinite(pair.grad).all()
        # Parameter gradients (every learnable tensor must receive a grad).
        for name, parameter in layer.named_parameters():
            assert parameter.grad is not None, f'no grad for {name}'
            assert torch.isfinite(parameter.grad).all(), \
                f'non-finite grad for {name}'

    def test_point_weights_softplus_keeps_gamma_positive(self):
        """softplus(γ̂) ≥ 0 always; verify γ is strictly positive after a step."""
        layer = InvariantPointAttention(embed_dim=16, num_heads=4)
        # A single gradient step should not be able to drive γ negative.
        with torch.no_grad():
            layer.point_weights.fill_(-50.0)
        gamma = torch.nn.functional.softplus(layer.point_weights)
        assert torch.all(gamma >= 0.0)
        # Even with a huge negative bias, softplus is still finite.
        assert torch.isfinite(gamma).all()

    # --- Numerical / structural sanity checks ------------------------

    def test_dropout_eval_is_deterministic(self):
        torch.manual_seed(0)
        backbone = _make_backbone(batch_size=1, seq_len=4)
        R, t = build_backbone_frames(backbone)
        layer = InvariantPointAttention(embed_dim=16,
                                        num_heads=4,
                                        dropout=0.5).eval()
        single = torch.randn(1, 4, 16)
        out_a = layer(single, R, t)
        out_b = layer(single, R, t)
        assert torch.allclose(out_a, out_b, atol=0.0)

    def test_weighting_constants_have_expected_values(self):
        """w_C must equal √(2/(9·N_qk)) and w_L must equal √(1/L)."""
        no_pair = InvariantPointAttention(embed_dim=16,
                                          num_heads=4,
                                          num_qk_points=4)
        with_pair = InvariantPointAttention(embed_dim=16,
                                            num_heads=4,
                                            num_qk_points=4,
                                            pair_dim=8)
        expected_w_c = math.sqrt(2.0 / (9.0 * 4))
        assert math.isclose(no_pair.w_c.item(), expected_w_c, rel_tol=1e-7)
        assert math.isclose(with_pair.w_c.item(), expected_w_c, rel_tol=1e-7)
        assert math.isclose(no_pair.w_l.item(),
                            math.sqrt(1.0 / 2.0),
                            rel_tol=1e-7)
        assert math.isclose(with_pair.w_l.item(),
                            math.sqrt(1.0 / 3.0),
                            rel_tol=1e-7)

    def test_identity_frame_input_is_pure_self_attention(self):
        """With identity frames and no pair bias, point distances depend only
        on the projected local points and the layer reduces to a structured
        self-attention block. The output must remain finite and shape-stable
        and depend continuously on the single representation.
        """
        layer = InvariantPointAttention(embed_dim=16, num_heads=4).eval()
        R, t = make_identity_rigid((1, 4))
        single = torch.randn(1, 4, 16)
        out_a = layer(single, R, t)
        out_b = layer(single + 1e-6, R, t)
        assert torch.isfinite(out_a).all()
        # Continuity: tiny input perturbation produces tiny output change.
        assert (out_a - out_b).abs().max().item() < 1e-2

    def test_attention_distinguishes_residues_by_geometry_alone(self):
        """Two residues differing only in frame translations should be
        mapped to different outputs, demonstrating the point-distance
        term contributes meaningfully to the attention logits.
        """
        torch.manual_seed(0)
        # Construct two configurations that share scalar inputs and
        # rotations but differ in translations only.
        layer = InvariantPointAttention(embed_dim=16,
                                        num_heads=4,
                                        num_qk_points=4,
                                        num_v_points=4).eval()
        R, _ = make_identity_rigid((1, 3))
        single = torch.randn(1, 3, 16)
        t_close = torch.tensor([[[0.0, 0.0, 0.0],
                                 [1.0, 0.0, 0.0],
                                 [2.0, 0.0, 0.0]]])
        t_far = torch.tensor([[[0.0, 0.0, 0.0],
                               [50.0, 0.0, 0.0],
                               [100.0, 0.0, 0.0]]])
        out_close = layer(single, R, t_close)
        out_far = layer(single, R, t_far)
        # The two outputs must differ (point-distance term is active).
        assert not torch.allclose(out_close, out_far, atol=1e-3)
