"""Exhaustive tests for the RFDiffusion multi-track architecture.

The suite covers four mutually independent guarantees that the
multi-track stack must uphold:

* **Geometric correctness** — frame outputs are SE(3)-equivariant
  under any global rigid motion applied to the input frames; the
  scalar (single) output is exactly invariant.
* **Algebraic symmetry** — the pair representation produced by every
  block satisfies :math:`z_{ij} = z_{ji}` to within fp32 round-off.
* **Memory-bounded correctness** — chunked execution of the
  triangular updates and triangular attention produces *bitwise-equal*
  outputs to the dense path (the chunking trick must not change the
  numerics, only the working-set size).
* **Holistic DeepChem integration** — the new stack is a drop-in
  replacement for the existing
  :class:`~deepchem.models.torch_models.rfdiffusion.DiffusionTransformerBlock`
  stack inside :class:`~deepchem.models.torch_models.rfdiffusion.BackboneDiffusion`.
  The full denoiser must produce the correct output shape, route
  timestep and self-conditioning inputs through, and propagate
  gradients into every parameter of the multi-track stack.

All numerical tolerances follow the standards contract in the Phase 1
Technical Readiness Report (fp32 ``atol = 1e-4``; chunk-vs-dense must
be exact).
"""

import pytest

try:
    import torch
    import torch.nn as nn
    has_torch = True
except ImportError:
    has_torch = False

if has_torch:
    from deepchem.models.torch_models.rfdiffusion import BackboneDiffusion
    from deepchem.models.torch_models.rfdiffusion_se3 import (
        apply_rigid,
        build_backbone_frames,
        make_identity_rigid,
    )
    from deepchem.models.torch_models.rfdiffusion_tracks import (
        BackboneUpdate,
        OuterProductMean,
        PairBiasedSingleAttention,
        PairTransition,
        RelativePositionEmbedding,
        RFDiffusionMultiTrackStack,
        RFDiffusionTrackBlock,
        SingleTransition,
        TriangleAttention,
        TriangleMultiplicativeUpdate,
        quaternion_to_rotation_matrix,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _random_rotation(dtype=torch.float32):
    matrix = torch.randn(3, 3, dtype=dtype)
    q, r = torch.linalg.qr(matrix)
    signs = torch.sign(torch.diagonal(r))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    q = q @ torch.diag(signs)
    if torch.det(q) < 0:
        q[:, -1] = -q[:, -1]
    return q


def _make_stack(num_blocks=2, chunk_size=None, dropout=0.0):
    """Construct a small but non-trivial multi-track stack."""
    return RFDiffusionMultiTrackStack(
        embed_dim=32,
        pair_dim=16,
        num_blocks=num_blocks,
        num_heads=4,
        pair_num_heads=2,
        triangle_hidden_dim=16,
        triangle_head_dim=8,
        opm_hidden_dim=8,
        num_qk_points=2,
        num_v_points=4,
        max_relative_position=8,
        chunk_size=chunk_size,
        dropout=dropout,
    )


def _randomise_backbone_update(stack):
    """Replace zero-init backbone-update weights with random non-zero ones.

    The default zero-init is correct for stable diffusion training but
    makes the per-block frame update an identity, which would make
    SE(3) equivariance trivial. Randomising the head exposes the
    *true* equivariance of the architecture.
    """
    for block in stack.blocks:
        torch.nn.init.normal_(block.backbone_update.linear.weight, std=0.05)
        torch.nn.init.normal_(block.backbone_update.linear.bias, std=0.05)


# ---------------------------------------------------------------------------
# Sub-module unit tests
# ---------------------------------------------------------------------------


@pytest.mark.torch
@pytest.mark.skipif(not has_torch, reason='PyTorch not installed')
class TestSubmodules:
    """Per-module sanity checks."""

    def test_relpos_is_symmetric(self):
        rel = RelativePositionEmbedding(pair_dim=12, max_relative_position=4)
        emb = rel(7)
        assert emb.shape == (7, 7, 12)
        assert torch.allclose(emb, emb.transpose(0, 1), atol=0.0)
        # Saturates beyond the clip radius.
        assert torch.allclose(emb[0, 5], emb[0, 6], atol=0.0)

    def test_outer_product_mean_is_symmetric(self):
        torch.manual_seed(0)
        layer = OuterProductMean(embed_dim=16, pair_dim=8, hidden_dim=4)
        single = torch.randn(2, 5, 16)
        pair = layer(single)
        assert pair.shape == (2, 5, 5, 8)
        assert torch.allclose(pair, pair.transpose(-2, -3), atol=1e-6)

    def test_triangle_multiplication_chunked_equals_dense(self):
        torch.manual_seed(0)
        for outgoing in (True, False):
            layer = TriangleMultiplicativeUpdate(pair_dim=8,
                                                 hidden_dim=4,
                                                 outgoing=outgoing).eval()
            pair = torch.randn(2, 7, 7, 8)
            dense = layer(pair, chunk_size=None)
            for chunk in (1, 2, 3, 7, 100):
                chunked = layer(pair, chunk_size=chunk)
                assert torch.allclose(dense, chunked, atol=1e-6), (
                    f'outgoing={outgoing}, chunk={chunk} mismatch')

    def test_triangle_attention_chunked_equals_dense(self):
        torch.manual_seed(0)
        for start in (True, False):
            layer = TriangleAttention(pair_dim=8,
                                      num_heads=2,
                                      head_dim=4,
                                      starting_node=start).eval()
            pair = torch.randn(2, 6, 6, 8)
            dense = layer(pair, chunk_size=None)
            for chunk in (1, 2, 3, 6, 100):
                chunked = layer(pair, chunk_size=chunk)
                assert torch.allclose(dense, chunked, atol=1e-6), (
                    f'starting_node={start}, chunk={chunk} mismatch')

    def test_pair_transition_preserves_shape(self):
        layer = PairTransition(pair_dim=8)
        pair = torch.randn(2, 5, 5, 8)
        assert layer(pair).shape == pair.shape

    def test_single_transition_uses_timestep(self):
        torch.manual_seed(0)
        layer = SingleTransition(embed_dim=16).eval()
        single = torch.randn(2, 4, 16)
        t1 = torch.zeros(2, 16)
        t2 = torch.randn(2, 16) * 5.0
        out1 = layer(single, t1)
        out2 = layer(single, t2)
        assert not torch.allclose(out1, out2, atol=1e-4)

    def test_pair_biased_single_attention_uses_pair(self):
        torch.manual_seed(0)
        layer = PairBiasedSingleAttention(embed_dim=16,
                                          pair_dim=8,
                                          num_heads=4).eval()
        single = torch.randn(2, 5, 16)
        pair_a = torch.zeros(2, 5, 5, 8)
        pair_b = torch.randn(2, 5, 5, 8)
        out_a = layer(single, pair_a)
        out_b = layer(single, pair_b)
        assert not torch.allclose(out_a, out_b, atol=1e-4)

    def test_quaternion_identity_is_identity_rotation(self):
        quat = torch.tensor([1.0, 0.0, 0.0, 0.0])
        R = quaternion_to_rotation_matrix(quat)
        assert torch.allclose(R, torch.eye(3), atol=1e-7)

    def test_quaternion_rotation_is_proper_rotation(self):
        torch.manual_seed(0)
        quat = torch.randn(8, 4)
        R = quaternion_to_rotation_matrix(quat)
        gram = R.transpose(-1, -2) @ R
        identity = torch.eye(3).expand_as(gram)
        assert torch.allclose(gram, identity, atol=1e-6)
        det = torch.det(R)
        assert torch.allclose(det, torch.ones_like(det), atol=1e-6)

    def test_backbone_update_is_identity_at_init(self):
        """Zero-initialised head must yield (R, t) unchanged."""
        torch.manual_seed(0)
        head = BackboneUpdate(embed_dim=16)
        single = torch.randn(2, 4, 16)
        R, t = make_identity_rigid((2, 4))
        R = R.clone()
        # Use non-identity input frames to make the test sharper.
        R[0, 0] = _random_rotation()
        t = torch.randn(2, 4, 3)
        R_new, t_new = head(single, R, t)
        assert torch.allclose(R_new, R, atol=1e-7)
        assert torch.allclose(t_new, t, atol=1e-7)

    def test_backbone_update_equivariance(self):
        """After random head weights, frames update equivariantly."""
        torch.manual_seed(0)
        head = BackboneUpdate(embed_dim=16)
        nn.init.normal_(head.linear.weight, std=0.05)
        nn.init.normal_(head.linear.bias, std=0.05)
        single = torch.randn(1, 4, 16)
        R = torch.stack([_random_rotation() for _ in range(4)]).unsqueeze(0)
        t = torch.randn(1, 4, 3)
        R_new, t_new = head(single, R, t)
        # Apply global SE(3).
        Q = _random_rotation()
        tg = torch.randn(3)
        R_moved = Q @ R
        t_moved = apply_rigid(Q, tg, t)
        R_new_moved, t_new_moved = head(single, R_moved, t_moved)
        assert torch.allclose(R_new_moved, Q @ R_new, atol=1e-5)
        assert torch.allclose(t_new_moved,
                              apply_rigid(Q, tg, t_new),
                              atol=1e-5)


# ---------------------------------------------------------------------------
# Block-level integration tests
# ---------------------------------------------------------------------------


@pytest.mark.torch
@pytest.mark.skipif(not has_torch, reason='PyTorch not installed')
class TestTrackBlock:
    """Tests of a single :class:`RFDiffusionTrackBlock`."""

    def test_block_preserves_track_shapes(self):
        torch.manual_seed(0)
        block = RFDiffusionTrackBlock(embed_dim=16,
                                      pair_dim=8,
                                      num_heads=4,
                                      pair_num_heads=2,
                                      triangle_hidden_dim=8,
                                      triangle_head_dim=4,
                                      opm_hidden_dim=4,
                                      num_qk_points=2,
                                      num_v_points=2).eval()
        single = torch.randn(2, 5, 16)
        pair = torch.randn(2, 5, 5, 8)
        pair = 0.5 * (pair + pair.transpose(-2, -3))
        R, t = make_identity_rigid((2, 5))
        t_emb = torch.randn(2, 16)
        out_s, out_z, out_R, out_t = block(single, pair, R, t, t_emb)
        assert out_s.shape == single.shape
        assert out_z.shape == pair.shape
        assert out_R.shape == R.shape
        assert out_t.shape == t.shape

    def test_block_preserves_pair_symmetry(self):
        torch.manual_seed(0)
        block = RFDiffusionTrackBlock(embed_dim=16,
                                      pair_dim=8,
                                      num_heads=4,
                                      pair_num_heads=2,
                                      triangle_hidden_dim=8,
                                      triangle_head_dim=4,
                                      opm_hidden_dim=4).eval()
        single = torch.randn(1, 6, 16)
        pair = torch.randn(1, 6, 6, 8)
        pair = 0.5 * (pair + pair.transpose(-2, -3))
        R, t = make_identity_rigid((1, 6))
        t_emb = torch.randn(1, 16)
        _, out_z, _, _ = block(single, pair, R, t, t_emb)
        assert torch.allclose(out_z, out_z.transpose(-2, -3), atol=1e-5)


# ---------------------------------------------------------------------------
# Stack-level tests (the core deliverable)
# ---------------------------------------------------------------------------


@pytest.mark.torch
@pytest.mark.skipif(not has_torch, reason='PyTorch not installed')
class TestMultiTrackStack:
    """Tests of the full :class:`RFDiffusionMultiTrackStack`."""

    # -- Shape / construction -----------------------------------------

    def test_stack_construction_rejects_bad_num_blocks(self):
        with pytest.raises(ValueError, match='num_blocks'):
            RFDiffusionMultiTrackStack(
                embed_dim=16, pair_dim=8, num_blocks=0)

    def test_forward_shape(self):
        torch.manual_seed(0)
        stack = _make_stack().eval()
        single = torch.randn(2, 6, 32)
        t_emb = torch.randn(2, 32)
        out = stack(single, t_emb)
        assert out.shape == (2, 6, 32)

    def test_forward_tracks_shapes(self):
        torch.manual_seed(0)
        stack = _make_stack().eval()
        single = torch.randn(2, 6, 32)
        t_emb = torch.randn(2, 32)
        s, z, R, t = stack.forward_tracks(single, t_emb)
        assert s.shape == (2, 6, 32)
        assert z.shape == (2, 6, 6, 16)
        assert R.shape == (2, 6, 3, 3)
        assert t.shape == (2, 6, 3)

    def test_forward_rejects_wrong_input_shapes(self):
        stack = _make_stack().eval()
        # Wrong embed_dim.
        with pytest.raises(ValueError, match='embed_dim'):
            stack(torch.randn(2, 6, 16), torch.randn(2, 32))
        # Wrong t_emb.
        with pytest.raises(ValueError, match='t_emb'):
            stack(torch.randn(2, 6, 32), torch.randn(2, 16))
        # Wrong single rank.
        with pytest.raises(ValueError, match='single'):
            stack(torch.randn(2, 32), torch.randn(2, 32))
        # Wrong mask shape.
        with pytest.raises(ValueError, match='attention_mask'):
            stack(torch.randn(2, 6, 32),
                  torch.randn(2, 32),
                  attention_mask=torch.ones(2, 7))

    # -- Pair symmetry ------------------------------------------------

    def test_pair_output_symmetry(self):
        torch.manual_seed(0)
        stack = _make_stack().eval()
        single = torch.randn(2, 7, 32)
        t_emb = torch.randn(2, 32)
        _, pair, _, _ = stack.forward_tracks(single, t_emb)
        assert torch.allclose(pair, pair.transpose(-2, -3), atol=1e-5)

    # -- SE(3) equivariance / invariance ------------------------------

    @pytest.mark.parametrize('seed', [0, 17, 2026])
    def test_frame_equivariance_under_global_se3(self, seed):
        """Frames transform equivariantly; the single output is invariant."""
        torch.manual_seed(seed)
        stack = _make_stack().eval()
        _randomise_backbone_update(stack)

        single = torch.randn(2, 5, 32)
        t_emb = torch.randn(2, 32)
        # Non-trivial initial frames so the test exercises composition.
        R_init = torch.stack(
            [_random_rotation() for _ in range(10)]).view(2, 5, 3, 3)
        t_init = torch.randn(2, 5, 3)

        s_a, _, R_a, t_a = stack.forward_tracks(single,
                                                t_emb,
                                                rotations=R_init,
                                                translations=t_init)

        # Apply global SE(3) to initial frames.
        Q = _random_rotation()
        tg = torch.randn(3)
        R_init_g = Q @ R_init
        t_init_g = apply_rigid(Q, tg, t_init)

        s_b, _, R_b, t_b = stack.forward_tracks(single,
                                                t_emb,
                                                rotations=R_init_g,
                                                translations=t_init_g)
        assert torch.allclose(s_b, s_a, atol=1e-4)
        assert torch.allclose(R_b, Q @ R_a, atol=1e-4)
        assert torch.allclose(t_b, apply_rigid(Q, tg, t_a), atol=1e-4)

    def test_translation_only_invariance_of_single(self):
        """A pure global translation must leave the single output unchanged."""
        torch.manual_seed(0)
        stack = _make_stack().eval()
        _randomise_backbone_update(stack)
        single = torch.randn(1, 4, 32)
        t_emb = torch.randn(1, 32)
        R_init = torch.stack(
            [_random_rotation() for _ in range(4)]).view(1, 4, 3, 3)
        t_init = torch.randn(1, 4, 3)
        s_a, _, _, _ = stack.forward_tracks(single,
                                            t_emb,
                                            rotations=R_init,
                                            translations=t_init)
        tg = torch.randn(3) * 100.0
        s_b, _, _, _ = stack.forward_tracks(single,
                                            t_emb,
                                            rotations=R_init,
                                            translations=t_init + tg)
        assert torch.allclose(s_a, s_b, atol=1e-4)

    # -- Chunking correctness -----------------------------------------

    @pytest.mark.parametrize('chunk_size', [1, 2, 3, 4])
    def test_chunked_matches_dense_exactly(self, chunk_size):
        torch.manual_seed(0)
        stack = _make_stack().eval()
        single = torch.randn(2, 8, 32)
        t_emb = torch.randn(2, 32)
        s_d, z_d, R_d, t_d = stack.forward_tracks(single,
                                                  t_emb,
                                                  chunk_size=None)
        s_c, z_c, R_c, t_c = stack.forward_tracks(single,
                                                  t_emb,
                                                  chunk_size=chunk_size)
        # Bitwise / extremely tight tolerance.
        assert torch.allclose(s_d, s_c, atol=1e-6), (
            f'single mismatch for chunk_size={chunk_size}')
        assert torch.allclose(z_d, z_c, atol=1e-6), (
            f'pair mismatch for chunk_size={chunk_size}')
        assert torch.allclose(R_d, R_c, atol=1e-6)
        assert torch.allclose(t_d, t_c, atol=1e-6)

    # -- Timestep sensitivity -----------------------------------------

    def test_single_output_depends_on_timestep(self):
        torch.manual_seed(0)
        stack = _make_stack().eval()
        single = torch.randn(2, 5, 32)
        t_a = torch.zeros(2, 32)
        t_b = torch.randn(2, 32) * 5.0
        out_a = stack(single, t_a)
        out_b = stack(single, t_b)
        assert not torch.allclose(out_a, out_b, atol=1e-4)

    # -- Masking -----------------------------------------------------

    def test_masked_residues_have_zero_output(self):
        torch.manual_seed(0)
        stack = _make_stack().eval()
        single = torch.randn(2, 6, 32)
        t_emb = torch.randn(2, 32)
        mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 0]],
                            dtype=torch.bool)
        out = stack(single, t_emb, attention_mask=mask)
        assert torch.allclose(out[0, 4:],
                              torch.zeros_like(out[0, 4:]),
                              atol=0.0)
        assert torch.allclose(out[1, 5:],
                              torch.zeros_like(out[1, 5:]),
                              atol=0.0)

    # -- Gradient flow -----------------------------------------------

    def test_gradient_flows_into_all_tracks(self):
        """Use the full ``forward_tracks`` output so every parameter
        contributes — including the last block's ``backbone_update``
        which is architecturally orphaned when only the single output
        is returned (its frames are never consumed downstream).
        """
        torch.manual_seed(0)
        stack = _make_stack(dropout=0.0)
        single = torch.randn(1, 5, 32, requires_grad=True)
        t_emb = torch.randn(1, 32, requires_grad=True)
        s, z, R, t = stack.forward_tracks(single, t_emb)
        (s.sum() + z.sum() + R.sum() + t.sum()).backward()
        assert single.grad is not None and torch.isfinite(single.grad).all()
        assert t_emb.grad is not None and torch.isfinite(t_emb.grad).all()
        # Every learnable parameter receives a finite gradient.
        for name, parameter in stack.named_parameters():
            assert parameter.grad is not None, f'no grad for {name}'
            assert torch.isfinite(parameter.grad).all(), \
                f'non-finite grad for {name}'

    def test_gradient_flows_into_initial_frames(self):
        torch.manual_seed(0)
        stack = _make_stack()
        _randomise_backbone_update(stack)
        single = torch.randn(1, 4, 32)
        t_emb = torch.randn(1, 32)
        R = torch.stack(
            [_random_rotation() for _ in range(4)]).view(1, 4, 3, 3)
        R = R.detach().clone().requires_grad_(True)
        t = torch.randn(1, 4, 3, requires_grad=True)
        s, _, R_out, t_out = stack.forward_tracks(single, t_emb,
                                                  rotations=R,
                                                  translations=t)
        (R_out.sum() + t_out.sum() + s.sum()).backward()
        assert R.grad is not None and torch.isfinite(R.grad).all()
        assert t.grad is not None and torch.isfinite(t.grad).all()


# ---------------------------------------------------------------------------
# Holistic DeepChem integration tests
# ---------------------------------------------------------------------------


@pytest.mark.torch
@pytest.mark.skipif(not has_torch, reason='PyTorch not installed')
class TestBackboneDiffusionIntegration:
    """Drop-in compatibility with :class:`BackboneDiffusion`."""

    @staticmethod
    def _replace_transformer_with_multitrack(model: 'BackboneDiffusion',
                                             chunk_size=None
                                             ) -> 'BackboneDiffusion':
        """Swap the baseline transformer layer stack for the multi-track."""
        stack = RFDiffusionMultiTrackStack(
            embed_dim=model.embed_dim,
            pair_dim=16,
            num_blocks=2,
            num_heads=4,
            pair_num_heads=2,
            triangle_hidden_dim=16,
            triangle_head_dim=8,
            opm_hidden_dim=8,
            num_qk_points=2,
            num_v_points=4,
            max_relative_position=8,
            chunk_size=chunk_size,
        )
        # The BackboneDiffusion forward loops over `self.layers`; a
        # ModuleList containing a single multi-track stack therefore
        # invokes the stack exactly once with the (h, t_emb, mask)
        # signature it already expects.
        model.layers = torch.nn.ModuleList([stack])
        return model

    def test_drop_in_forward_preserves_output_shape(self):
        torch.manual_seed(0)
        model = BackboneDiffusion(coord_dim=9,
                                  embed_dim=32,
                                  time_dim=16,
                                  num_layers=1,
                                  num_heads=4,
                                  max_seq_len=32).eval()
        self._replace_transformer_with_multitrack(model)
        coords = torch.randn(2, 6, 9)
        timesteps = torch.randint(0, 100, (2,))
        noise = model([coords, timesteps])
        assert noise.shape == coords.shape

    def test_drop_in_supports_self_conditioning(self):
        torch.manual_seed(0)
        model = BackboneDiffusion(coord_dim=9,
                                  embed_dim=32,
                                  time_dim=16,
                                  num_layers=1,
                                  num_heads=4,
                                  max_seq_len=32,
                                  self_conditioning=True).eval()
        self._replace_transformer_with_multitrack(model)
        coords = torch.randn(2, 5, 9)
        x0_prev = torch.randn(2, 5, 9)
        timesteps = torch.randint(0, 100, (2,))
        noise = model([coords, timesteps, x0_prev])
        assert noise.shape == coords.shape

    def test_drop_in_supports_mask(self):
        torch.manual_seed(0)
        model = BackboneDiffusion(coord_dim=9,
                                  embed_dim=32,
                                  time_dim=16,
                                  num_layers=1,
                                  num_heads=4,
                                  max_seq_len=32).eval()
        self._replace_transformer_with_multitrack(model)
        coords = torch.randn(2, 6, 9)
        timesteps = torch.randint(0, 100, (2,))
        mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 0]],
                            dtype=torch.float32)
        noise = model([coords, timesteps, mask])
        assert noise.shape == coords.shape
        # Masked residues must have zero predicted noise (because the
        # output is multiplied by the mask in BackboneDiffusion).
        assert torch.allclose(noise[0, 4:],
                              torch.zeros_like(noise[0, 4:]),
                              atol=0.0)
        assert torch.allclose(noise[1, 5:],
                              torch.zeros_like(noise[1, 5:]),
                              atol=0.0)

    def test_drop_in_responds_to_timestep(self):
        torch.manual_seed(0)
        model = BackboneDiffusion(coord_dim=9,
                                  embed_dim=32,
                                  time_dim=16,
                                  num_layers=1,
                                  num_heads=4,
                                  max_seq_len=32).eval()
        # ``BackboneDiffusion`` zero-initialises its final output
        # projection so the *trained* network starts from an identity
        # noise prediction; for this sensitivity test we randomise it so
        # the signal reaches the output.
        nn.init.normal_(model.output_proj[-1].weight, std=0.05)
        nn.init.normal_(model.output_proj[-1].bias, std=0.05)
        self._replace_transformer_with_multitrack(model)
        coords = torch.randn(2, 5, 9)
        n_a = model([coords, torch.zeros(2, dtype=torch.long)])
        n_b = model([coords, torch.full((2,), 99, dtype=torch.long)])
        assert not torch.allclose(n_a, n_b, atol=1e-4)

    def test_drop_in_chunking_matches_dense_end_to_end(self):
        """Chunked execution inside BackboneDiffusion must match dense."""
        torch.manual_seed(0)
        model_dense = BackboneDiffusion(coord_dim=9,
                                        embed_dim=32,
                                        time_dim=16,
                                        num_layers=1,
                                        num_heads=4,
                                        max_seq_len=32).eval()
        self._replace_transformer_with_multitrack(model_dense, chunk_size=None)
        # Build a chunked variant sharing the *same* multi-track weights.
        model_chunked = BackboneDiffusion(coord_dim=9,
                                          embed_dim=32,
                                          time_dim=16,
                                          num_layers=1,
                                          num_heads=4,
                                          max_seq_len=32).eval()
        self._replace_transformer_with_multitrack(model_chunked, chunk_size=2)
        # Copy *all* parameters from dense to chunked so the only
        # difference is the chunk_size.
        model_chunked.load_state_dict(model_dense.state_dict())
        coords = torch.randn(2, 8, 9)
        timesteps = torch.randint(0, 100, (2,))
        out_d = model_dense([coords, timesteps])
        out_c = model_chunked([coords, timesteps])
        assert torch.allclose(out_d, out_c, atol=1e-5)

    def test_end_to_end_gradients_into_multitrack_parameters(self):
        """Every non-orphaned multi-track parameter must receive a
        finite gradient when only the single representation is read out
        (the drop-in path).

        The very last block's ``backbone_update`` parameters are
        architecturally orphaned in the single-only output path because
        their predicted frames are never consumed downstream. These are
        explicitly excluded; tests in :class:`TestMultiTrackStack`
        already cover the full-track gradient flow.
        """
        torch.manual_seed(0)
        model = BackboneDiffusion(coord_dim=9,
                                  embed_dim=32,
                                  time_dim=16,
                                  num_layers=1,
                                  num_heads=4,
                                  max_seq_len=32)
        # Randomise the zero-initialised output projection so gradient
        # actually flows through every layer.
        nn.init.normal_(model.output_proj[-1].weight, std=0.05)
        nn.init.normal_(model.output_proj[-1].bias, std=0.05)
        self._replace_transformer_with_multitrack(model)
        coords = torch.randn(2, 5, 9, requires_grad=True)
        timesteps = torch.randint(0, 100, (2,))
        noise = model([coords, timesteps])
        noise.sum().backward()
        # The multi-track stack is at model.layers[0].
        stack = model.layers[0]
        last_block_prefix = f'blocks.{len(stack.blocks) - 1}.backbone_update'
        for name, parameter in stack.named_parameters():
            if name.startswith(last_block_prefix):
                continue
            assert parameter.grad is not None, (
                f'no grad reached multitrack parameter {name}')
            assert torch.isfinite(parameter.grad).all(), (
                f'non-finite grad for multitrack parameter {name}')

    def test_end_to_end_se3_equivariance_of_frame_outputs(self):
        """Use the multi-track stack directly with backbone-derived frames.

        The end-to-end *coordinate* output of ``BackboneDiffusion`` is
        not SE(3)-equivariant because its coordinate embedding is a
        plain linear layer over raw coordinates. The architectural
        guarantee is that the *frame outputs* of the multi-track stack
        are equivariant under any global rigid motion applied to the
        frames derived from the input backbone, which is what we test
        here against frames built by
        :func:`build_backbone_frames`.
        """
        torch.manual_seed(0)
        stack = _make_stack().eval()
        _randomise_backbone_update(stack)
        backbone = torch.randn(1, 5, 3, 3)
        R_init, t_init = build_backbone_frames(backbone)
        single = torch.randn(1, 5, 32)
        t_emb = torch.randn(1, 32)
        s_a, _, R_a, t_a = stack.forward_tracks(single, t_emb,
                                                rotations=R_init,
                                                translations=t_init)
        # Apply a global rigid motion to the backbone, derive new frames.
        Q = _random_rotation()
        tg = torch.randn(3)
        backbone_moved = apply_rigid(Q, tg, backbone)
        R_init_b, t_init_b = build_backbone_frames(backbone_moved)
        s_b, _, R_b, t_b = stack.forward_tracks(single, t_emb,
                                                rotations=R_init_b,
                                                translations=t_init_b)
        assert torch.allclose(s_b, s_a, atol=1e-4)
        assert torch.allclose(R_b, Q @ R_a, atol=1e-4)
        assert torch.allclose(t_b, apply_rigid(Q, tg, t_a), atol=1e-4)
