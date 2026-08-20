"""Tests for RFDiffusion All-Atom loss functions."""

import math

import pytest

try:
    import torch
    from deepchem.models.torch_models.rfdiffusion_frames import (
        build_backbone_frames, make_identity_rigid)
    from deepchem.models.torch_models.rfdiffusion_losses import (
        DEFAULT_VDW_RADII, chi_angle_loss, dihedral_angle,
        frame_aligned_point_error, ligand_clash_loss, masked_all_atom_l2_loss,
        vdw_radii_from_symbols)
    has_torch = True
except ImportError:
    has_torch = False

requires_torch = pytest.mark.skipif(not has_torch,
                                    reason='PyTorch not installed')


@pytest.mark.torch
@requires_torch
class TestFrameAlignedPointError:

    def test_zero_for_perfect_prediction(self):
        R, t = make_identity_rigid((3,))
        points = torch.randn(7, 3)
        loss = frame_aligned_point_error(R, t, R, t, points, points)
        assert round(float(loss), 5) == 0.0

    def test_positive_for_wrong_prediction(self):
        R, t = make_identity_rigid((2,))
        true_points = torch.zeros(4, 3)
        pred_points = torch.ones(4, 3)
        loss = frame_aligned_point_error(R, t, R, t, pred_points, true_points)
        assert float(loss) > 0.0

    def test_invariant_to_global_rotation(self):
        torch.manual_seed(0)
        backbone = torch.randn(1, 3, 3, 3)
        R, t = build_backbone_frames(backbone)
        points = torch.randn(1, 5, 3)
        loss_before = frame_aligned_point_error(R, t, R, t, points, points)

        # Apply the same global rigid motion to everything.
        rot = build_backbone_frames(torch.randn(1, 1, 3, 3))[0][:,
                                                                0]  # (1, 3, 3)
        shift = torch.randn(1, 1, 3)
        R2 = torch.matmul(rot.unsqueeze(1), R)
        t2 = torch.matmul(t.unsqueeze(-2), rot.transpose(
            -1, -2)).squeeze(-2) + shift
        points2 = torch.matmul(points, rot.transpose(-1, -2)) + shift
        loss_after = frame_aligned_point_error(R2, t2, R2, t2, points2, points2)
        assert round(float(loss_before), 4) == round(float(loss_after), 4)

    def test_clamping_bounds_large_errors(self):
        R, t = make_identity_rigid((1,))
        true_points = torch.zeros(1, 3)
        pred_points = torch.tensor([[1000.0, 0.0, 0.0]])
        clamped = frame_aligned_point_error(R,
                                            t,
                                            R,
                                            t,
                                            pred_points,
                                            true_points,
                                            clamp_distance=5.0)
        unclamped = frame_aligned_point_error(R,
                                              t,
                                              R,
                                              t,
                                              pred_points,
                                              true_points,
                                              clamp_distance=None)
        assert float(clamped) < float(unclamped)

    def test_masked_positions_ignored(self):
        R, t = make_identity_rigid((1,))
        true_points = torch.zeros(3, 3)
        pred_points = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                                    [50.0, 0.0, 0.0]])
        mask = torch.tensor([1.0, 1.0, 0.0])
        loss = frame_aligned_point_error(R,
                                         t,
                                         R,
                                         t,
                                         pred_points,
                                         true_points,
                                         position_mask=mask)
        assert round(float(loss), 5) == 0.0


@pytest.mark.torch
@requires_torch
class TestDihedralAngle:

    def test_cis_configuration_is_zero(self):
        p0 = torch.tensor([1.0, 0.0, 0.0])
        p1 = torch.tensor([0.0, 0.0, 0.0])
        p2 = torch.tensor([0.0, 1.0, 0.0])
        p3 = torch.tensor([1.0, 1.0, 0.0])
        angle = dihedral_angle(p0, p1, p2, p3)
        assert round(float(angle), 4) == 0.0

    def test_trans_configuration_is_pi(self):
        p0 = torch.tensor([1.0, 0.0, 0.0])
        p1 = torch.tensor([0.0, 0.0, 0.0])
        p2 = torch.tensor([0.0, 1.0, 0.0])
        p3 = torch.tensor([-1.0, 1.0, 0.0])
        angle = dihedral_angle(p0, p1, p2, p3)
        assert round(abs(float(angle)), 4) == round(math.pi, 4)

    def test_batched_shape(self):
        p = torch.randn(4, 3)
        angle = dihedral_angle(p[0], p[1], p[2], p[3])
        assert angle.shape == ()
        p_batched = torch.randn(5, 4, 3)
        angle_batched = dihedral_angle(p_batched[:, 0], p_batched[:, 1],
                                       p_batched[:, 2], p_batched[:, 3])
        assert angle_batched.shape == (5,)


@pytest.mark.torch
@requires_torch
class TestChiAngleLoss:

    def test_zero_for_identical_dihedral(self):
        atoms = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                              [1.0, 1.0, 0.0]])
        loss = chi_angle_loss(atoms.unsqueeze(0), atoms.unsqueeze(0))
        assert round(float(loss), 6) == 0.0

    def test_max_loss_for_opposite_dihedral(self):
        cis = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                            [1.0, 1.0, 0.0]])
        trans = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                              [-1.0, 1.0, 0.0]])
        loss = chi_angle_loss(cis.unsqueeze(0), trans.unsqueeze(0))
        # 1 - cos(pi) == 2, the maximum of this loss.
        assert round(float(loss), 4) == 2.0

    def test_mask_zeroes_out_excluded_entries(self):
        cis = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                            [1.0, 1.0, 0.0]])
        trans = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                              [-1.0, 1.0, 0.0]])
        pred = torch.stack([cis, cis])
        true = torch.stack([cis, trans])
        mask = torch.tensor([1.0, 0.0])
        loss = chi_angle_loss(pred, true, mask=mask)
        assert round(float(loss), 6) == 0.0


@pytest.mark.torch
@requires_torch
class TestLigandClashLoss:

    def test_zero_when_far_apart(self):
        coords = torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        radii = torch.tensor([1.7, 1.7])
        loss = ligand_clash_loss(coords, radii)
        assert round(float(loss), 6) == 0.0

    def test_positive_when_overlapping(self):
        coords = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
        radii = torch.tensor([1.7, 1.7])
        loss = ligand_clash_loss(coords, radii)
        assert float(loss) > 0.0

    def test_self_pairs_excluded(self):
        # A single atom can never clash with itself.
        coords = torch.zeros(1, 3)
        radii = torch.tensor([1.7])
        loss = ligand_clash_loss(coords, radii)
        assert round(float(loss), 6) == 0.0

    def test_masked_atoms_excluded(self):
        coords = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0],
                               [20.0, 0.0, 0.0]])
        radii = torch.tensor([1.7, 1.7, 1.7])
        mask = torch.tensor([1.0, 0.0, 1.0])  # clashing pair masked out
        loss = ligand_clash_loss(coords, radii, mask=mask)
        assert round(float(loss), 6) == 0.0

    def test_tolerance_allows_small_overlap(self):
        coords = torch.tensor([[0.0, 0.0, 0.0], [3.39, 0.0, 0.0]])
        radii = torch.tensor([1.7, 1.7])  # sum = 3.4
        loose = ligand_clash_loss(coords, radii, tolerance=0.4)
        tight = ligand_clash_loss(coords, radii, tolerance=0.0)
        assert float(loose) == 0.0
        assert float(tight) > 0.0


@pytest.mark.torch
@requires_torch
class TestMaskedAllAtomL2Loss:

    def test_zero_for_perfect_prediction(self):
        coords = torch.randn(6, 3)
        loss = masked_all_atom_l2_loss(coords, coords)
        assert round(float(loss), 6) == 0.0

    def test_known_answer(self):
        pred = torch.zeros(4, 3)
        true = torch.ones(4, 3)
        mask = torch.tensor([1.0, 1.0, 0.0, 0.0])
        # each unmasked atom contributes ||[-1,-1,-1]||^2 = 3
        loss = masked_all_atom_l2_loss(pred, true, mask)
        assert float(loss) == 3.0

    def test_unmasked_matches_plain_mean(self):
        pred = torch.randn(5, 3)
        true = torch.randn(5, 3)
        masked = masked_all_atom_l2_loss(pred, true, torch.ones(5)).item()
        unmasked = masked_all_atom_l2_loss(pred, true).item()
        assert round(masked, 5) == round(unmasked, 5)


@pytest.mark.torch
@requires_torch
class TestVdwRadiiFromSymbols:

    def test_known_elements(self):
        radii = vdw_radii_from_symbols(['C', 'O', 'N'])
        assert round(radii[0].item(), 2) == DEFAULT_VDW_RADII['C']
        assert round(radii[1].item(), 2) == DEFAULT_VDW_RADII['O']
        assert round(radii[2].item(), 2) == DEFAULT_VDW_RADII['N']

    def test_unknown_element_falls_back_to_carbon(self):
        radii = vdw_radii_from_symbols(['Xx'])
        assert round(radii[0].item(), 2) == DEFAULT_VDW_RADII['C']

    def test_empty_list(self):
        radii = vdw_radii_from_symbols([])
        assert radii.shape == (0,)
