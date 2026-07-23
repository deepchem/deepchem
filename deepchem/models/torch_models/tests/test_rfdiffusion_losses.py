"""Tests for RFDiffusion losses (FAPE, χ-angle, ligand contact)."""

import math

import pytest
import torch

from deepchem.models.torch_models.rfdiffusion_losses import (
    all_atom_l2_loss,
    backbone_fape_loss,
    chi_angle_loss,
    ligand_contact_loss,
)


def _random_rotation(seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(3, 3, generator=g)
    Q, _ = torch.linalg.qr(A)
    if torch.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


class TestBackboneFAPE:

    def _make_frames(self, batch, length, seed=0):
        torch.manual_seed(seed)
        rots = torch.stack(
            [_random_rotation(seed + i) for i in range(length)])
        rots = rots.unsqueeze(0).expand(batch, -1, -1, -1).contiguous()
        trans = torch.randn(batch, length, 3)
        return rots, trans

    def test_zero_when_pred_equals_true(self):
        rots, trans = self._make_frames(2, 6)
        atoms = torch.randn(2, 6, 3, 3)
        loss = backbone_fape_loss(rots, trans, rots, trans, atoms, atoms)
        # ε = 1e-4 floor inside sqrt yields ~1e-3 when difference is zero.
        assert loss.item() < 5e-3

    def test_positive_with_random_diff(self):
        rots, trans = self._make_frames(2, 6)
        rots2, trans2 = self._make_frames(2, 6, seed=99)
        atoms = torch.randn(2, 6, 3, 3)
        atoms2 = torch.randn(2, 6, 3, 3)
        loss = backbone_fape_loss(rots, trans, rots2, trans2, atoms, atoms2)
        assert loss.item() > 0

    def test_clamp_caps_distance(self):
        rots, trans = self._make_frames(1, 4)
        atoms_true = torch.zeros(1, 4, 3, 3)
        atoms_pred = atoms_true + 1000.0  # far away
        loss_unclamped_proxy = backbone_fape_loss(
            rots, trans, rots, trans, atoms_pred, atoms_true,
            clamp_distance=10.0, length_scale=10.0)
        # All distances clamped to 10 → loss ≈ 10/10 = 1.
        assert 0.95 < loss_unclamped_proxy.item() <= 1.05

    def test_mask_zeros_invalid(self):
        rots, trans = self._make_frames(1, 4)
        atoms = torch.randn(1, 4, 3, 3, requires_grad=True)
        mask = torch.tensor([[True, True, False, False]])
        loss = backbone_fape_loss(
            rots, trans, rots, trans + 1.0,
            atoms, atoms.detach(), mask=mask)
        loss.backward()
        # Gradient on masked atoms must be exactly zero.
        assert torch.all(atoms.grad[:, 2:] == 0)
        assert torch.any(atoms.grad[:, :2] != 0)

    def test_shape_validation(self):
        with pytest.raises(ValueError):
            backbone_fape_loss(torch.zeros(1, 4, 3, 3),
                               torch.zeros(1, 4, 3),
                               torch.zeros(1, 5, 3, 3),
                               torch.zeros(1, 5, 3),
                               torch.zeros(1, 4, 3, 3),
                               torch.zeros(1, 4, 3, 3))


class TestChiAngleLoss:

    def test_zero_when_equal(self):
        a = torch.randn(2, 5, 4)
        loss = chi_angle_loss(a, a)
        assert loss.item() < 1e-7

    def test_periodicity(self):
        a = torch.zeros(1, 1, 4)
        b = torch.full((1, 1, 4), 2 * math.pi)
        loss = chi_angle_loss(a, b)
        # 1 − cos(2π) = 0.
        assert loss.item() < 1e-6

    def test_max_at_pi(self):
        a = torch.zeros(1, 1, 4)
        b = torch.full((1, 1, 4), math.pi)
        loss = chi_angle_loss(a, b)
        # 1 − cos(π) = 2 exactly.
        assert abs(loss.item() - 2.0) < 1e-6

    def test_mask_excludes_entries(self):
        a = torch.zeros(1, 1, 4)
        b = torch.tensor([[[math.pi, 0.0, 0.0, 0.0]]])
        m = torch.tensor([[[False, True, True, True]]])
        loss = chi_angle_loss(a, b, m)
        # All unmasked χ are equal → loss zero.
        assert loss.item() < 1e-6


class TestLigandContact:

    def test_zero_when_inside_contact(self):
        ca = torch.zeros(3, 3)
        lig = torch.zeros(5, 3)
        mask = torch.ones(3, dtype=torch.bool)
        loss = ligand_contact_loss(ca, lig, mask, contact_distance=6.0)
        assert loss.item() < 1e-6

    def test_positive_when_far(self):
        ca = torch.tensor([[100.0, 0.0, 0.0]])
        lig = torch.tensor([[0.0, 0.0, 0.0]])
        mask = torch.ones(1, dtype=torch.bool)
        loss = ligand_contact_loss(ca, lig, mask, contact_distance=6.0)
        assert loss.item() > 90.0

    def test_masked_residues_contribute_nothing(self):
        ca = torch.tensor([[100.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        lig = torch.tensor([[0.0, 0.0, 0.0]])
        mask = torch.tensor([False, True])
        loss = ligand_contact_loss(ca, lig, mask)
        assert loss.item() < 1e-6


class TestAllAtomL2:

    def test_masked_atoms_zero_gradient(self):
        pred = torch.randn(4, 14, 3, requires_grad=True)
        true = torch.randn(4, 14, 3)
        mask = torch.zeros(4, 14, dtype=torch.bool)
        mask[:, :5] = True  # only first five atoms valid.
        loss = all_atom_l2_loss(pred, true, mask)
        loss.backward()
        # Strict: masked-atom gradient is *exactly* zero.
        assert torch.all(pred.grad[:, 5:] == 0)
        # Unmasked atoms should have gradient.
        assert torch.any(pred.grad[:, :5] != 0)

    def test_zero_loss_when_pred_equals_true(self):
        x = torch.randn(2, 14, 3)
        mask = torch.ones(2, 14, dtype=torch.bool)
        loss = all_atom_l2_loss(x, x, mask)
        assert loss.item() < 1e-9
