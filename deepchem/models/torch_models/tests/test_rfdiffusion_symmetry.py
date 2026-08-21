"""Tests for the point-group symmetry helpers."""

import math

import pytest
import torch

from deepchem.models.torch_models.rfdiffusion_symmetry import (
    cyclic_group,
    dihedral_group,
    icosahedral_group,
    octahedral_group,
    symmetrise_coords,
    symmetrise_frames,
    tetrahedral_group,
)


def _is_rotation(R, tol=1e-6):
    eye = torch.eye(3, dtype=R.dtype)
    err = (R @ R.transpose(-1, -2) - eye).abs().max()
    det = torch.det(R)
    return float(err) < tol and float((det - 1).abs().max()) < tol


class TestGroupConstruction:

    @pytest.mark.parametrize('n', [1, 2, 3, 4, 5, 6, 12])
    def test_cyclic_order(self, n):
        g = cyclic_group(n)
        assert g.shape == (n, 3, 3)
        for R in g:
            assert _is_rotation(R)

    @pytest.mark.parametrize('n', [2, 3, 4, 5])
    def test_dihedral_order(self, n):
        g = dihedral_group(n)
        assert g.shape == (2 * n, 3, 3)

    def test_tetrahedral(self):
        g = tetrahedral_group()
        assert g.shape == (12, 3, 3)

    def test_octahedral(self):
        g = octahedral_group()
        assert g.shape == (24, 3, 3)

    def test_icosahedral(self):
        g = icosahedral_group()
        assert g.shape == (60, 3, 3)

    def test_group_contains_identity(self):
        for g in [cyclic_group(4), dihedral_group(3), tetrahedral_group(),
                  octahedral_group(), icosahedral_group()]:
            eye = torch.eye(3, dtype=g.dtype)
            diffs = (g - eye).reshape(g.shape[0], -1).abs().sum(-1)
            assert diffs.min() < 1e-8

    def test_validation(self):
        with pytest.raises(ValueError):
            cyclic_group(0)
        with pytest.raises(ValueError):
            dihedral_group(1)


class TestSymmetrise:

    def test_coords_match_under_group_action(self):
        n = 4
        g = cyclic_group(n)
        unit = 5
        coords = torch.randn(g.shape[0] * unit, 3, dtype=torch.float64) * 2.0
        sym = symmetrise_coords(coords, g)
        # The symmetric image must satisfy x_g = R_g · x_0.
        reshaped = sym.view(n, unit, 3)
        for k in range(n):
            expected = torch.einsum('ij,uj->ui', g[k], reshaped[0])
            err = (reshaped[k] - expected).abs().max()
            assert err < 1e-8

    def test_rmsd_between_units_below_tolerance(self):
        """RMSD between copies after applying inverse group action
        must be < 1e-5 Å."""
        n = 6
        g = cyclic_group(n)
        unit = 7
        torch.manual_seed(0)
        coords = torch.randn(n * unit, 3, dtype=torch.float64)
        sym = symmetrise_coords(coords, g).view(n, unit, 3)
        ref = sym[0]
        for k in range(1, n):
            back = torch.einsum('ij,uj->ui', g[k].transpose(-1, -2), sym[k])
            rmsd = (back - ref).pow(2).mean().sqrt()
            assert rmsd < 1e-5

    def test_frames_are_proper_rotations(self):
        g = octahedral_group()
        unit = 3
        torch.manual_seed(1)
        rots_raw = torch.randn(g.shape[0] * unit, 3, 3, dtype=torch.float64)
        # Orthogonalise random matrices.
        rots_raw, _ = torch.linalg.qr(rots_raw)
        trans = torch.randn(g.shape[0] * unit, 3, dtype=torch.float64)
        r_sym, _ = symmetrise_frames(rots_raw, trans, g)
        for R in r_sym:
            assert _is_rotation(R, tol=1e-6)

    def test_frames_equivariant(self):
        g = cyclic_group(3)
        unit = 4
        torch.manual_seed(2)
        rots = torch.stack([
            torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))[0]
            for _ in range(g.shape[0] * unit)])
        # Ensure proper rotations.
        rots = rots * torch.det(rots).sign().view(-1, 1, 1)
        trans = torch.randn(g.shape[0] * unit, 3, dtype=torch.float64)
        r_sym, t_sym = symmetrise_frames(rots, trans, g)
        r_sym = r_sym.view(3, unit, 3, 3)
        t_sym = t_sym.view(3, unit, 3)
        for k in range(1, 3):
            # R_k = R_g_k · R_0
            expected_r = torch.einsum('ij,ujk->uik', g[k], r_sym[0])
            expected_t = torch.einsum('ij,uj->ui', g[k], t_sym[0])
            assert (r_sym[k] - expected_r).abs().max() < 1e-6
            assert (t_sym[k] - expected_t).abs().max() < 1e-6

    def test_validation_rejects_bad_length(self):
        with pytest.raises(ValueError):
            symmetrise_coords(torch.zeros(5, 3), cyclic_group(2))
        with pytest.raises(ValueError):
            symmetrise_frames(torch.zeros(5, 3, 3), torch.zeros(5, 3),
                              cyclic_group(2))
