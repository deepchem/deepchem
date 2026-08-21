"""Tests for RFDiffusion All-Atom point-group symmetry utilities."""

import pytest

try:
    import torch
    from deepchem.models.torch_models.rfdiffusion_frames import (
        make_identity_rigid)
    from deepchem.models.torch_models.rfdiffusion_symmetry import (
        cyclic_group, dihedral_group, icosahedral_group, octahedral_group,
        symmetrize_coords, symmetrize_frames, tetrahedral_group)
    has_torch = True
except ImportError:
    has_torch = False

requires_torch = pytest.mark.skipif(not has_torch,
                                    reason='PyTorch not installed')


def _is_valid_rotation_batch(group, atol=1e-4):
    det = torch.linalg.det(group)
    orth = torch.matmul(group, group.transpose(-1, -2))
    eye = torch.eye(3).expand_as(orth)
    return torch.allclose(det, torch.ones_like(det), atol=atol) and \
        torch.allclose(orth, eye, atol=atol)


@pytest.mark.torch
@requires_torch
class TestCyclicGroup:

    def test_order(self):
        assert cyclic_group(5).shape == (5, 3, 3)

    def test_identity_is_first_element(self):
        group = cyclic_group(4)
        assert torch.allclose(group[0], torch.eye(3), atol=1e-6)

    def test_all_valid_rotations(self):
        assert _is_valid_rotation_batch(cyclic_group(7))

    def test_n_applications_return_to_identity(self):
        n = 6
        group = cyclic_group(n)
        g = group[1]
        power = torch.eye(3)
        for _ in range(n):
            power = torch.matmul(g, power)
        assert torch.allclose(power, torch.eye(3), atol=1e-4)

    def test_invalid_n_raises(self):
        with pytest.raises(ValueError):
            cyclic_group(0)


@pytest.mark.torch
@requires_torch
class TestDihedralGroup:

    def test_order_is_double_cyclic(self):
        assert dihedral_group(5).shape == (10, 3, 3)

    def test_all_valid_rotations(self):
        assert _is_valid_rotation_batch(dihedral_group(3))

    def test_invalid_n_raises(self):
        with pytest.raises(ValueError):
            dihedral_group(-1)


@pytest.mark.torch
@requires_torch
class TestPolyhedralGroups:

    def test_tetrahedral_order(self):
        group = tetrahedral_group()
        assert group.shape == (12, 3, 3)
        assert _is_valid_rotation_batch(group)

    def test_octahedral_order(self):
        group = octahedral_group()
        assert group.shape == (24, 3, 3)
        assert _is_valid_rotation_batch(group)

    def test_icosahedral_order(self):
        group = icosahedral_group()
        assert group.shape == (60, 3, 3)
        assert _is_valid_rotation_batch(group)

    def test_groups_have_no_duplicate_elements(self):
        for group in (tetrahedral_group(), octahedral_group()):
            n = group.shape[0]
            for i in range(n):
                for j in range(i + 1, n):
                    assert not torch.allclose(group[i], group[j], atol=1e-4), (
                        f'duplicate elements at {i}, {j}')


@pytest.mark.torch
@requires_torch
class TestSymmetrizeCoords:

    def test_output_shape(self):
        group = cyclic_group(3)
        coords = torch.randn(3, 5, 3)
        out = symmetrize_coords(coords, group)
        assert out.shape == (3, 5, 3)

    def test_exact_symmetric_input_is_fixed_point(self):
        group = dihedral_group(4)
        asu = torch.randn(6, 3)
        exact = torch.matmul(asu, group.transpose(-1, -2))
        out = symmetrize_coords(exact, group)
        assert torch.allclose(out, exact, atol=1e-4)

    def test_mismatched_group_size_raises(self):
        group = cyclic_group(3)
        coords = torch.randn(4, 5, 3)  # 4 copies but group has 3 elements
        with pytest.raises(ValueError):
            symmetrize_coords(coords, group)

    def test_explicit_center(self):
        group = cyclic_group(2)
        coords = torch.zeros(2, 3, 3)
        out = symmetrize_coords(coords, group, center=torch.zeros(3))
        assert torch.allclose(out, torch.zeros(2, 3, 3), atol=1e-5)


@pytest.mark.torch
@requires_torch
class TestSymmetrizeFrames:

    def test_output_shapes(self):
        group = cyclic_group(4)
        R, t = make_identity_rigid((4, 6))
        sym_R, sym_t = symmetrize_frames(R, t, group)
        assert sym_R.shape == (4, 6, 3, 3)
        assert sym_t.shape == (4, 6, 3)

    def test_identity_frames_stay_identity_at_origin(self):
        group = cyclic_group(3)
        R, t = make_identity_rigid((3, 2))
        sym_R, sym_t = symmetrize_frames(R, t, group, center=torch.zeros(3))
        assert torch.allclose(sym_t, torch.zeros(3, 2, 3), atol=1e-4)

    def test_symmetrized_rotations_are_valid(self):
        torch.manual_seed(0)
        group = tetrahedral_group()
        R = torch.stack(
            [torch.linalg.qr(torch.randn(3, 3))[0] for _ in range(12)])
        # make them proper rotations (det=1)
        det = torch.linalg.det(R)
        R[det < 0, :, 0] *= -1
        R = R.unsqueeze(1)  # (12, 1, 3, 3)
        t = torch.randn(12, 1, 3)
        sym_R, _ = symmetrize_frames(R, t, group)
        assert _is_valid_rotation_batch(sym_R.reshape(-1, 3, 3), atol=1e-3)

    def test_mismatched_group_size_raises(self):
        group = cyclic_group(3)
        R, t = make_identity_rigid((4, 2))
        with pytest.raises(ValueError):
            symmetrize_frames(R, t, group)
