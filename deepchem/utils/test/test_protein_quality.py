"""Tests for protein structure quality metrics."""

import numpy as np
import pytest

from deepchem.utils.protein_quality import (backbone_bond_validity, clash_score,
                                            kabsch_align, radius_of_gyration,
                                            rmsd, sc_rmsd, tm_score)


class TestRadiusOfGyration:

    def test_known_answer_two_points(self):
        coords = np.array([[-2.0, 0, 0], [2.0, 0, 0]])
        assert radius_of_gyration(coords) == pytest.approx(2.0)

    def test_single_point_is_zero(self):
        coords = np.zeros((1, 3))
        assert radius_of_gyration(coords) == pytest.approx(0.0)

    def test_weighted_pulls_toward_heavier_point(self):
        coords = np.array([[-1.0, 0, 0], [1.0, 0, 0]])
        unweighted = radius_of_gyration(coords)
        weighted = radius_of_gyration(coords, masses=np.array([10.0, 1.0]))
        assert weighted < unweighted

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            radius_of_gyration(np.zeros((0, 3)))

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError):
            radius_of_gyration(np.zeros((5, 2)))

    def test_mismatched_masses_raises(self):
        with pytest.raises(ValueError):
            radius_of_gyration(np.zeros((5, 3)), masses=np.zeros(3))


class TestClashScore:

    def test_no_clash_far_apart(self):
        coords = np.array([[0.0, 0, 0], [20.0, 0, 0]])
        radii = np.array([1.7, 1.7])
        assert clash_score(coords, radii) == 0.0

    def test_all_pairs_clash(self):
        coords = np.array([[0.0, 0, 0], [0.1, 0, 0], [0.2, 0, 0]])
        radii = np.array([1.7, 1.7, 1.7])
        assert clash_score(coords, radii) == 1.0

    def test_known_fraction(self):
        # 3 atoms: (0,1) clash, (0,2) and (1,2) do not.
        coords = np.array([[0.0, 0, 0], [0.5, 0, 0], [20.0, 0, 0]])
        radii = np.array([1.7, 1.7, 1.7])
        assert clash_score(coords, radii) == pytest.approx(1.0 / 3.0)

    def test_too_few_atoms_raises(self):
        with pytest.raises(ValueError):
            clash_score(np.zeros((1, 3)), np.array([1.7]))

    def test_mismatched_radii_raises(self):
        with pytest.raises(ValueError):
            clash_score(np.zeros((3, 3)), np.array([1.7, 1.7]))


class TestKabschAlign:

    def test_pure_translation(self):
        target = np.array([[0.0, 0, 0], [1.0, 0, 0], [0.0, 1.0, 0]])
        mobile = target + np.array([5.0, -2.0, 1.0])
        rot, _, aligned = kabsch_align(mobile, target)
        assert np.allclose(rot, np.eye(3), atol=1e-6)
        assert np.allclose(aligned, target, atol=1e-6)

    def test_rotation_and_translation_recovered(self):
        rng = np.random.RandomState(0)
        target = rng.randn(10, 3)
        theta = 0.7
        true_rot = np.array([[np.cos(theta), -np.sin(theta), 0],
                             [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
        mobile = target @ true_rot.T + np.array([1.0, 2.0, 3.0])
        _, _, aligned = kabsch_align(mobile, target)
        assert np.allclose(aligned, target, atol=1e-6)

    def test_result_is_proper_rotation(self):
        rng = np.random.RandomState(1)
        mobile = rng.randn(8, 3)
        target = rng.randn(8, 3)
        rot, _, _ = kabsch_align(mobile, target)
        assert np.linalg.det(rot) == pytest.approx(1.0, abs=1e-6)
        assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-6)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            kabsch_align(np.zeros((3, 3)), np.zeros((4, 3)))

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            kabsch_align(np.zeros((0, 3)), np.zeros((0, 3)))


class TestRmsd:

    def test_identical_is_zero(self):
        coords = np.random.RandomState(0).randn(5, 3)
        assert rmsd(coords, coords) == pytest.approx(0.0)

    def test_known_answer(self):
        a = np.zeros((2, 3))
        b = np.array([[3.0, 4.0, 0.0], [3.0, 4.0, 0.0]])
        assert rmsd(a, b) == pytest.approx(5.0)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            rmsd(np.zeros((3, 3)), np.zeros((4, 3)))


class TestScRmsd:

    def test_zero_for_rigidly_related_structures(self):
        rng = np.random.RandomState(2)
        reference = rng.randn(12, 3)
        theta = 1.1
        rot = np.array([[np.cos(theta), -np.sin(theta), 0],
                        [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
        designed = reference @ rot.T + np.array([2.0, -1.0, 0.5])
        assert sc_rmsd(designed, reference) == pytest.approx(0.0, abs=1e-6)

    def test_positive_for_different_structures(self):
        rng = np.random.RandomState(3)
        reference = rng.randn(10, 3)
        designed = reference + rng.randn(10, 3) * 2.0
        assert sc_rmsd(designed, reference) > 0.0


class TestTmScore:

    def test_perfect_match_is_one(self):
        coords = np.random.RandomState(4).randn(40, 3) * 3
        assert tm_score(coords, coords) == pytest.approx(1.0, abs=1e-6)

    def test_rigidly_related_is_one(self):
        rng = np.random.RandomState(5)
        reference = rng.randn(30, 3) * 3
        theta = 0.4
        rot = np.array([[np.cos(theta), -np.sin(theta), 0],
                        [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
        designed = reference @ rot.T + np.array([1.0, 1.0, 1.0])
        assert tm_score(designed, reference) == pytest.approx(1.0, abs=1e-5)

    def test_lower_for_dissimilar_structures(self):
        rng = np.random.RandomState(6)
        reference = rng.randn(30, 3) * 3
        shuffled = reference.copy()
        rng.shuffle(shuffled)
        assert tm_score(shuffled, reference) < 1.0

    def test_score_in_valid_range(self):
        rng = np.random.RandomState(7)
        a = rng.randn(20, 3) * 5
        b = rng.randn(20, 3) * 5
        score = tm_score(a, b)
        assert 0.0 < score <= 1.0

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            tm_score(np.zeros((3, 3)), np.zeros((4, 3)))


class TestBackboneBondValidity:

    def test_all_valid(self):
        ca = np.array([[i * 3.8, 0.0, 0.0] for i in range(5)])
        result = backbone_bond_validity(ca)
        assert result['num_bonds'] == 4
        assert result['num_valid'] == 4
        assert result['fraction_valid'] == pytest.approx(1.0)
        assert result['invalid_indices'].tolist() == []

    def test_detects_broken_bond(self):
        ca = np.array([[0.0, 0, 0], [3.8, 0, 0], [3.8 + 30, 0, 0]])
        result = backbone_bond_validity(ca)
        assert result['num_bonds'] == 2
        assert result['num_valid'] == 1
        assert result['invalid_indices'].tolist() == [1]

    def test_custom_tolerance(self):
        ca = np.array([[0.0, 0, 0], [4.0, 0, 0]])  # 0.2 off from 3.8
        loose = backbone_bond_validity(ca, tolerance=0.3)
        tight = backbone_bond_validity(ca, tolerance=0.1)
        assert loose['num_valid'] == 1
        assert tight['num_valid'] == 0

    def test_too_short_raises(self):
        with pytest.raises(ValueError):
            backbone_bond_validity(np.zeros((1, 3)))
