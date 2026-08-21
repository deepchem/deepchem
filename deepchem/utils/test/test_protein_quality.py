"""Tests for :mod:`deepchem.utils.protein_quality`.

Ground-truth values are computed by hand on a 3-residue toy structure
so the tests double as a specification.
"""

import math
import os
import tempfile

import numpy as np
import pytest

try:
    from Bio.PDB import PDBParser  # noqa: F401
    has_biopython = True
except ImportError:
    has_biopython = False

from deepchem.utils.protein_quality import (
    clash_score,
    kabsch_align,
    radius_of_gyration,
    sc_rmsd,
    sc_tm,
    tm_score,
)


# ----------------------------------------------------------------------
# radius_of_gyration
# ----------------------------------------------------------------------
class TestRadiusOfGyration:

    def test_known_value_unit_cube(self):
        # Two points at (±1, 0, 0): centroid = origin, R_g = 1.
        pts = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
        assert math.isclose(radius_of_gyration(pts), 1.0, rel_tol=1e-12)

    def test_three_residue_toy(self):
        # Equilateral triangle of side 1 in the xy-plane.
        pts = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, math.sqrt(3) / 2.0, 0.0],
        ])
        # Centroid = (0.5, √3/6, 0). Distances² to centroid:
        # corner1: 0.25 + 1/12 = 1/3; same for the other two.
        # R_g = √(1/3) ≈ 0.577350.
        assert math.isclose(radius_of_gyration(pts),
                            math.sqrt(1.0 / 3.0),
                            rel_tol=1e-12)

    def test_translation_invariance(self):
        rng = np.random.default_rng(0)
        pts = rng.normal(size=(50, 3))
        rg = radius_of_gyration(pts)
        rg_shift = radius_of_gyration(pts + np.array([10.0, -3.5, 7.2]))
        assert math.isclose(rg, rg_shift, rel_tol=1e-12)

    def test_rotation_invariance(self):
        rng = np.random.default_rng(1)
        pts = rng.normal(size=(50, 3))
        # Random rotation via QR.
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(q) < 0:
            q[:, 0] = -q[:, 0]
        assert math.isclose(radius_of_gyration(pts),
                            radius_of_gyration(pts @ q.T),
                            rel_tol=1e-12)

    def test_multi_atom_input_flattens(self):
        pts = np.zeros((4, 3, 3))
        pts[0, 0] = [1.0, 0.0, 0.0]
        pts[0, 1] = [-1.0, 0.0, 0.0]
        # All others zero. Centroid = (0,0,0)/12 = origin; R_g = √(2/12).
        expected = math.sqrt(2.0 / 12.0)
        assert math.isclose(radius_of_gyration(pts), expected,
                            rel_tol=1e-12)

    def test_validation(self):
        with pytest.raises(ValueError):
            radius_of_gyration(np.zeros((4, 2)))
        with pytest.raises(ValueError):
            radius_of_gyration(np.zeros((0, 3)))


# ----------------------------------------------------------------------
# clash_score
# ----------------------------------------------------------------------
class TestClashScore:

    def test_no_pairs(self):
        coords = np.zeros((1, 3))
        radii = np.array([1.0])
        assert clash_score(coords, radii) == 0.0

    def test_handcomputed_three_atoms(self):
        # Atoms at distances 1, 5, 4 with radii (1, 1, 1) and t=0.6 →
        # cutoff = 2 - 0.6 = 1.4 for every pair.
        # d(0,1)=1.0 < 1.4 → clash. d(0,2)=4.0, d(1,2)=5.0 → no clash.
        # Score = 1 / 3 pairs.
        coords = np.array([[0.0, 0.0, 0.0],
                           [1.0, 0.0, 0.0],
                           [4.0, 0.0, 0.0]])
        radii = np.ones(3)
        assert math.isclose(clash_score(coords, radii, threshold=0.6),
                            1.0 / 3.0, rel_tol=1e-12)

    def test_all_clashing(self):
        # Three coincident atoms always clash.
        coords = np.zeros((3, 3))
        radii = np.array([1.0, 1.0, 1.0])
        assert clash_score(coords, radii) == 1.0

    def test_none_clashing(self):
        # Atoms 100 Å apart: nothing can clash.
        coords = np.arange(15).reshape(5, 3).astype(np.float64) * 100.0
        radii = np.ones(5) * 1.7
        assert clash_score(coords, radii) == 0.0

    def test_threshold_relaxation_reduces_clashes(self):
        coords = np.array([[0.0, 0.0, 0.0],
                           [1.5, 0.0, 0.0]])
        radii = np.array([1.0, 1.0])
        strict = clash_score(coords, radii, threshold=0.0)  # cutoff 2.0
        lenient = clash_score(coords, radii, threshold=1.0)  # cutoff 1.0
        assert strict == 1.0
        assert lenient == 0.0

    def test_validation(self):
        with pytest.raises(ValueError):
            clash_score(np.zeros((4, 2)), np.ones(4))
        with pytest.raises(ValueError):
            clash_score(np.zeros((4, 3)), np.ones(3))


# ----------------------------------------------------------------------
# kabsch_align + tm_score
# ----------------------------------------------------------------------
class TestKabsch:

    def test_aligns_translated_target(self):
        rng = np.random.default_rng(2)
        target = rng.normal(size=(20, 3))
        mobile = target + np.array([5.0, -2.0, 1.5])
        aligned = kabsch_align(mobile, target)
        assert np.allclose(aligned, target, atol=1e-10)

    def test_aligns_rotated_target(self):
        rng = np.random.default_rng(3)
        target = rng.normal(size=(30, 3))
        # Build a proper rotation.
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(q) < 0:
            q[:, 0] = -q[:, 0]
        mobile = (target @ q.T) + np.array([10.0, 0.0, 0.0])
        aligned = kabsch_align(mobile, target)
        assert np.allclose(aligned, target, atol=1e-9)


class TestTMScore:

    def test_identity_is_one(self):
        rng = np.random.default_rng(4)
        coords = rng.normal(size=(25, 3))
        assert math.isclose(tm_score(coords, coords), 1.0, rel_tol=1e-12)

    def test_three_residue_toy(self):
        # Toy: target on x-axis, mobile = target + 1Å shift on x.
        # After Kabsch the shift is removed → TM = 1 because residues
        # superpose perfectly.
        target = np.array([[0.0, 0.0, 0.0],
                           [1.0, 0.0, 0.0],
                           [2.0, 0.0, 0.0]])
        mobile = target + np.array([1.0, 0.0, 0.0])
        assert math.isclose(tm_score(mobile, target), 1.0, rel_tol=1e-9)

    def test_random_alignment_low_score(self):
        rng = np.random.default_rng(5)
        target = rng.normal(size=(50, 3)) * 5.0
        mobile = rng.normal(size=(50, 3)) * 5.0
        score = tm_score(mobile, target)
        # Random pairs should not look like the same fold.
        assert 0.0 < score < 0.5

    def test_zhang_d0_three_residue(self):
        # For L < 17 the Zhang-Skolnick formula returns d0 = 0.5.
        # Per-residue di = 1 → contribution 1/(1 + 4) = 0.2; with three
        # residues TM = 3 * 0.2 / 3 = 0.2.
        target = np.zeros((3, 3))
        # Mobile must be incompatible with rigid alignment, otherwise
        # Kabsch would remove the difference. Use three orthogonal
        # offsets so that no rigid transform reduces the per-atom
        # distance below 1 Å.
        mobile = np.array([[1.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0],
                           [0.0, 0.0, 1.0]])
        score = tm_score(mobile, target)
        # All three distances become exactly 1 (centroid removed):
        # offsets minus mean offset → norms 1·√(2/3). With d0 = 0.5
        # and d² = 2/3, contribution 1/(1 + (2/3)/0.25) = 1/(1+8/3)
        # = 3/11. TM = 3 · 3/11 / 3 = 3/11 ≈ 0.2727.
        assert math.isclose(score, 3.0 / 11.0, rel_tol=1e-9)


# ----------------------------------------------------------------------
# sc_rmsd / sc_tm with a synthetic refolder.
# ----------------------------------------------------------------------
_TOY_PDB = (
    'ATOM      1  N   ALA A   1       0.000   0.000   0.000  '
    '1.00  0.00           N\n'
    'ATOM      2  CA  ALA A   1       1.000   0.000   0.000  '
    '1.00  0.00           C\n'
    'ATOM      3  C   ALA A   1       2.000   0.000   0.000  '
    '1.00  0.00           C\n'
    'ATOM      4  N   ALA A   2       3.000   0.000   0.000  '
    '1.00  0.00           N\n'
    'ATOM      5  CA  ALA A   2       4.000   0.000   0.000  '
    '1.00  0.00           C\n'
    'ATOM      6  C   ALA A   2       5.000   0.000   0.000  '
    '1.00  0.00           C\n'
    'ATOM      7  N   ALA A   3       6.000   0.000   0.000  '
    '1.00  0.00           N\n'
    'ATOM      8  CA  ALA A   3       7.000   0.000   0.000  '
    '1.00  0.00           C\n'
    'ATOM      9  C   ALA A   3       8.000   0.000   0.000  '
    '1.00  0.00           C\n'
    'END\n')


@pytest.fixture
def designed_pdb():
    fh = tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False)
    fh.write(_TOY_PDB)
    fh.close()
    yield fh.name
    os.unlink(fh.name)


@pytest.mark.skipif(not has_biopython, reason='BioPython required.')
class TestSelfConsistency:

    def test_sc_rmsd_zero_against_self(self, designed_pdb):
        # Refolder returns the input path unchanged → RMSD = 0.
        rmsd = sc_rmsd(designed_pdb, refolded_pdb='',
                       refolder=lambda p: p)
        assert rmsd < 1e-10

    def test_sc_tm_one_against_self(self, designed_pdb):
        score = sc_tm(designed_pdb, refolded_pdb='',
                      refolder=lambda p: p)
        assert math.isclose(score, 1.0, rel_tol=1e-9)

    def test_refolder_invoked_only_when_needed(self, designed_pdb):
        called = {'n': 0}

        def fake_refolder(path):
            called['n'] += 1
            return path

        sc_rmsd(designed_pdb, refolded_pdb='', refolder=fake_refolder)
        sc_rmsd(designed_pdb, refolded_pdb=designed_pdb,
                refolder=fake_refolder)
        assert called['n'] == 1

    def test_missing_refolder_raises(self, designed_pdb):
        with pytest.raises(ValueError):
            sc_rmsd(designed_pdb, refolded_pdb='')

    def test_perturbed_refolder_positive_rmsd(self, designed_pdb,
                                              tmp_path):
        # Build a second PDB by shifting the original by 1Å on y. After
        # Kabsch alignment the shift is removed → RMSD must be < 1e-9.
        # Use a non-rigid perturbation (per-residue shifts) instead.
        path2 = tmp_path / 'perturbed.pdb'
        with open(designed_pdb) as fh:
            text = fh.read()
        # Replace CA y-coords by injecting per-residue jitter.
        lines = text.splitlines()
        new_lines = []
        offsets = {1: 0.3, 2: -0.4, 3: 0.5}
        for line in lines:
            if line.startswith('ATOM') and ' CA ' in line:
                resnum = int(line[22:26])
                # PDB y coordinate columns 39:46 (8-wide right-aligned).
                y = offsets[resnum]
                line = line[:38] + f'{y:8.3f}' + line[46:]
            new_lines.append(line)
        path2.write_text('\n'.join(new_lines) + '\n')
        rmsd = sc_rmsd(designed_pdb, refolded_pdb=str(path2))
        assert rmsd > 1e-3

    def test_length_mismatch_raises(self, designed_pdb, tmp_path):
        short = tmp_path / 'short.pdb'
        with open(designed_pdb) as fh:
            text = fh.read()
        # Keep only residues 1 and 2.
        kept = []
        for line in text.splitlines():
            if line.startswith('ATOM'):
                resnum = int(line[22:26])
                if resnum > 2:
                    continue
            kept.append(line)
        short.write_text('\n'.join(kept) + '\n')
        with pytest.raises(ValueError):
            sc_rmsd(designed_pdb, refolded_pdb=str(short))
