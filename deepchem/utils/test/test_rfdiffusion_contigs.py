"""Tests for contig-map parsing and motif scaffolding."""

import numpy as np
import pytest

from deepchem.utils.rfdiffusion_contigs import (
    ContigMap,
    LinkerSegment,
    MotifSegment,
    build_motif_mask,
    fixed_layout,
    freeze_motif_coords,
    parse_contig_string,
)


class TestParser:

    def test_simple(self):
        cm = parse_contig_string('5-10/A12-30/5-10')
        assert isinstance(cm, ContigMap)
        assert len(cm.segments) == 3
        assert isinstance(cm.segments[0], LinkerSegment)
        assert isinstance(cm.segments[1], MotifSegment)
        lo, hi = cm.total_length_range()
        assert lo == 5 + 19 + 5
        assert hi == 10 + 19 + 10

    def test_fixed_linker_token(self):
        cm = parse_contig_string('5/A1-3/7')
        lo, hi = cm.total_length_range()
        assert lo == hi == 5 + 3 + 7

    def test_whitespace_separator(self):
        cm = parse_contig_string('5-5 A1-5 5-5')
        assert len(cm.segments) == 3
        assert cm.total_length_range() == (15, 15)

    def test_rejects_empty(self):
        with pytest.raises(ValueError):
            parse_contig_string('')
        with pytest.raises(ValueError):
            parse_contig_string('   ')

    def test_rejects_garbage_token(self):
        with pytest.raises(ValueError):
            parse_contig_string('5-10/!!!')

    def test_motif_validation(self):
        with pytest.raises(ValueError):
            MotifSegment(chain='AB', start=1, end=5)
        with pytest.raises(ValueError):
            MotifSegment(chain='A', start=5, end=3)

    def test_linker_validation(self):
        with pytest.raises(ValueError):
            LinkerSegment(lo=5, hi=3)


class TestRealisation:

    def test_realise_within_range(self):
        cm = parse_contig_string('5-10/A1-5/5-10')
        rng = np.random.default_rng(0)
        rc = cm.realise(rng)
        lo, hi = cm.total_length_range()
        assert lo <= rc.total_length <= hi
        # The motif length (5) must be included exactly once.
        motif_total = sum(length for seg, length in rc.layout
                          if hasattr(seg, 'chain'))
        assert motif_total == 5

    def test_motif_mask(self):
        rc = fixed_layout(motif_lengths=[3, 2],
                          linker_lengths=[4, 5, 2])
        mask = rc.motif_mask()
        # Layout: 4-linker, 3-motif, 5-linker, 2-motif, 2-linker → L=16.
        expected = np.array(
            [0]*4 + [1]*3 + [0]*5 + [1]*2 + [0]*2, dtype=bool)
        np.testing.assert_array_equal(mask, expected)

    def test_motif_source_index(self):
        rc = fixed_layout(motif_lengths=[3, 2],
                          linker_lengths=[1, 1, 1], chain='B', motif_start=10)
        srcs = rc.motif_source_index()
        assert srcs[0] is None
        assert srcs[1] == ('B', 10)
        assert srcs[2] == ('B', 11)
        assert srcs[3] == ('B', 12)
        assert srcs[4] is None
        assert srcs[5] == ('B', 13)
        assert srcs[6] == ('B', 14)
        assert srcs[7] is None

    def test_fixed_layout_validation(self):
        with pytest.raises(ValueError):
            fixed_layout(motif_lengths=[3], linker_lengths=[1, 1, 1])


class TestMotifMaskBuilding:

    def test_motif_copies_reference_coords(self):
        rc = fixed_layout(motif_lengths=[2], linker_lengths=[1, 1])
        ref = {'A': np.arange(2 * 3 * 3, dtype=np.float32).reshape(2, 3, 3)
               + 10.0}
        coords, mask = build_motif_mask(rc, reference_coords=ref)
        assert coords.shape == (4, 3, 3)
        assert mask.tolist() == [False, True, True, False]
        np.testing.assert_array_equal(coords[1], ref['A'][0])
        np.testing.assert_array_equal(coords[2], ref['A'][1])
        # Linker positions are zero.
        assert (coords[0] == 0).all()
        assert (coords[3] == 0).all()

    def test_missing_chain_raises(self):
        rc = fixed_layout(motif_lengths=[1], linker_lengths=[0, 0])
        with pytest.raises(KeyError):
            build_motif_mask(rc, reference_coords={'B': np.zeros((1, 3, 3))})

    def test_out_of_range_raises(self):
        rc = fixed_layout(motif_lengths=[3], linker_lengths=[0, 0],
                          motif_start=10)
        with pytest.raises(IndexError):
            build_motif_mask(rc, reference_coords={'A': np.zeros((2, 3, 3))})

    def test_atom14_layout(self):
        rc = fixed_layout(motif_lengths=[1], linker_lengths=[0, 0])
        ref = {'A': np.ones((1, 14, 3), dtype=np.float32)}
        coords, _ = build_motif_mask(rc, reference_coords=ref,
                                     atoms_per_residue=14)
        assert coords.shape == (1, 14, 3)
        np.testing.assert_array_equal(coords[0], np.ones((14, 3)))


class TestFreezeMotif:

    def test_freeze_replaces_motif_only(self):
        rc = fixed_layout(motif_lengths=[2], linker_lengths=[1, 1])
        mask = rc.motif_mask()
        ref = np.ones((4, 3, 3), dtype=np.float32) * 7.0
        gen = np.zeros((4, 3, 3), dtype=np.float32)
        out = freeze_motif_coords(gen, ref, mask)
        # Motif positions become reference.
        assert np.allclose(out[1], 7.0)
        assert np.allclose(out[2], 7.0)
        # Linker positions stay zero.
        assert np.allclose(out[0], 0.0)
        assert np.allclose(out[3], 0.0)
        # Reference must remain untouched (function does not mutate).
        assert np.allclose(ref, 7.0)
        assert np.allclose(gen, 0.0)

    def test_motif_freeze_within_tolerance(self):
        """After freezing, the motif positions deviate from the
        reference by < 1e-4 Å (in fact they are exactly equal)."""
        rc = fixed_layout(motif_lengths=[3, 2], linker_lengths=[2, 2, 2])
        ref = np.random.default_rng(0).normal(
            size=(rc.total_length, 3, 3)).astype(np.float32)
        gen = np.random.default_rng(1).normal(
            size=(rc.total_length, 3, 3)).astype(np.float32)
        mask = rc.motif_mask()
        out = freeze_motif_coords(gen, ref, mask)
        err = np.abs(out[mask] - ref[mask]).max()
        assert err < 1e-4

    def test_shape_validation(self):
        with pytest.raises(ValueError):
            freeze_motif_coords(np.zeros((4, 3, 3)), np.zeros((5, 3, 3)),
                                np.zeros(4, dtype=bool))
        with pytest.raises(ValueError):
            freeze_motif_coords(np.zeros((4, 3, 3)), np.zeros((4, 3, 3)),
                                np.zeros(3, dtype=bool))
