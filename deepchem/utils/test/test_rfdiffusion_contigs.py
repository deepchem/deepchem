"""Tests for the RFDiffusion contig parser and motif scaffolding utilities."""

import numpy as np
import pytest

from deepchem.utils.rfdiffusion_contigs import (
    LinkerSegment,
    MotifSegment,
    parse_contig_string,
    build_motif_mask,
    freeze_motif_coords,
    fixed_layout,
)


class TestSegments:
    """Linker and motif segment dataclasses."""

    def test_linker_sample_length_in_range(self):
        """A sampled linker length stays within its inclusive bounds."""
        seg = LinkerSegment(lo=5, hi=10)
        rng = np.random.default_rng(0)
        for _ in range(50):
            assert 5 <= seg.sample_length(rng) <= 10

    def test_linker_fixed_length(self):
        """A linker with lo == hi always samples that exact length."""
        seg = LinkerSegment(lo=7, hi=7)
        assert seg.sample_length(np.random.default_rng(0)) == 7

    def test_linker_invalid_range_raises(self):
        """A reversed or negative linker range is rejected."""
        with pytest.raises(ValueError):
            LinkerSegment(lo=10, hi=5)
        with pytest.raises(ValueError):
            LinkerSegment(lo=-1, hi=5)

    def test_motif_length(self):
        """Motif length counts residues inclusively."""
        assert MotifSegment(chain='A', start=12, end=30).length == 19

    def test_motif_invalid_raises(self):
        """A bad chain id or reversed range is rejected."""
        with pytest.raises(ValueError):
            MotifSegment(chain='AB', start=1, end=5)
        with pytest.raises(ValueError):
            MotifSegment(chain='A', start=10, end=5)


class TestParseContigString:
    """Parsing CLI-style contig strings."""

    def test_parse_mixed(self):
        """A mixed string yields the right segment types, order and range."""
        cm = parse_contig_string('5-10/A12-30/5-10')
        assert isinstance(cm.segments[0], LinkerSegment)
        assert isinstance(cm.segments[1], MotifSegment)
        assert isinstance(cm.segments[2], LinkerSegment)
        assert cm.segments[1].chain == 'A'
        assert cm.total_length_range() == (29, 39)

    def test_parse_fixed_linker(self):
        """A bare integer token is a fixed-length linker."""
        cm = parse_contig_string('8')
        assert cm.segments[0].lo == 8
        assert cm.segments[0].hi == 8

    def test_parse_whitespace_or_slash(self):
        """Tokens may be separated by slashes or whitespace."""
        a = parse_contig_string('A1-5/10-20')
        b = parse_contig_string('A1-5 10-20')
        assert len(a.segments) == len(b.segments) == 2

    def test_parse_empty_raises(self):
        """An empty or whitespace-only string is rejected."""
        with pytest.raises(ValueError):
            parse_contig_string('')
        with pytest.raises(ValueError):
            parse_contig_string('   ')

    def test_parse_bad_token_raises(self):
        """An unrecognised token is rejected."""
        with pytest.raises(ValueError):
            parse_contig_string('A1-5/xyz')


class TestContigMap:
    """Length ranges and realisation."""

    def test_total_length_range(self):
        """Length range sums motif lengths and linker bounds."""
        cm = parse_contig_string('5-10/A1-20/5-10')
        assert cm.total_length_range() == (30, 40)

    def test_realise_length_within_range(self):
        """A realised total length falls inside the declared range."""
        cm = parse_contig_string('5-10/A1-20/5-10')
        rng = np.random.default_rng(0)
        lo, hi = cm.total_length_range()
        for _ in range(20):
            assert lo <= cm.realise(rng).total_length <= hi

    def test_motif_mask_and_sources(self):
        """The mask and source map flag motif residues consistently."""
        r = fixed_layout(motif_lengths=[3],
                         linker_lengths=[2, 2],
                         chain='A',
                         motif_start=5)
        assert r.motif_mask().tolist() == [
            False, False, True, True, True, False, False
        ]
        sources = r.motif_source_index()
        assert sources[2] == ('A', 5)
        assert sources[4] == ('A', 7)
        assert sources[0] is None


class TestMotifMasking:
    """Building and applying motif masks against reference coordinates."""

    def test_build_motif_mask_copies_reference(self):
        """Motif rows copy reference coordinates; linkers stay zero."""
        r = fixed_layout(motif_lengths=[2],
                         linker_lengths=[1, 1],
                         chain='A',
                         motif_start=1)
        ref = {'A': np.arange(2 * 3 * 3).reshape(2, 3, 3).astype(np.float32)}
        coords, mask = build_motif_mask(r, ref)
        assert coords.shape == (4, 3, 3)
        assert mask.tolist() == [False, True, True, False]
        assert np.allclose(coords[1], ref['A'][0])
        assert np.allclose(coords[2], ref['A'][1])
        assert np.allclose(coords[0], 0.0)

    def test_build_motif_mask_missing_chain_raises(self):
        """A motif on an absent reference chain raises KeyError."""
        r = fixed_layout([1], [0, 0], chain='B')
        with pytest.raises(KeyError):
            build_motif_mask(r, {'A': np.zeros((5, 3, 3))})

    def test_build_motif_mask_out_of_range_raises(self):
        """A motif residue past the reference length raises IndexError."""
        r = fixed_layout([3], [0, 0], chain='A', motif_start=1)
        with pytest.raises(IndexError):
            build_motif_mask(r, {'A': np.zeros((2, 3, 3))})

    def test_freeze_motif_coords(self):
        """Freezing overwrites only masked rows and leaves the input alone."""
        gen = np.zeros((3, 3))
        ref = np.ones((3, 3))
        mask = np.array([True, False, True])
        out = freeze_motif_coords(gen, ref, mask)
        assert out[:, 0].tolist() == [1.0, 0.0, 1.0]
        assert np.allclose(gen, 0.0)

    def test_freeze_shape_mismatch_raises(self):
        """Mismatched coordinate shapes raise ValueError."""
        with pytest.raises(ValueError):
            freeze_motif_coords(np.zeros((3, 3)), np.zeros((4, 3)),
                                np.array([True, False, True]))
