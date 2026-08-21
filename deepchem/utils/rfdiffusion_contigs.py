"""Contig-map parsing and motif scaffolding utilities for RFDiffusion.

RFDiffusion's inpainting / motif-scaffolding pipeline [Watson2023]_ uses a
*contig string* to describe an output protein in terms of fixed motif
segments copied from a reference structure interspersed with sampled
linkers of variable length. This module implements a self-contained
parser and a coordinate-masking helper that the diffusion sampler can
use to keep motif atoms frozen at the reference geometry while the
remainder of the chain is denoised.

The contig grammar is the same dialect accepted by the official
RFDiffusion CLI:

* ``"10-20"`` — a sampled linker of between 10 and 20 residues.
* ``"A12-30"`` — copy residues 12-30 of chain ``A`` from the reference.
* ``"A12-30/0 5-10"`` — chain-break after the motif then 5-10 linker.
* ``"/"`` — separator between segments.

Each contig segment is one of two types:

``MotifSegment``
    Fixed-coordinate residues sourced from the reference.
``LinkerSegment``
    Variable-length sampled residues with an inclusive ``[lo, hi]``
    range.

Developer-facing summary
------------------------
Call ``parse_contig_string`` first to turn a CLI-style contig string
into a :class:`ContigMap`. Then call ``realise()`` to sample concrete
linker lengths, ``build_motif_mask`` to align the realised layout with
reference coordinates, and ``freeze_motif_coords`` inside the sampling
loop to keep motif residues fixed.

References
----------
.. [Watson2023] Watson et al. "De novo design of protein structure and
   function with RFdiffusion." Nature 620 (2023) 1089-1100.
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

try:
    import numpy as np
except ModuleNotFoundError:
    raise ImportError('rfdiffusion_contigs requires NumPy to be installed.')

__all__ = [
    'LinkerSegment',
    'MotifSegment',
    'ContigMap',
    'parse_contig_string',
    'build_motif_mask',
]

_MOTIF_RE = re.compile(r'^([A-Za-z])(-?\d+)-(-?\d+)$')
_LINKER_RE = re.compile(r'^(\d+)-(\d+)$')
_FIXED_LINKER_RE = re.compile(r'^(\d+)$')


@dataclass(frozen=True)
class LinkerSegment:
    """Variable-length sampled linker segment.

    Parameters
    ----------
    lo : int
        Inclusive lower bound on the linker length.
    hi : int
        Inclusive upper bound on the linker length (``hi ≥ lo``).
    """

    lo: int
    hi: int

    def sample_length(self,
                      rng: Optional[np.random.Generator] = None) -> int:
        """Draw a length uniformly from [lo, hi]."""
        if rng is None:
            rng = np.random.default_rng()
        return int(rng.integers(self.lo, self.hi + 1))

    def __post_init__(self) -> None:
        if self.lo < 0 or self.hi < self.lo:
            raise ValueError(
                f'Invalid linker range [{self.lo}, {self.hi}].')


@dataclass(frozen=True)
class MotifSegment:
    """Fixed motif segment copied from a reference structure.

    Parameters
    ----------
    chain : str
        Single-letter chain identifier in the reference.
    start : int
        Inclusive start residue index (PDB numbering) in the reference.
    end : int
        Inclusive end residue index (PDB numbering) in the reference.
    """

    chain: str
    start: int
    end: int

    def __post_init__(self) -> None:
        if len(self.chain) != 1 or not self.chain.isalpha():
            raise ValueError(
                f'Motif chain must be a single letter; got {self.chain!r}.')
        if self.end < self.start:
            raise ValueError(
                f'Motif end {self.end} < start {self.start}.')

    @property
    def length(self) -> int:
        """Number of residues in the motif (end − start + 1)."""
        return self.end - self.start + 1


Segment = Union[LinkerSegment, MotifSegment]


@dataclass
class ContigMap:
    """Parsed contig map describing a generated chain.

    Parameters
    ----------
    segments : list of Segment
        Ordered list of motif and linker segments.
    """

    segments: List[Segment]

    def total_length_range(self) -> Tuple[int, int]:
        """Return ``(min_total, max_total)`` for the chain length."""
        lo = 0
        hi = 0
        for seg in self.segments:
            if isinstance(seg, MotifSegment):
                lo += seg.length
                hi += seg.length
            else:
                lo += seg.lo
                hi += seg.hi
        return lo, hi

    def realise(self,
                rng: Optional[np.random.Generator] = None
                ) -> 'RealisedContig':
        """Sample concrete linker lengths and produce a ``RealisedContig``.

        Parameters
        ----------
        rng : numpy.random.Generator, optional
            Random number generator.

        Returns
        -------
        RealisedContig
            Concrete plan with sampled linker lengths and the implied
            mapping from generated positions to motif source residues.
        """
        if rng is None:
            rng = np.random.default_rng()
        layout: List[Tuple[Segment, int]] = []
        for seg in self.segments:
            if isinstance(seg, MotifSegment):
                layout.append((seg, seg.length))
            else:
                layout.append((seg, seg.sample_length(rng)))
        return RealisedContig(layout=layout)


@dataclass
class RealisedContig:
    """Concrete chain layout with sampled linker lengths.

    Attributes
    ----------
    layout : list of (Segment, int)
        Ordered list of ``(segment, realised_length)`` pairs.
    """

    layout: List[Tuple[Segment, int]]

    @property
    def total_length(self) -> int:
        """Realised chain length (sum of motif + sampled linker lengths)."""
        return sum(length for _, length in self.layout)

    def motif_mask(self) -> np.ndarray:
        """Boolean mask of shape ``(L,)`` flagging fixed motif positions.

        Returns
        -------
        numpy.ndarray
            Array of ``bool`` with ``True`` at motif residues.
        """
        mask = np.zeros(self.total_length, dtype=bool)
        cursor = 0
        for seg, length in self.layout:
            if isinstance(seg, MotifSegment):
                mask[cursor:cursor + length] = True
            cursor += length
        return mask

    def motif_source_index(self) -> List[Optional[Tuple[str, int]]]:
        """Per-position list of source ``(chain, residue_index)`` tuples.

        Returns
        -------
        list of (str, int) or None
            ``None`` for sampled-linker residues; ``(chain, idx)`` for
            motif residues. Length matches ``total_length``.
        """
        sources: List[Optional[Tuple[str, int]]] = []
        for seg, length in self.layout:
            if isinstance(seg, MotifSegment):
                for offset in range(length):
                    sources.append((seg.chain, seg.start + offset))
            else:
                sources.extend([None] * length)
        return sources


def parse_contig_string(text: str) -> ContigMap:
    """Parse a contig specification string into a :class:`ContigMap`.

    Tokens are separated by '/' or whitespace. Each token is one of:

    * ``<chain><start>-<end>`` — motif segment.
    * ``<lo>-<hi>`` — variable linker.
    * ``<n>`` — fixed-length linker of exactly ``n`` residues.

    Parameters
    ----------
    text : str
        Contig string in the RFDiffusion CLI dialect.

    Returns
    -------
    ContigMap
        Parsed contig map.

    Raises
    ------
    ValueError
        If a token does not match any of the accepted patterns.

    Examples
    --------
    >>> cm = parse_contig_string('5-10/A12-30/5-10')
    >>> cm.total_length_range()
    (29, 39)
    """
    if not text or not text.strip():
        raise ValueError('Empty contig string.')
    tokens = [tok for tok in re.split(r'[/\s]+', text.strip()) if tok]
    segments: List[Segment] = []
    for tok in tokens:
        m_motif = _MOTIF_RE.match(tok)
        if m_motif is not None:
            chain = m_motif.group(1)
            start = int(m_motif.group(2))
            end = int(m_motif.group(3))
            segments.append(MotifSegment(chain=chain, start=start, end=end))
            continue
        m_linker = _LINKER_RE.match(tok)
        if m_linker is not None:
            lo = int(m_linker.group(1))
            hi = int(m_linker.group(2))
            segments.append(LinkerSegment(lo=lo, hi=hi))
            continue
        m_fixed = _FIXED_LINKER_RE.match(tok)
        if m_fixed is not None:
            n = int(m_fixed.group(1))
            segments.append(LinkerSegment(lo=n, hi=n))
            continue
        raise ValueError(f'Unrecognised contig token: {tok!r}')
    if not segments:
        raise ValueError('Contig string produced no segments.')
    return ContigMap(segments=segments)


def build_motif_mask(realised: RealisedContig,
                     reference_coords: Dict[str, np.ndarray],
                     reference_offset: Optional[Dict[str, int]] = None,
                     atoms_per_residue: int = 3,
                     atom_dim: int = 3
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """Build aligned ``(reference_coords, motif_mask)`` arrays.

    Iterates over a :class:`RealisedContig` and, for every position whose
    source is a motif residue, copies the corresponding atom coordinates
    from ``reference_coords[chain]`` into the output array. Linker
    positions receive zeros and the mask flags them as not fixed.

    Parameters
    ----------
    realised : RealisedContig
        Realised contig with concrete linker lengths.
    reference_coords : dict
        Mapping from chain id to ``(N_chain, atoms_per_residue, atom_dim)``
        array of reference coordinates. PDB residue indices map to
        array rows via ``reference_offset[chain]`` (default 0 — i.e.
        residue 1 lives at array index ``1 - offset``).
    reference_offset : dict, optional
        Mapping from chain id to the PDB residue index that lives at
        row 0 of the reference array. Defaults to ``{c: 1 for c in
        reference_coords}`` matching the most common PDB convention.
    atoms_per_residue : int, default 3
        Number of atoms stored per residue (3 for N/CA/C backbones,
        14 for AlphaFold2 atom14).
    atom_dim : int, default 3
        Atom coordinate dimensionality (3 for ℝ³).

    Returns
    -------
    coords : numpy.ndarray
        Array of shape ``(L, atoms_per_residue, atom_dim)`` with motif
        coordinates filled in.
    motif_mask : numpy.ndarray
        Boolean array of shape ``(L,)`` — ``True`` for fixed motif
        positions.

    Raises
    ------
    KeyError
        If a motif refers to a chain not present in ``reference_coords``.
    IndexError
        If a motif residue index is out of range for its chain.
    """
    if reference_offset is None:
        reference_offset = {c: 1 for c in reference_coords}
    total_length = realised.total_length
    coords = np.zeros((total_length, atoms_per_residue, atom_dim),
                      dtype=np.float32)
    mask = realised.motif_mask()
    sources = realised.motif_source_index()
    for out_idx, src in enumerate(sources):
        if src is None:
            continue
        chain, res = src
        if chain not in reference_coords:
            raise KeyError(
                f'Reference does not contain chain {chain!r}.')
        offset = reference_offset.get(chain, 1)
        row = res - offset
        ref = reference_coords[chain]
        if row < 0 or row >= ref.shape[0]:
            raise IndexError(
                f'Motif residue {chain}{res} is out of range; '
                f'reference chain {chain} has {ref.shape[0]} residues '
                f'starting at PDB index {offset}.')
        coords[out_idx] = ref[row, :atoms_per_residue, :atom_dim]
    return coords, mask


def freeze_motif_coords(generated: np.ndarray,
                        reference: np.ndarray,
                        motif_mask: np.ndarray) -> np.ndarray:
    """Overwrite motif positions of ``generated`` with ``reference``.

    Parameters
    ----------
    generated : numpy.ndarray
        Array of shape ``(L, ...)`` containing the current generated
        coordinates.
    reference : numpy.ndarray
        Array of shape ``(L, ...)`` with the same suffix dims as
        ``generated`` containing the frozen reference coordinates.
    motif_mask : numpy.ndarray
        Boolean array of shape ``(L,)`` flagging motif positions.

    Returns
    -------
    numpy.ndarray
        New array equal to ``reference`` at motif positions and
        ``generated`` elsewhere. The input arrays are not modified.

    Raises
    ------
    ValueError
        If the shapes are inconsistent.
    """
    if generated.shape != reference.shape:
        raise ValueError(
            f'Shape mismatch: generated {generated.shape} vs '
            f'reference {reference.shape}.')
    if motif_mask.shape[0] != generated.shape[0]:
        raise ValueError(
            f'motif_mask length {motif_mask.shape[0]} does not match '
            f'coordinate length {generated.shape[0]}.')
    out = generated.copy()
    out[motif_mask] = reference[motif_mask]
    return out


def realised_to_index_arrays(
        realised: RealisedContig) -> Tuple[np.ndarray, np.ndarray]:
    """Return (motif_indices, linker_indices) arrays of int positions."""
    mask = realised.motif_mask()
    idx = np.arange(realised.total_length, dtype=np.int64)
    return idx[mask], idx[~mask]


def fixed_layout(motif_lengths: Sequence[int],
                 linker_lengths: Sequence[int],
                 chain: str = 'A',
                 motif_start: int = 1) -> RealisedContig:
    """Build a deterministic ``RealisedContig`` from explicit lengths.

    Convenience constructor used in tests where we want a deterministic
    motif placement without invoking the random sampler.

    Parameters
    ----------
    motif_lengths : sequence of int
        Length of each motif block.
    linker_lengths : sequence of int
        Length of each linker block. Must satisfy
        ``len(linker_lengths) == len(motif_lengths) + 1``: layout is
        linker-motif-linker-motif-…-linker.
    chain : str, default 'A'
        Reference chain id assigned to every motif.
    motif_start : int, default 1
        Starting PDB residue index assigned to the first motif.

    Returns
    -------
    RealisedContig
        Deterministic realised contig.
    """
    if len(linker_lengths) != len(motif_lengths) + 1:
        raise ValueError(
            'linker_lengths must have one more entry than motif_lengths.')
    layout: List[Tuple[Segment, int]] = []
    cursor = motif_start
    for i, lin_len in enumerate(linker_lengths):
        layout.append((LinkerSegment(lo=lin_len, hi=lin_len), lin_len))
        if i < len(motif_lengths):
            m = MotifSegment(chain=chain, start=cursor,
                             end=cursor + motif_lengths[i] - 1)
            layout.append((m, motif_lengths[i]))
            cursor += motif_lengths[i]
    return RealisedContig(layout=layout)
