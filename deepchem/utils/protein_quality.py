"""Protein-structure quality metrics for RFDiffusion evaluation.

All functions here are stateless: they take ``numpy`` arrays (or
file paths for the self-consistency metrics) and return a Python
``float``, so they wrap cleanly as ``deepchem.metrics.Metric``
callables.

Implemented metrics
-------------------
* :func:`radius_of_gyration` — R_g = √(Σ ‖x_i − x̄‖² / N).
* :func:`clash_score` — fraction of atom pairs whose distance is
  smaller than r_i + r_j − threshold Å.
* :func:`sc_rmsd` — self-consistency RMSD between a designed PDB and
  its refolded prediction. The refolder is provided as a callable so
  no hard dependency on ESMFold/AlphaFold/etc. is introduced.
* :func:`sc_tm` — self-consistency TM-score, computed with a pure
  Python implementation of the Zhang-Skolnick formula
  [Zhang2004]_ — no ``TMalign`` binary required.

References
----------
.. [Zhang2004] Zhang, Y. & Skolnick, J. "Scoring function for
   automated assessment of protein structure template quality."
   Proteins 57 (2004) 702-710.
.. [Kabsch1976] Kabsch, W. "A solution for the best rotation to
   relate two sets of vectors." Acta Crystallographica A32 (1976)
   922-923.
.. [Watson2023] Watson, J. L., et al. "De novo design of protein
   structure and function with RFdiffusion." Nature 620 (2023)
   1089-1100.
"""

from typing import Callable, Optional

import numpy as np

__all__ = [
    'radius_of_gyration',
    'clash_score',
    'sc_rmsd',
    'sc_tm',
    'kabsch_align',
    'tm_score',
    'load_ca_coordinates',
]


# ----------------------------------------------------------------------
# Geometric metrics on bare coordinate arrays.
# ----------------------------------------------------------------------
def radius_of_gyration(coords: np.ndarray) -> float:
    """Radius of gyration of a point cloud.

    Computes ``R_g = √( Σ_i ‖x_i − x̄‖² / N )`` where ``x̄`` is the
    centroid of ``coords``.

    Parameters
    ----------
    coords : numpy.ndarray
        Array of shape ``(N, 3)`` or ``(N, A, 3)``. Multi-atom inputs
        are flattened to a single point cloud of size ``N · A``.

    Returns
    -------
    float
        Radius of gyration in the same units as ``coords``.

    Raises
    ------
    ValueError
        If ``coords`` does not have a trailing dimension of 3 or
        contains fewer than one point.
    """
    arr = np.asarray(coords, dtype=np.float64)
    if arr.ndim < 2 or arr.shape[-1] != 3:
        raise ValueError(
            f'coords must end in a dimension of size 3; got shape {arr.shape}.')
    pts = arr.reshape(-1, 3)
    if pts.shape[0] == 0:
        raise ValueError('coords must contain at least one point.')
    centroid = pts.mean(axis=0)
    sq = ((pts - centroid) ** 2).sum(axis=1)
    return float(np.sqrt(sq.mean()))


def clash_score(coords: np.ndarray,
                atom_radii: np.ndarray,
                threshold: float = 0.6) -> float:
    """Fraction of atom pairs whose distance violates van-der-Waals contact.

    A pair (i, j) (with i < j) is counted as a clash when

    .. math::

        \\| x_i - x_j \\|_2 < r_i + r_j - t,

    where ``t`` is the ``threshold`` argument (default 0.6 Å, matching
    the convention used by the RFDiffusion evaluation pipeline
    [Watson2023]_). The score returned is the fraction
    ``num_clashes / num_pairs`` and therefore lies in ``[0, 1]``.

    Parameters
    ----------
    coords : numpy.ndarray
        Array of shape ``(N, 3)`` containing heavy-atom coordinates.
    atom_radii : numpy.ndarray
        Array of shape ``(N,)`` containing the van-der-Waals radius
        of each atom in the same units as ``coords``.
    threshold : float, default 0.6
        Tolerance ``t`` subtracted from the sum of radii. A positive
        value allows the atoms to overlap by that amount before
        being declared clashing.

    Returns
    -------
    float
        Clash fraction in ``[0, 1]``.
    """
    coords = np.asarray(coords, dtype=np.float64)
    radii = np.asarray(atom_radii, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError('coords must have shape (N, 3).')
    if radii.shape != (coords.shape[0],):
        raise ValueError('atom_radii must have shape (N,).')
    n = coords.shape[0]
    if n < 2:
        return 0.0
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff * diff).sum(-1))
    cutoff = radii[:, None] + radii[None, :] - float(threshold)
    iu = np.triu_indices(n, k=1)
    pair_dist = dist[iu]
    pair_cut = cutoff[iu]
    clashes = int((pair_dist < pair_cut).sum())
    return float(clashes) / float(pair_dist.size)


# ----------------------------------------------------------------------
# Kabsch alignment + TM-score (Zhang-Skolnick).
# ----------------------------------------------------------------------
def kabsch_align(mobile: np.ndarray,
                 target: np.ndarray) -> np.ndarray:
    """Return ``mobile`` superposed onto ``target`` (Kabsch algorithm).

    Implements the closed-form least-squares rigid alignment of
    [Kabsch1976]_. Reflections are removed by the standard
    determinant sign correction.

    Parameters
    ----------
    mobile, target : numpy.ndarray
        Coordinate arrays of shape ``(N, 3)`` with matching ``N``.

    Returns
    -------
    numpy.ndarray
        Aligned copy of ``mobile`` of shape ``(N, 3)``.
    """
    mobile = np.asarray(mobile, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if mobile.shape != target.shape:
        raise ValueError('mobile and target must have the same shape.')
    if mobile.ndim != 2 or mobile.shape[1] != 3:
        raise ValueError('inputs must have shape (N, 3).')
    cm = mobile.mean(axis=0)
    ct = target.mean(axis=0)
    p = mobile - cm
    q = target - ct
    h = p.T @ q
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    diag = np.diag([1.0, 1.0, d if d != 0.0 else 1.0])
    r = vt.T @ diag @ u.T
    return (p @ r.T) + ct


def _zhang_d0(length: int) -> float:
    """Normalisation scale ``d_0(L)`` from the Zhang-Skolnick formula."""
    if length < 17:
        return 0.5
    return 1.24 * (length - 15) ** (1.0 / 3.0) - 1.8


def tm_score(mobile: np.ndarray,
             target: np.ndarray,
             length_normaliser: Optional[int] = None) -> float:
    """Pure-Python TM-score [Zhang2004]_.

    Computes

    .. math::

        \\mathrm{TM} = \\frac{1}{L_{\\rm ref}}
            \\sum_{i=1}^{L_{\\rm aln}}
            \\frac{1}{1 + (d_i / d_0(L_{\\rm ref}))^2},

    where ``L_ref`` defaults to the length of ``target`` and ``d_i``
    is the per-residue distance after Kabsch superposition.

    Parameters
    ----------
    mobile, target : numpy.ndarray
        Cα-coordinate arrays of shape ``(L, 3)`` with matching ``L``.
        Residues are assumed to be in 1:1 correspondence.
    length_normaliser : int, optional
        Length used for the ``d_0`` normalisation. Defaults to the
        target length, matching the convention of TMalign's
        ``-l ref`` flag.

    Returns
    -------
    float
        TM-score in ``(0, 1]``. Values above 0.5 indicate the same
        global fold; values below 0.2 are statistically random.
    """
    mobile = np.asarray(mobile, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if mobile.shape != target.shape:
        raise ValueError('mobile and target must have the same shape.')
    if mobile.ndim != 2 or mobile.shape[1] != 3:
        raise ValueError('inputs must have shape (L, 3).')
    length = mobile.shape[0]
    l_ref = int(length_normaliser) if length_normaliser is not None else length
    if l_ref <= 0:
        raise ValueError('length_normaliser must be positive.')
    aligned = kabsch_align(mobile, target)
    di = np.linalg.norm(aligned - target, axis=1)
    d0 = _zhang_d0(l_ref)
    return float(np.sum(1.0 / (1.0 + (di / d0) ** 2)) / l_ref)


# ----------------------------------------------------------------------
# Self-consistency metrics (pluggable refolder, no external binaries).
# ----------------------------------------------------------------------
def load_ca_coordinates(pdb_path: str) -> np.ndarray:
    """Load Cα coordinates from a PDB file (requires BioPython).

    Parameters
    ----------
    pdb_path : str
        Path to a PDB file.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(L, 3)`` containing Cα coordinates of the
        first model, in residue order across all chains.
    """
    try:
        from Bio.PDB import PDBParser, is_aa
    except ImportError as exc:
        raise ImportError(
            'load_ca_coordinates requires BioPython.') from exc
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('p', pdb_path)
    coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if not is_aa(residue, standard=True):
                    continue
                if 'CA' in residue:
                    coords.append(residue['CA'].get_coord())
        break  # only first model
    if not coords:
        raise ValueError(f'No Cα atoms found in {pdb_path}.')
    return np.asarray(coords, dtype=np.float64)


def sc_rmsd(designed_pdb: str,
            refolded_pdb: str,
            refolder: Optional[Callable[[str], str]] = None) -> float:
    """Self-consistency RMSD between a designed and a refolded structure.

    The function computes the Kabsch-aligned Cα RMSD between
    ``designed_pdb`` and ``refolded_pdb``. If ``refolded_pdb`` is
    ``None`` *or* a non-existent path *and* ``refolder`` is supplied,
    the refolder is invoked with ``designed_pdb`` and is expected to
    return the path to a refolded PDB. This keeps the API decoupled
    from any specific structure-prediction backend (ESMFold,
    AlphaFold, RoseTTAFold, …).

    Parameters
    ----------
    designed_pdb : str
        Path to the designed structure (PDB file).
    refolded_pdb : str
        Path to the refolded structure. May be ``''`` or ``None`` if
        a ``refolder`` callable is supplied.
    refolder : Callable[[str], str], optional
        Function ``designed_pdb -> refolded_pdb_path``. Invoked only
        when ``refolded_pdb`` is missing.

    Returns
    -------
    float
        Cα RMSD in Å.
    """
    import os
    if not refolded_pdb or not os.path.exists(refolded_pdb):
        if refolder is None:
            raise ValueError(
                'refolded_pdb does not exist and no refolder was supplied.')
        refolded_pdb = refolder(designed_pdb)
    designed = load_ca_coordinates(designed_pdb)
    refolded = load_ca_coordinates(refolded_pdb)
    if designed.shape != refolded.shape:
        raise ValueError(
            f'Length mismatch: designed has {designed.shape[0]} residues, '
            f'refolded has {refolded.shape[0]}.')
    aligned = kabsch_align(designed, refolded)
    diff = aligned - refolded
    return float(np.sqrt((diff * diff).sum(axis=1).mean()))


def sc_tm(designed_pdb: str,
          refolded_pdb: str,
          refolder: Optional[Callable[[str], str]] = None) -> float:
    """Self-consistency TM-score between a designed and a refolded structure.

    See :func:`sc_rmsd` for the refolder protocol. The TM-score is
    computed via the pure-Python :func:`tm_score`, so no external
    ``TMalign`` binary is required.

    Parameters
    ----------
    designed_pdb : str
        Path to the designed structure (PDB).
    refolded_pdb : str
        Path to the refolded structure, or empty/missing to defer to
        ``refolder``.
    refolder : Callable[[str], str], optional
        Function ``designed_pdb -> refolded_pdb_path``.

    Returns
    -------
    float
        TM-score in ``(0, 1]``.
    """
    import os
    if not refolded_pdb or not os.path.exists(refolded_pdb):
        if refolder is None:
            raise ValueError(
                'refolded_pdb does not exist and no refolder was supplied.')
        refolded_pdb = refolder(designed_pdb)
    designed = load_ca_coordinates(designed_pdb)
    refolded = load_ca_coordinates(refolded_pdb)
    if designed.shape != refolded.shape:
        raise ValueError(
            f'Length mismatch: designed has {designed.shape[0]} residues, '
            f'refolded has {refolded.shape[0]}.')
    return tm_score(designed, refolded)
