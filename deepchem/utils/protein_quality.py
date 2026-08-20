"""Protein structure quality metrics for evaluating RFDiffusion output.

These are the structural sanity checks and comparison metrics used to
evaluate generated protein backbones [Watson2023]_: is the structure
compact and free of steric clashes, are consecutive residues a sane
distance apart, and (when a reference structure is available, e.g. a
refolded prediction of a designed sequence) how close is the generated
structure to it.

Every function here is a plain NumPy function operating directly on
coordinate arrays, so they can be used standalone or wrapped in a
``deepchem.metrics.Metric`` where that fits an evaluation pipeline.

References
----------
.. [Watson2023] Watson, J. L., et al. "De novo design of protein
   structure and function with RFdiffusion." Nature 620 (2023)
   1089-1100.
.. [Kabsch1976] Kabsch, W. "A solution for the best rotation to relate
   two sets of vectors." Acta Crystallographica A32 (1976) 922-923.
.. [Zhang2004] Zhang, Y. & Skolnick, J. "Scoring function for automated
   assessment of protein structure template quality." Proteins 57
   (2004) 702-710.
"""

from typing import Dict, Optional, Tuple

import numpy as np

__all__ = [
    'radius_of_gyration',
    'clash_score',
    'kabsch_align',
    'rmsd',
    'sc_rmsd',
    'tm_score',
    'backbone_bond_validity',
]


def radius_of_gyration(coords: np.ndarray,
                       masses: Optional[np.ndarray] = None) -> float:
    r"""Radius of gyration of a point cloud.

    .. math::

        R_g = \sqrt{\frac{\sum_i m_i \lVert x_i - \bar{x} \rVert^2}
                         {\sum_i m_i}}

    where :math:`\bar{x}` is the (mass-weighted) centroid.

    Parameters
    ----------
    coords : numpy.ndarray
        Atom coordinates, shape ``(N, 3)``.
    masses : numpy.ndarray, optional
        Per-atom weights, shape ``(N,)``. Defaults to uniform weights
        (i.e. the unweighted radius of gyration).

    Returns
    -------
    float
        Radius of gyration, in the same length units as ``coords``.

    Raises
    ------
    ValueError
        If ``coords`` is empty or not shape ``(N, 3)``.

    Examples
    --------
    >>> import numpy as np
    >>> from deepchem.utils.protein_quality import radius_of_gyration
    >>> coords = np.array([[-1.0, 0, 0], [1.0, 0, 0]])
    >>> round(radius_of_gyration(coords), 4)
    1.0
    """
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3 or coords.shape[0] == 0:
        raise ValueError(
            f'coords must have shape (N, 3) with N > 0, got {coords.shape}.')
    if masses is None:
        weights = np.ones(coords.shape[0])
    else:
        weights = np.asarray(masses, dtype=np.float64)
        if weights.shape != (coords.shape[0],):
            raise ValueError(
                f'masses must have shape ({coords.shape[0]},), got '
                f'{weights.shape}.')
    centroid = np.average(coords, axis=0, weights=weights)
    sq_dev = np.sum((coords - centroid)**2, axis=1)
    return float(np.sqrt(np.average(sq_dev, weights=weights)))


def clash_score(coords: np.ndarray,
                radii: np.ndarray,
                tolerance: float = 0.4) -> float:
    """Fraction of atom pairs that sterically clash.

    An atom pair clashes when its distance is smaller than the sum of
    the two atoms' radii minus ``tolerance``.

    Parameters
    ----------
    coords : numpy.ndarray
        Atom coordinates, shape ``(N, 3)``.
    radii : numpy.ndarray
        Per-atom radius, shape ``(N,)``. See
        :func:`~deepchem.models.torch_models.rfdiffusion_losses.vdw_radii_from_symbols`
        for a convenience lookup by element.
    tolerance : float, default 0.4
        Overlap allowed before a pair counts as a clash, in the same
        units as ``coords`` (Angstrom for van der Waals radii).

    Returns
    -------
    float
        Number of clashing atom pairs divided by the total number of
        distinct pairs, in ``[0, 1]``.

    Raises
    ------
    ValueError
        If ``coords`` and ``radii`` have inconsistent shapes, or there
        are fewer than 2 atoms.

    Examples
    --------
    >>> import numpy as np
    >>> from deepchem.utils.protein_quality import clash_score
    >>> coords = np.array([[0.0, 0, 0], [10.0, 0, 0]])
    >>> radii = np.array([1.7, 1.7])
    >>> clash_score(coords, radii)
    0.0
    """
    coords = np.asarray(coords, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64)
    n = coords.shape[0]
    if coords.ndim != 2 or coords.shape[1] != 3 or n < 2:
        raise ValueError('coords must have shape (N, 3) with N >= 2.')
    if radii.shape != (n,):
        raise ValueError(f'radii must have shape ({n},), got {radii.shape}.')

    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=-1))
    radii_sum = radii[:, None] + radii[None, :]
    clashing = dist < (radii_sum - tolerance)
    iu = np.triu_indices(n, k=1)
    num_pairs = iu[0].shape[0]
    return float(np.count_nonzero(clashing[iu])) / num_pairs


def kabsch_align(
        mobile: np.ndarray,
        target: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Optimal rigid superposition of one point set onto another.

    Finds the rotation and translation that minimize the RMSD between
    ``mobile`` and ``target`` using the Kabsch algorithm [Kabsch1976]_:
    SVD of the cross-covariance matrix between the two centered point
    sets, with a reflection correction so the result is always a proper
    rotation (``det(R) = +1``).

    Parameters
    ----------
    mobile : numpy.ndarray
        Point set to be aligned, shape ``(N, 3)``.
    target : numpy.ndarray
        Reference point set, shape ``(N, 3)``.

    Returns
    -------
    rotation : numpy.ndarray
        Optimal rotation matrix, shape ``(3, 3)``.
    translation : numpy.ndarray
        Optimal translation vector, shape ``(3,)``.
    aligned : numpy.ndarray
        ``mobile`` after applying the optimal rotation and translation,
        shape ``(N, 3)``.

    Raises
    ------
    ValueError
        If ``mobile`` and ``target`` do not have the same shape ``(N,
        3)`` with ``N > 0``.

    Examples
    --------
    >>> import numpy as np
    >>> from deepchem.utils.protein_quality import kabsch_align
    >>> target = np.array([[0.0, 0, 0], [1.0, 0, 0], [0.0, 1.0, 0]])
    >>> theta = np.pi / 4
    >>> rot = np.array([[np.cos(theta), -np.sin(theta), 0],
    ...                 [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    >>> mobile = target @ rot.T + np.array([5.0, -3.0, 2.0])
    >>> _, _, aligned = kabsch_align(mobile, target)
    >>> bool(np.allclose(aligned, target, atol=1e-6))
    True
    """
    mobile = np.asarray(mobile, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if mobile.shape != target.shape or mobile.ndim != 2 or \
            mobile.shape[1] != 3 or mobile.shape[0] == 0:
        raise ValueError(
            f'mobile and target must both have shape (N, 3) with N > 0, '
            f'got {mobile.shape} and {target.shape}.')

    mobile_centroid = mobile.mean(axis=0)
    target_centroid = target.mean(axis=0)
    mobile_c = mobile - mobile_centroid
    target_c = target - target_centroid

    covariance = mobile_c.T @ target_c
    u, _, vt = np.linalg.svd(covariance)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    correction = np.diag([1.0, 1.0, d])
    rotation = vt.T @ correction @ u.T

    translation = target_centroid - rotation @ mobile_centroid
    aligned = mobile @ rotation.T + translation
    return rotation, translation, aligned


def rmsd(coords1: np.ndarray, coords2: np.ndarray) -> float:
    """Root-mean-square deviation between two point sets, no alignment.

    Parameters
    ----------
    coords1 : numpy.ndarray
        Shape ``(N, 3)``.
    coords2 : numpy.ndarray
        Shape ``(N, 3)``.

    Returns
    -------
    float
        RMSD between the two point sets, in the same units as the
        inputs.

    Examples
    --------
    >>> import numpy as np
    >>> from deepchem.utils.protein_quality import rmsd
    >>> a = np.zeros((3, 3))
    >>> b = np.ones((3, 3))
    >>> round(rmsd(a, b), 4)
    1.7321
    """
    coords1 = np.asarray(coords1, dtype=np.float64)
    coords2 = np.asarray(coords2, dtype=np.float64)
    if coords1.shape != coords2.shape:
        raise ValueError(f'coords1 and coords2 must have the same shape, got '
                         f'{coords1.shape} and {coords2.shape}.')
    return float(np.sqrt(np.mean(np.sum((coords1 - coords2)**2, axis=-1))))


def sc_rmsd(designed: np.ndarray, reference: np.ndarray) -> float:
    """Self-consistency RMSD: best-fit RMSD after Kabsch alignment.

    The standard way to compare a designed backbone with an independent
    structure prediction of its sequence (e.g. from a folding model):
    align the two structures optimally first, since only their internal
    geometry -- not their absolute position/orientation -- should match.

    Parameters
    ----------
    designed : numpy.ndarray
        Designed structure coordinates, shape ``(N, 3)``.
    reference : numpy.ndarray
        Independently predicted structure coordinates, shape ``(N,
        3)``.

    Returns
    -------
    float
        RMSD after optimal superposition.

    Examples
    --------
    >>> import numpy as np
    >>> from deepchem.utils.protein_quality import sc_rmsd
    >>> reference = np.array([[0.0, 0, 0], [1.0, 0, 0], [0.0, 1.0, 0]])
    >>> designed = reference + np.array([3.0, 4.0, 0.0])  # pure translation
    >>> round(sc_rmsd(designed, reference), 6)
    0.0
    """
    _, _, aligned = kabsch_align(designed, reference)
    return rmsd(aligned, reference)


def tm_score(coords1: np.ndarray,
             coords2: np.ndarray,
             target_length: Optional[int] = None) -> float:
    r"""TM-score between two aligned structures [Zhang2004]_.

    After an optimal Kabsch superposition, each residue pair contributes
    :math:`1 / (1 + (d_i / d_0)^2)` to the score, where :math:`d_i` is
    its distance after alignment and

    .. math::

        d_0 = 1.24 \sqrt[3]{L - 15} - 1.8

    for a reference length :math:`L > 15` (the empirical length-
    dependent distance scale from [Zhang2004]_; ``d_0`` is floored at
    0.5 for very short ``L`` where the cube-root formula would otherwise
    give a non-physical value). A score of 1.0 is a perfect match; above
    ~0.5 is generally considered the same fold.

    Parameters
    ----------
    coords1 : numpy.ndarray
        Shape ``(N, 3)``, e.g. a designed structure.
    coords2 : numpy.ndarray
        Shape ``(N, 3)``, the reference structure to align against and
        to normalize the score by (unless ``target_length`` is given).
    target_length : int, optional
        Reference length used in the ``d_0`` formula and to normalize
        the sum. Defaults to ``len(coords2)``, the usual convention when
        both structures have the same, fully-resolved length.

    Returns
    -------
    float
        TM-score in ``(0, 1]``.

    Raises
    ------
    ValueError
        If ``coords1`` and ``coords2`` do not have the same shape ``(N,
        3)`` with ``N > 0``.

    Examples
    --------
    >>> import numpy as np
    >>> from deepchem.utils.protein_quality import tm_score
    >>> reference = np.random.RandomState(0).randn(50, 3) * 5
    >>> round(tm_score(reference, reference), 4)
    1.0
    """
    coords1 = np.asarray(coords1, dtype=np.float64)
    coords2 = np.asarray(coords2, dtype=np.float64)
    if coords1.shape != coords2.shape or coords1.ndim != 2 or \
            coords1.shape[1] != 3 or coords1.shape[0] == 0:
        raise ValueError(
            f'coords1 and coords2 must both have shape (N, 3) with N > 0, '
            f'got {coords1.shape} and {coords2.shape}.')

    length = target_length if target_length is not None else coords2.shape[0]
    d0 = 1.24 * (max(length - 15, 1))**(1.0 / 3.0) - 1.8
    d0 = max(d0, 0.5)

    _, _, aligned = kabsch_align(coords1, coords2)
    d = np.sqrt(np.sum((aligned - coords2)**2, axis=-1))
    return float(np.sum(1.0 / (1.0 + (d / d0)**2)) / length)


def backbone_bond_validity(ca_coords: np.ndarray,
                           expected_distance: float = 3.8,
                           tolerance: float = 0.3) -> Dict[str, object]:
    """Check that consecutive C-alpha atoms are a plausible distance apart.

    A quick structural sanity check for a generated backbone: adjacent
    residues' C-alpha atoms should be about 3.8 Angstrom apart (the
    typical peptide-bond-constrained C-alpha to C-alpha distance); a
    generated structure with residues much closer or farther apart than
    that has a broken or physically implausible backbone somewhere.

    Parameters
    ----------
    ca_coords : numpy.ndarray
        C-alpha coordinates in chain order, shape ``(L, 3)``.
    expected_distance : float, default 3.8
        Expected consecutive C-alpha to C-alpha distance, in Angstrom.
    tolerance : float, default 0.3
        Allowed deviation from ``expected_distance`` before a bond is
        flagged as invalid, in Angstrom.

    Returns
    -------
    dict
        ``{'num_bonds': int, 'num_valid': int, 'fraction_valid': float,
        'invalid_indices': numpy.ndarray}``. ``invalid_indices`` holds
        the indices ``i`` (0-based, into ``ca_coords[:-1]``) of bonds
        between residue ``i`` and ``i + 1`` that fall outside tolerance.

    Raises
    ------
    ValueError
        If ``ca_coords`` does not have shape ``(L, 3)`` with ``L >= 2``.

    Examples
    --------
    >>> import numpy as np
    >>> from deepchem.utils.protein_quality import backbone_bond_validity
    >>> ca = np.array([[0.0, 0, 0], [3.8, 0, 0], [7.6, 0, 0], [50.0, 0, 0]])
    >>> result = backbone_bond_validity(ca)
    >>> result['num_bonds']
    3
    >>> result['num_valid']
    2
    >>> result['invalid_indices'].tolist()
    [2]
    """
    ca_coords = np.asarray(ca_coords, dtype=np.float64)
    if ca_coords.ndim != 2 or ca_coords.shape[1] != 3 or \
            ca_coords.shape[0] < 2:
        raise ValueError(f'ca_coords must have shape (L, 3) with L >= 2, got '
                         f'{ca_coords.shape}.')

    deltas = ca_coords[1:] - ca_coords[:-1]
    distances = np.sqrt(np.sum(deltas**2, axis=-1))
    valid = np.abs(distances - expected_distance) <= tolerance
    invalid_indices = np.nonzero(~valid)[0]
    num_bonds = distances.shape[0]
    return {
        'num_bonds': num_bonds,
        'num_valid': int(np.count_nonzero(valid)),
        'fraction_valid': float(np.count_nonzero(valid)) / num_bonds,
        'invalid_indices': invalid_indices,
    }
