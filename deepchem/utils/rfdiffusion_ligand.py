"""Minimal optional ligand parser for RFDiffusion-style conditioning.

This utility reads an SDF / MOL file via RDKit and returns a
``(num_atoms, 3)`` Cartesian point cloud plus per-atom integer atomic
numbers. RDKit is imported lazily so the rest of the diffusion stack
remains usable without it.

Developer-facing summary
------------------------
Use ``parse_sdf`` to turn an on-disk ligand file into one or more
``LigandPointCloud`` objects. Downstream code can then call
``closest_distances`` or ``pairwise_distances`` to build residue-to-
ligand geometry features without depending on RDKit after parsing.

References
----------
.. [Watson2023] Watson et al. "De novo design of protein structure and
   function with RFdiffusion." Nature 620 (2023) 1089-1100.
"""

from typing import List, Optional, Tuple

import numpy as np

__all__ = [
    'LigandPointCloud',
    'parse_sdf',
]


class LigandPointCloud:
    """Light-weight container for ligand point-cloud features.

    Parameters
    ----------
    coords : numpy.ndarray
        Array of shape ``(N, 3)`` of atom Cartesian coordinates.
    atomic_numbers : numpy.ndarray
        Integer array of shape ``(N,)`` with atomic numbers.
    name : str, optional
        Human-readable ligand identifier.
    """

    __slots__ = ('coords', 'atomic_numbers', 'name')

    def __init__(self,
                 coords: np.ndarray,
                 atomic_numbers: np.ndarray,
                 name: Optional[str] = None) -> None:
        coords = np.asarray(coords, dtype=np.float32)
        atomic_numbers = np.asarray(atomic_numbers, dtype=np.int32)
        if coords.ndim != 2 or coords.shape[-1] != 3:
            raise ValueError(
                f'coords must have shape (N, 3); got {coords.shape}.')
        if atomic_numbers.shape != (coords.shape[0],):
            raise ValueError(
                'atomic_numbers must be of shape (N,) matching coords.')
        self.coords = coords
        self.atomic_numbers = atomic_numbers
        self.name = name

    @property
    def num_atoms(self) -> int:
        """Number of atoms in the ligand."""
        return int(self.coords.shape[0])

    def center(self) -> 'LigandPointCloud':
        """Return a copy translated so the centroid is at the origin."""
        coords = self.coords - self.coords.mean(axis=0, keepdims=True)
        return LigandPointCloud(coords, self.atomic_numbers, self.name)


def parse_sdf(path: str,
              heavy_atoms_only: bool = True,
              first_only: bool = True) -> List[LigandPointCloud]:
    """Read an SDF file into ``LigandPointCloud`` objects.

    Parameters
    ----------
    path : str
        Path to the SDF / MOL file.
    heavy_atoms_only : bool, default True
        If ``True`` drop explicit hydrogens before extracting atoms.
    first_only : bool, default True
        If ``True`` parse only the first molecule in the file.

    Returns
    -------
    list of LigandPointCloud
        One entry per molecule successfully parsed.

    Raises
    ------
    ImportError
        If RDKit is not available.
    """
    try:
        from rdkit import Chem
    except ImportError as exc:  # pragma: no cover - exercised only without rdkit
        raise ImportError(
            'parse_sdf requires RDKit. Install with `pip install rdkit`.'
        ) from exc
    suppl = Chem.SDMolSupplier(path, removeHs=heavy_atoms_only)
    clouds: List[LigandPointCloud] = []
    for mol in suppl:
        if mol is None:
            continue
        conf = mol.GetConformer() if mol.GetNumConformers() else None
        if conf is None:
            continue
        coords = np.array([
            [conf.GetAtomPosition(i).x,
             conf.GetAtomPosition(i).y,
             conf.GetAtomPosition(i).z]
            for i in range(mol.GetNumAtoms())], dtype=np.float32)
        atomic_numbers = np.array(
            [a.GetAtomicNum() for a in mol.GetAtoms()], dtype=np.int32)
        clouds.append(LigandPointCloud(
            coords, atomic_numbers, name=mol.GetProp('_Name') or None))
        if first_only:
            break
    return clouds


def pairwise_distances(query: np.ndarray,
                       target: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean distances between two point sets.

    Parameters
    ----------
    query : numpy.ndarray
        Array of shape ``(M, 3)``.
    target : numpy.ndarray
        Array of shape ``(N, 3)``.

    Returns
    -------
    numpy.ndarray
        Distance matrix of shape ``(M, N)``.
    """
    diff = query[:, None, :] - target[None, :, :]
    return np.sqrt((diff * diff).sum(-1))


def closest_distances(ligand: 'LigandPointCloud',
                      protein_ca: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-residue closest ligand-atom distance.

    Parameters
    ----------
    ligand : LigandPointCloud
        Ligand point cloud.
    protein_ca : numpy.ndarray
        Array of shape ``(L, 3)`` of protein Cα coordinates.

    Returns
    -------
    distances : numpy.ndarray
        Array of shape ``(L,)`` with the distance from each Cα to its
        nearest ligand atom.
    nearest_atom : numpy.ndarray
        Integer array of shape ``(L,)`` with the index of the nearest
        ligand atom for each residue.
    """
    dmat = pairwise_distances(protein_ca, ligand.coords)
    nearest_atom = np.argmin(dmat, axis=1).astype(np.int32)
    distances = dmat[np.arange(dmat.shape[0]), nearest_atom]
    return distances, nearest_atom
