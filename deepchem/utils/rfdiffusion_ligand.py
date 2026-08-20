"""Ligand parsing utilities for RFDiffusion All-Atom conditioning.

Turns an on-disk ligand file (SDF, MOL2, or PDB) into a point cloud of
atom coordinates plus the per-atom and per-atom-pair features the
RFdiffusion All-Atom network needs: an element-based atom type, the raw
atomic number, and a bond-order feature matrix. RDKit is imported lazily
so the rest of the diffusion stack stays usable without it; parsing
itself is delegated to :func:`deepchem.utils.rdkit_utils.load_molecule`
rather than reimplementing file-format handling.

The atom type vocabulary and bond feature convention (bond order 1-3,
aromatic bonds coded as 4) follow RFdiffusion All-Atom's own
``rf2aa.chemical.ChemData.frame_priority2atom`` / ``get_bond_feats``
[Krishna2024]_, so a model trained against these features sees the same
atom typing scheme the reference implementation uses. Automorphism
enumeration (:func:`find_ligand_automorphisms`) mirrors
``rf2aa.util.get_automorphs``: it lists every atom-index permutation that
maps the ligand onto itself, so a loss can be evaluated against every
permutation and take the best match instead of penalizing a model for
correctly generating a symmetric ligand (e.g. a fluorinated ring) with a
different, but equally valid, atom-index labeling than the reference.

References
----------
.. [Krishna2024] Krishna, R., et al. "Generalized biomolecular modeling
   and design with RoseTTAFold All-Atom." Science 384 (2024) eadl2528.
"""

from typing import List, Optional

import numpy as np

__all__ = [
    'LIGAND_ATOM_TYPES',
    'LigandPointCloud',
    'parse_ligand_file',
    'find_ligand_automorphisms',
]

# Element vocabulary, ordered by RFdiffusion All-Atom's atom "frame
# priority" list (rf2aa/chemical.py: ChemData.frame_priority2atom), plus
# the catch-all 'ATM' bucket it uses for anything else. The specific
# order does not matter for a point-cloud featurization (it is only used
# as a fixed embedding-table index), but keeping it identical to the
# reference makes it easy to line features up against upstream code.
LIGAND_ATOM_TYPES: List[str] = [
    'F', 'Cl', 'Br', 'I', 'O', 'S', 'Se', 'Te', 'N', 'P', 'As', 'Sb', 'C', 'Si',
    'Sn', 'Pb', 'B', 'Al', 'Zn', 'Hg', 'Cu', 'Au', 'Ni', 'Pd', 'Pt', 'Co', 'Rh',
    'Ir', 'Pr', 'Fe', 'Ru', 'Os', 'Mn', 'Re', 'Cr', 'Mo', 'W', 'V', 'U', 'Tb',
    'Y', 'Be', 'Mg', 'Ca', 'Li', 'K', 'ATM'
]
_ATOM_TYPE_TO_INDEX = {t: i for i, t in enumerate(LIGAND_ATOM_TYPES)}
_UNKNOWN_ATOM_TYPE_INDEX = _ATOM_TYPE_TO_INDEX['ATM']


class LigandPointCloud:
    """Point-cloud representation of a parsed ligand.

    Parameters
    ----------
    coords : numpy.ndarray
        Atom Cartesian coordinates, shape ``(N, 3)``.
    atom_types : numpy.ndarray
        Integer index into :data:`LIGAND_ATOM_TYPES` for each atom, shape
        ``(N,)``.
    atomic_numbers : numpy.ndarray
        Raw atomic number for each atom, shape ``(N,)``.
    bond_features : numpy.ndarray
        Symmetric integer bond-order matrix, shape ``(N, N)``: 0 for no
        bond, 1/2/3 for single/double/triple, 4 for aromatic.
    name : str, optional
        Optional identifier (e.g. the source file path) kept for
        bookkeeping only.

    Examples
    --------
    >>> import numpy as np
    >>> from deepchem.utils.rfdiffusion_ligand import LigandPointCloud
    >>> coords = np.zeros((2, 3), dtype=np.float32)
    >>> atom_types = np.array([4, 12])  # O, C
    >>> atomic_numbers = np.array([8, 6])
    >>> bonds = np.array([[0, 1], [1, 0]])
    >>> ligand = LigandPointCloud(coords, atom_types, atomic_numbers, bonds)
    >>> ligand.num_atoms
    2
    """

    def __init__(self,
                 coords: np.ndarray,
                 atom_types: np.ndarray,
                 atomic_numbers: np.ndarray,
                 bond_features: np.ndarray,
                 name: Optional[str] = None) -> None:
        coords = np.asarray(coords, dtype=np.float32)
        atom_types = np.asarray(atom_types, dtype=np.int64)
        atomic_numbers = np.asarray(atomic_numbers, dtype=np.int64)
        bond_features = np.asarray(bond_features, dtype=np.int64)

        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f'coords must have shape (N, 3), got '
                             f'{coords.shape}.')
        n = coords.shape[0]
        if atom_types.shape != (n,):
            raise ValueError(f'atom_types must have shape ({n},), got '
                             f'{atom_types.shape}.')
        if atomic_numbers.shape != (n,):
            raise ValueError(f'atomic_numbers must have shape ({n},), got '
                             f'{atomic_numbers.shape}.')
        if bond_features.shape != (n, n):
            raise ValueError(f'bond_features must have shape ({n}, {n}), '
                             f'got {bond_features.shape}.')

        self.coords = coords
        self.atom_types = atom_types
        self.atomic_numbers = atomic_numbers
        self.bond_features = bond_features
        self.name = name

    @property
    def num_atoms(self) -> int:
        """Number of atoms in the point cloud."""
        return self.coords.shape[0]


def _element_to_type_index(symbol: str) -> int:
    """Map an element symbol to its :data:`LIGAND_ATOM_TYPES` index."""
    return _ATOM_TYPE_TO_INDEX.get(symbol, _UNKNOWN_ATOM_TYPE_INDEX)


def _bond_feature_matrix(mol) -> np.ndarray:
    """Build the ``(N, N)`` bond-order feature matrix for an RDKit mol."""
    n = mol.GetNumAtoms()
    feats = np.zeros((n, n), dtype=np.int64)
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        order = 4 if bond.GetIsAromatic() else int(bond.GetBondTypeAsDouble())
        feats[i, j] = order
        feats[j, i] = order
    return feats


def parse_ligand_file(molecule_file: str,
                      remove_hydrogens: bool = True) -> LigandPointCloud:
    """Parse a ligand file into a :class:`LigandPointCloud`.

    Supports the same file types as
    :func:`deepchem.utils.rdkit_utils.load_molecule` (SDF, MOL2, PDB,
    PDBQT), which this function calls to do the actual file parsing.

    Parameters
    ----------
    molecule_file : str
        Path to a ligand file. The format is inferred from the file
        extension.
    remove_hydrogens : bool, default True
        Whether to strip explicit hydrogens before building the point
        cloud, matching RFdiffusion All-Atom's default ligand
        preprocessing.

    Returns
    -------
    LigandPointCloud
        Parsed atom coordinates, types, atomic numbers, and bond
        features.

    Raises
    ------
    ImportError
        If RDKit is not installed.
    ValueError
        If the file cannot be parsed, or the parsed molecule has no 3-D
        conformer.

    Examples
    --------
    >>> from rdkit import Chem
    >>> from rdkit.Chem import AllChem
    >>> import tempfile, os
    >>> from deepchem.utils.rfdiffusion_ligand import parse_ligand_file
    >>> mol = Chem.AddHs(Chem.MolFromSmiles('CCO'))
    >>> _ = AllChem.EmbedMolecule(mol, randomSeed=0)
    >>> path = os.path.join(tempfile.mkdtemp(), 'ethanol.sdf')
    >>> writer = Chem.SDWriter(path)
    >>> writer.write(mol)
    >>> writer.close()
    >>> ligand = parse_ligand_file(path)
    >>> ligand.num_atoms
    3
    >>> sorted(ligand.atomic_numbers.tolist())
    [6, 6, 8]
    """
    try:
        from rdkit import Chem
    except ImportError:
        raise ImportError('parse_ligand_file requires RDKit to be '
                          'installed (pip install rdkit).')

    from deepchem.utils.rdkit_utils import load_molecule

    _, mol = load_molecule(molecule_file,
                           add_hydrogens=False,
                           calc_charges=False,
                           sanitize=True)
    if mol is None:
        raise ValueError(f'Could not parse ligand file: {molecule_file}')
    if remove_hydrogens:
        mol = Chem.RemoveHs(mol)
    if mol.GetNumConformers() == 0:
        raise ValueError(
            f'Parsed molecule from {molecule_file} has no 3-D conformer.')

    conf = mol.GetConformer()
    coords = np.array(
        [conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())],
        dtype=np.float32)
    atomic_numbers = np.array([a.GetAtomicNum() for a in mol.GetAtoms()],
                              dtype=np.int64)
    atom_types = np.array(
        [_element_to_type_index(a.GetSymbol()) for a in mol.GetAtoms()],
        dtype=np.int64)
    bond_features = _bond_feature_matrix(mol)

    return LigandPointCloud(coords,
                            atom_types,
                            atomic_numbers,
                            bond_features,
                            name=molecule_file)


def find_ligand_automorphisms(molecule_file: str,
                              remove_hydrogens: bool = True,
                              max_matches: int = 1000) -> np.ndarray:
    """Enumerate graph-automorphism atom permutations for a ligand.

    Two atoms are interchangeable under a permutation exactly when
    swapping them maps the molecular graph onto itself (e.g. the two
    fluorines of a -CF2- group, or the six carbons of an unsubstituted
    benzene ring). Comparing a generated ligand against every such
    permutation and keeping the best-matching one, instead of a single
    fixed atom ordering, is what
    :func:`~deepchem.models.torch_models.rfdiffusion_losses.masked_all_atom_l2_loss`
    and the other losses in this module are meant to be combined with
    for symmetric ligands. Mirrors ``rf2aa.util.get_automorphs``.

    Parameters
    ----------
    molecule_file : str
        Path to a ligand file, as accepted by :func:`parse_ligand_file`.
    remove_hydrogens : bool, default True
        Whether to strip explicit hydrogens before enumerating
        automorphisms, matching :func:`parse_ligand_file`.
    max_matches : int, default 1000
        Upper bound on the number of automorphisms to enumerate, to keep
        highly symmetric ligands (e.g. long unsubstituted chains) from
        producing combinatorially many permutations.

    Returns
    -------
    numpy.ndarray
        Integer array of shape ``(num_automorphisms, num_atoms)``. Row 0
        is always the identity permutation.

    Raises
    ------
    ImportError
        If RDKit is not installed.

    Examples
    --------
    >>> from rdkit import Chem
    >>> from rdkit.Chem import AllChem
    >>> import tempfile, os
    >>> from deepchem.utils.rfdiffusion_ligand import (
    ...     find_ligand_automorphisms)
    >>> mol = Chem.AddHs(Chem.MolFromSmiles('c1ccccc1'))  # benzene
    >>> _ = AllChem.EmbedMolecule(mol, randomSeed=0)
    >>> path = os.path.join(tempfile.mkdtemp(), 'benzene.sdf')
    >>> writer = Chem.SDWriter(path)
    >>> writer.write(mol)
    >>> writer.close()
    >>> perms = find_ligand_automorphisms(path)
    >>> perms.shape[1]
    6
    >>> perms.shape[0] > 1
    True
    """
    try:
        from rdkit import Chem
    except ImportError:
        raise ImportError('find_ligand_automorphisms requires RDKit to be '
                          'installed (pip install rdkit).')

    from deepchem.utils.rdkit_utils import load_molecule

    _, mol = load_molecule(molecule_file,
                           add_hydrogens=False,
                           calc_charges=False,
                           sanitize=True)
    if mol is None:
        raise ValueError(f'Could not parse ligand file: {molecule_file}')
    if remove_hydrogens:
        mol = Chem.RemoveHs(mol)

    matches = mol.GetSubstructMatches(mol,
                                      uniquify=False,
                                      useChirality=True,
                                      maxMatches=max_matches)
    if not matches:
        matches = (tuple(range(mol.GetNumAtoms())),)
    return np.array(matches, dtype=np.int64)
