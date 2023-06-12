"""
RDKit Utilities.

This file contains utilities that compute useful properties of
molecules. Some of these are simple cleanup utilities, and
others are more sophisticated functions that detect chemical
properties of molecules.
"""

import os
import logging
import itertools
import numpy as np
from io import StringIO
from deepchem.utils.pdbqt_utils import pdbqt_to_pdb
from deepchem.utils.pdbqt_utils import convert_mol_to_pdbqt
from deepchem.utils.pdbqt_utils import convert_protein_to_pdbqt
from deepchem.utils.geometry_utils import compute_pairwise_distances
from deepchem.utils.geometry_utils import compute_centroid
from deepchem.utils.fragment_utils import MolecularFragment
from deepchem.utils.fragment_utils import MoleculeLoadException
from typing import Any, List, Sequence, Tuple, Set, Optional, Dict, Union
from deepchem.utils.typing import OneOrMany, RDKitMol

logger = logging.getLogger(__name__)


def get_xyz_from_mol(mol):
    """Extracts a numpy array of coordinates from a molecules.

    Returns a `(N, 3)` numpy array of 3d coords of given rdkit molecule

    Parameters
    ----------
    mol: rdkit Molecule
        Molecule to extract coordinates for

    Returns
    -------
    Numpy ndarray of shape `(N, 3)` where `N = mol.GetNumAtoms()`.
    """
    xyz = np.zeros((mol.GetNumAtoms(), 3))
    conf = mol.GetConformer()
    for i in range(conf.GetNumAtoms()):
        position = conf.GetAtomPosition(i)
        xyz[i, 0] = position.x
        xyz[i, 1] = position.y
        xyz[i, 2] = position.z
    return (xyz)


def add_hydrogens_to_mol(mol, is_protein=False):
    """
    Add hydrogens to a molecule object

    Parameters
    ----------
    mol: Rdkit Mol
        Molecule to hydrogenate
    is_protein: bool, optional (default False)
        Whether this molecule is a protein.


    Returns
    -------
    Rdkit Mol

    Note
    ----
    This function requires RDKit and PDBFixer to be installed.
    """
    return apply_pdbfixer(mol, hydrogenate=True, is_protein=is_protein)


def apply_pdbfixer(mol,
                   add_missing=True,
                   hydrogenate=True,
                   pH=7.4,
                   remove_heterogens=True,
                   is_protein=True):
    """
    Apply PDBFixer to a molecule to try to clean it up.

    Parameters
    ----------
    mol: Rdkit Mol
        Molecule to clean up.
    add_missing: bool, optional
        If true, add in missing residues and atoms
    hydrogenate: bool, optional
        If true, add hydrogens at specified pH
    pH: float, optional
        The pH at which hydrogens will be added if `hydrogenate==True`. Set to 7.4 by default.
    remove_heterogens: bool, optional
        Often times, PDB files come with extra waters and salts attached.
        If this field is set, remove these heterogens.
    is_protein: bool, optional
        If false, then don't remove heterogens (since this molecule is
        itself a heterogen).

    Returns
    -------
    Rdkit Mol

    Note
    ----
    This function requires RDKit and PDBFixer to be installed.
    """
    molecule_file = None
    try:
        from pdbfixer import PDBFixer
    except ModuleNotFoundError:
        raise ImportError("This function requires pdbfixer")

    try:
        import simtk
    except ModuleNotFoundError:
        raise ImportError("This function requires openmm")

    try:
        from rdkit import Chem
        pdbblock = Chem.MolToPDBBlock(mol)
        pdb_stringio = StringIO()
        pdb_stringio.write(pdbblock)
        pdb_stringio.seek(0)

        fixer = PDBFixer(pdbfile=pdb_stringio)
        if add_missing:
            fixer.findMissingResidues()
            fixer.findMissingAtoms()
            fixer.addMissingAtoms()
        if hydrogenate:
            fixer.addMissingHydrogens(pH)
        if is_protein and remove_heterogens:
            # False here specifies that water is to be removed
            fixer.removeHeterogens(False)

        hydrogenated_io = StringIO()
        simtk.openmm.app.PDBFile.writeFile(fixer.topology, fixer.positions,
                                           hydrogenated_io)
        hydrogenated_io.seek(0)
        return Chem.MolFromPDBBlock(hydrogenated_io.read(),
                                    sanitize=False,
                                    removeHs=False)
    except ValueError as e:
        logger.warning("Unable to add hydrogens %s", e)
        raise MoleculeLoadException(e)
    finally:
        try:
            os.remove(molecule_file)
        except (OSError, TypeError):
            pass


def compute_charges(mol):
    """Attempt to compute Gasteiger Charges on Mol

    This also has the side effect of calculating charges on mol.  The
    mol passed into this function has to already have been sanitized

    Parameters
    ----------
    mol: rdkit molecule

    Returns
    -------
    No return since updates in place.

    Note
    ----
    This function requires RDKit to be installed.
    """
    from rdkit.Chem import AllChem
    try:
        # Updates charges in place
        AllChem.ComputeGasteigerCharges(mol)
    except Exception as e:
        logging.exception("Unable to compute charges for mol")
        raise MoleculeLoadException(e)


def load_complex(molecular_complex: OneOrMany[str],
                 add_hydrogens: bool = True,
                 calc_charges: bool = True,
                 sanitize: bool = True) -> List[Tuple[np.ndarray, RDKitMol]]:
    """Loads a molecular complex.

    Given some representation of a molecular complex, returns a list of
    tuples, where each tuple contains (xyz coords, rdkit object) for
    that constituent molecule in the complex.

    For now, assumes that molecular_complex is a tuple of filenames.

    Parameters
    ----------
    molecular_complex: list or str
        If list, each entry should be a filename for a constituent
        molecule in complex. If str, should be the filename of a file that
        holds the full complex.
    add_hydrogens: bool, optional
        If true, add hydrogens via pdbfixer
    calc_charges: bool, optional
        If true, add charges via rdkit
    sanitize: bool, optional
        If true, sanitize molecules via rdkit

    Returns
    -------
    List of tuples (xyz, mol)

    Note
    ----
    This function requires RDKit to be installed.
    """
    if isinstance(molecular_complex, str):
        molecular_complex = [molecular_complex]
    fragments: List = []
    for mol in molecular_complex:
        loaded = load_molecule(mol,
                               add_hydrogens=add_hydrogens,
                               calc_charges=calc_charges,
                               sanitize=sanitize)
        if isinstance(loaded, list):
            fragments += loaded
        else:
            fragments.append(loaded)
    return fragments


def load_molecule(molecule_file,
                  add_hydrogens=True,
                  calc_charges=True,
                  sanitize=True,
                  is_protein=False):
    """Converts molecule file to (xyz-coords, obmol object)

    Given molecule_file, returns a tuple of xyz coords of molecule
    and an rdkit object representing that molecule in that order `(xyz,
    rdkit_mol)`. This ordering convention is used in the code in a few
    places.

    Parameters
    ----------
    molecule_file: str
        filename for molecule
    add_hydrogens: bool, optional (default True)
        If True, add hydrogens via pdbfixer
    calc_charges: bool, optional (default True)
        If True, add charges via rdkit
    sanitize: bool, optional (default False)
        If True, sanitize molecules via rdkit
    is_protein: bool, optional (default False)
        If True`, this molecule is loaded as a protein. This flag will
        affect some of the cleanup procedures applied.

    Returns
    -------
    Tuple (xyz, mol) if file contains single molecule. Else returns a
    list of the tuples for the separate molecules in this list.

    Note
    ----
    This function requires RDKit to be installed.
    """
    from rdkit import Chem
    from_pdb = False
    if ".mol2" in molecule_file:
        my_mol = Chem.MolFromMol2File(molecule_file,
                                      sanitize=False,
                                      removeHs=False)
    elif ".sdf" in molecule_file:
        suppl = Chem.SDMolSupplier(str(molecule_file), sanitize=False)
        # TODO: This is wrong. Should return all molecules
        my_mol = suppl[0]
    elif ".pdbqt" in molecule_file:
        pdb_block = pdbqt_to_pdb(molecule_file)
        my_mol = Chem.MolFromPDBBlock(str(pdb_block),
                                      sanitize=False,
                                      removeHs=False)
        from_pdb = True
    elif ".pdb" in molecule_file:
        my_mol = Chem.MolFromPDBFile(str(molecule_file),
                                     sanitize=False,
                                     removeHs=False)
        from_pdb = True  # noqa: F841
    else:
        raise ValueError("Unrecognized file type for %s" % str(molecule_file))

    if my_mol is None:
        raise ValueError("Unable to read non None Molecule Object")

    if add_hydrogens or calc_charges:
        my_mol = apply_pdbfixer(my_mol,
                                hydrogenate=add_hydrogens,
                                is_protein=is_protein)
    if sanitize:
        try:
            Chem.SanitizeMol(my_mol)
        # TODO: Ideally we should catch AtomValenceException but Travis seems to choke on it for some reason.
        except:
            logger.warning("Mol %s failed sanitization" %
                           Chem.MolToSmiles(my_mol))
    if calc_charges:
        # This updates in place
        compute_charges(my_mol)

    xyz = get_xyz_from_mol(my_mol)

    return xyz, my_mol


def write_molecule(mol, outfile, is_protein=False):
    """Write molecule to a file

    This function writes a representation of the provided molecule to
    the specified `outfile`. Doesn't return anything.

    Parameters
    ----------
    mol: rdkit Mol
        Molecule to write
    outfile: str
        Filename to write mol to
    is_protein: bool, optional
        Is this molecule a protein?

    Note
    ----
    This function requires RDKit to be installed.

    Raises
    ------
    ValueError: if `outfile` isn't of a supported format.
    """
    from rdkit import Chem
    if ".pdbqt" in outfile:
        writer = Chem.PDBWriter(outfile)
        writer.write(mol)
        writer.close()
        if is_protein:
            convert_protein_to_pdbqt(mol, outfile)
        else:
            convert_mol_to_pdbqt(mol, outfile)
    elif ".pdb" in outfile:
        writer = Chem.PDBWriter(outfile)
        writer.write(mol)
        writer.close()
    elif ".sdf" in outfile:
        writer = Chem.SDWriter(outfile)
        writer.write(mol)
        writer.close()
    else:
        raise ValueError("Unsupported Format")


def merge_molecules_xyz(xyzs):
    """Merges coordinates of multiple molecules.

    Parameters
    ----------
    xyzs: List
        List of numpy arrays each of shape `(N_i, 3)` where `N_i` is the number of atoms in the i-th atom.
    """
    return np.array(np.vstack(np.vstack(xyzs)))


def merge_molecules(molecules):
    """Helper method to merge two molecules.

    Parameters
    ----------
    molecules: list
        List of rdkit molecules

    Returns
    -------
    merged: rdkit molecule
    """
    from rdkit.Chem import rdmolops
    if len(molecules) == 0:
        return None
    elif len(molecules) == 1:
        return molecules[0]
    else:
        combined = molecules[0]
        for nextmol in molecules[1:]:
            combined = rdmolops.CombineMols(combined, nextmol)
        return combined


def compute_all_ecfp(mol: RDKitMol,
                     indices: Optional[Set[int]] = None,
                     degree: int = 2) -> Dict[int, str]:
    """Obtain molecular fragment for all atoms emanating outward to given degree.

    For each fragment, compute SMILES string (for now) and hash to
    an int. Return a dictionary mapping atom index to hashed
    SMILES.

    Parameters
    ----------
    mol: rdkit Molecule
        Molecule to compute ecfp fragments on
    indices: Optional[Set[int]]
        List of atom indices for molecule. Default is all indices. If
        specified will only compute fragments for specified atoms.
    degree: int
        Graph degree to use when computing ECFP fingerprints

    Returns
    ----------
    dict
        Dictionary mapping atom index to hashed smiles.
    """

    ecfp_dict = {}
    from rdkit import Chem
    for i in range(mol.GetNumAtoms()):
        if indices is not None and i not in indices:
            continue
        env = Chem.FindAtomEnvironmentOfRadiusN(mol, degree, i, useHs=True)
        submol = Chem.PathToSubmol(mol, env)
        smile = Chem.MolToSmiles(submol)
        ecfp_dict[i] = "%s,%s" % (mol.GetAtoms()[i].GetAtomicNum(), smile)

    return ecfp_dict


def compute_ecfp_features(mol, ecfp_degree=2, ecfp_power=11):
    """Computes ECFP features for provided rdkit molecule.
    Parameters
    ----------
    mol: rdkit molecule
        Molecule to featurize.
    ecfp_degree: int
        ECFP radius
    ecfp_power: int
        Number of bits to store ECFP features (2^ecfp_power will be length of
        ECFP array)
    Returns
    -------
    ecfp_array: np.ndarray
        Returns an array of size 2^ecfp_power where array at index i has a 1 if
        that ECFP fragment is found in the molecule and array at index j has a 0
        if ECFP fragment not in molecule.
    """
    from rdkit.Chem import AllChem
    bv = AllChem.GetMorganFingerprintAsBitVect(mol,
                                               ecfp_degree,
                                               nBits=2**ecfp_power)
    return np.array(bv)


def compute_contact_centroid(molecular_complex: Any,
                             cutoff: float = 4.5) -> np.ndarray:
    """Computes the (x,y,z) centroid of the contact regions of this molecular complex.

    For a molecular complex, it's necessary for various featurizations
    that compute voxel grids to find a reasonable center for the
    voxelization. This function computes the centroid of all the contact
    atoms, defined as an atom that's within `cutoff` Angstroms of an
    atom from a different molecule.

    Parameters
    ----------
    molecular_complex: Object
        A representation of a molecular complex, produced by
        `rdkit_util.load_complex`.
    cutoff: float, optional
        The distance in Angstroms considered for computing contacts.
    """
    fragments = reduce_molecular_complex_to_contacts(molecular_complex, cutoff)
    coords = [frag[0] for frag in fragments]
    contact_coords = merge_molecules_xyz(coords)
    centroid = np.mean(contact_coords, axis=0)
    return (centroid)


def reduce_molecular_complex_to_contacts(fragments: List,
                                         cutoff: float = 4.5) -> List:
    """Reduce a molecular complex to only those atoms near a contact.

    Molecular complexes can get very large. This can make it unwieldy to
    compute functions on them. To improve memory usage, it can be very
    useful to trim out atoms that aren't close to contact regions. This
    function takes in a molecular complex and returns a new molecular
    complex representation that contains only contact atoms. The contact
    atoms are computed by calling `get_contact_atom_indices` under the
    hood.

    Parameters
    ----------
    fragments: List
        As returned by `rdkit_util.load_complex`, a list of tuples of
        `(coords, mol)` where `coords` is a `(N_atoms, 3)` array and `mol`
        is the rdkit molecule object.
    cutoff: float
        The cutoff distance in angstroms.

    Returns
    -------
    A list of length `len(molecular_complex)`. Each entry in this list
    is a tuple of `(coords, MolecularShim)`. The coords is stripped down
    to `(N_contact_atoms, 3)` where `N_contact_atoms` is the number of
    contact atoms for this complex. `MolecularShim` is used since it's
    tricky to make a RDKit sub-molecule.
    """
    atoms_to_keep = get_contact_atom_indices(fragments, cutoff)
    reduced_complex = []
    for frag, keep in zip(fragments, atoms_to_keep):
        contact_frag = get_mol_subset(frag[0], frag[1], keep)
        reduced_complex.append(contact_frag)
    return reduced_complex


def compute_ring_center(mol, ring_indices):
    """Computes 3D coordinates of a center of a given ring.

    Parameters:
    -----------
    mol: rdkit.rdchem.Mol
        Molecule containing a ring
    ring_indices: array-like
        Indices of atoms forming a ring

    Returns:
    --------
    ring_centroid: np.ndarray
        Position of a ring center
    """
    conformer = mol.GetConformer()
    ring_xyz = np.zeros((len(ring_indices), 3))
    for i, atom_idx in enumerate(ring_indices):
        atom_position = conformer.GetAtomPosition(atom_idx)
        ring_xyz[i] = np.array(atom_position)
    ring_centroid = compute_centroid(ring_xyz)
    return ring_centroid


def get_contact_atom_indices(fragments: List, cutoff: float = 4.5) -> List:
    """Compute the atoms close to contact region.

    Molecular complexes can get very large. This can make it unwieldy to
    compute functions on them. To improve memory usage, it can be very
    useful to trim out atoms that aren't close to contact regions. This
    function computes pairwise distances between all pairs of molecules
    in the molecular complex. If an atom is within cutoff distance of
    any atom on another molecule in the complex, it is regarded as a
    contact atom. Otherwise it is trimmed.

    Parameters
    ----------
    fragments: List
        As returned by `rdkit_util.load_complex`, a list of tuples of
        `(coords, mol)` where `coords` is a `(N_atoms, 3)` array and `mol`
        is the rdkit molecule object.
    cutoff: float
        The cutoff distance in angstroms.

    Returns
    -------
    A list of length `len(molecular_complex)`. Each entry in this list
    is a list of atom indices from that molecule which should be kept, in
    sorted order.
    """
    # indices of atoms to keep
    keep_inds: List[Set] = [set([]) for _ in fragments]
    for (ind1, ind2) in itertools.combinations(range(len(fragments)), 2):
        frag1, frag2 = fragments[ind1], fragments[ind2]
        pairwise_distances = compute_pairwise_distances(frag1[0], frag2[0])
        # contacts is of form (x_coords, y_coords), a tuple of 2 lists
        contacts = np.nonzero((pairwise_distances < cutoff))
        # contacts[0] is the x_coords, that is the frag1 atoms that have
        # nonzero contact.
        frag1_atoms = set([int(c) for c in contacts[0].tolist()])
        # contacts[1] is the y_coords, the frag2 atoms with nonzero contacts
        frag2_atoms = set([int(c) for c in contacts[1].tolist()])
        keep_inds[ind1] = keep_inds[ind1].union(frag1_atoms)
        keep_inds[ind2] = keep_inds[ind2].union(frag2_atoms)
    keep_ind_lists = [sorted(list(keep)) for keep in keep_inds]
    return keep_ind_lists


def get_mol_subset(coords, mol, atom_indices_to_keep):
    """Strip a subset of the atoms in this molecule

    Parameters
    ----------
    coords: Numpy ndarray
        Must be of shape (N, 3) and correspond to coordinates of mol.
    mol: Rdkit mol or `MolecularFragment`
        The molecule to strip
    atom_indices_to_keep: list
        List of the indices of the atoms to keep. Each index is a unique
        number between `[0, N)`.

    Returns
    -------
    A tuple of (coords, mol_frag) where coords is a Numpy array of
    coordinates with hydrogen coordinates. mol_frag is a
    `MolecularFragment`.
    """
    from rdkit import Chem
    indexes_to_keep = []
    atoms_to_keep = []
    #####################################################
    # Compute partial charges on molecule if rdkit
    if isinstance(mol, Chem.Mol):
        compute_charges(mol)
    #####################################################
    atoms = list(mol.GetAtoms())
    for index in atom_indices_to_keep:
        indexes_to_keep.append(index)
        atoms_to_keep.append(atoms[index])
    coords = coords[indexes_to_keep]
    mol_frag = MolecularFragment(atoms_to_keep, coords)
    return coords, mol_frag


def compute_ring_normal(mol, ring_indices):
    """Computes normal to a plane determined by a given ring.

    Parameters:
    -----------
    mol: rdkit.rdchem.Mol
        Molecule containing a ring
    ring_indices: array-like
        Indices of atoms forming a ring

    Returns:
    --------
    normal: np.ndarray
        Normal vector
    """
    conformer = mol.GetConformer()
    points = np.zeros((3, 3))
    for i, atom_idx in enumerate(ring_indices[:3]):
        atom_position = conformer.GetAtomPosition(atom_idx)
        points[i] = np.array(atom_position)

    v1 = points[1] - points[0]
    v2 = points[2] - points[0]
    normal = np.cross(v1, v2)
    return normal


def compute_all_pairs_shortest_path(
        mol) -> Dict[Tuple[int, int], Tuple[int, int]]:
    """Computes the All pair shortest between every pair of nodes
    in terms of Rdkit Atom indexes.

    Parameters:
    -----------
    mol: rdkit.rdchem.Mol
        Molecule containing a ring

    Returns:
    --------
    paths_dict: Dict representing every atom-atom pair as key in Rdkit index
    and value as the shortest path between each atom pair in terms of Atom index.
    """
    try:
        from rdkit import Chem
    except:
        raise ImportError("This class requires RDkit installed")
    n_atoms = mol.GetNumAtoms()
    paths_dict = {(i, j): Chem.rdmolops.GetShortestPath(mol, i, j)
                  for i in range(n_atoms) for j in range(n_atoms) if i < j}
    return paths_dict


def compute_pairwise_ring_info(mol):
    """ Computes all atom-atom pair belong to same ring with
    its ring size and its aromaticity.

    Parameters:
    -----------
    mol: rdkit.rdchem.Mol
        Molecule containing a ring

    Returns:
    --------
    rings_dict: Key consisting of all node-node pair sharing the same ring
    and value as a tuple of size of ring and its aromaticity.
    """
    try:
        from rdkit import Chem
    except:
        raise ImportError("This class requires RDkit installed")
    rings_dict = {}

    def ordered_pair(a, b):
        return (a, b) if a < b else (b, a)

    ssr = [list(x) for x in Chem.GetSymmSSSR(mol)]
    for ring in ssr:
        ring_sz = len(ring)
        is_aromatic = True
        for atom_idx in ring:
            if not mol.GetAtoms()[atom_idx].GetIsAromatic():
                is_aromatic = False
                break
        for ring_idx, atom_idx in enumerate(ring):
            for other_idx in ring[ring_idx:]:
                atom_pair = ordered_pair(atom_idx, other_idx)
                if atom_pair not in rings_dict:
                    rings_dict[atom_pair] = [(ring_sz, is_aromatic)]
                else:
                    if (ring_sz, is_aromatic) not in rings_dict[atom_pair]:
                        rings_dict[atom_pair].append((ring_sz, is_aromatic))

    return rings_dict


class DescriptorsNormalizationParameters:
    """
    A class for holding a dictionary of 200 RDKit descriptors and the corresponding distribution functions (based on `scipy.stats` module nomenclature)
    and associated parameters required for creating normalized descriptor functions.

    (The distribution functions and parameters data was collected from the source code of 'descriptastorus' library.
    Link to the source of the parameters:
    https://github.com/bp-kelley/descriptastorus/blob/baed6a56d63dd27c1bc5f6924a7c9e0d48594012/descriptastorus/descriptors/dists.py)
    """
    try:
        from math import inf
    except ImportError:
        inf = float('Inf')

    desc_norm_params: Dict[str, Sequence[Union[str, Sequence[float], float]]]

    desc_norm_params = {
        'VSA_EState1': ('betaprime', (0.12979362790686721, 2.0084510281921832,
                                      -1.7874327445880742e-34,
                                      0.4022005514286884), 0.0, 0.0, 0.0, 0.0),
        'Chi4n': ('mielke', (3.4334547302721328, 4.644325894559659,
                             -0.1540765028544061, 3.7724626101152783), 0.0,
                  60.15570624032009, 3.5583748823804937, 1.9340011133496022),
        'EState_VSA3':
            ('pearson3', (1.2130333692507862, 20.490424246483514,
                          14.913598628728794), 0.0, 707.4193712196022,
             20.490458921014422, 15.02692979610578),
        'PEOE_VSA3':
            ('recipinvgauss', (2039990.2035692804, -1.514157558116536e-12,
                               5.862765691466683), 0.0, 278.49556054006456,
             5.883620301814973, 7.114346140218968),
        'PEOE_VSA10':
            ('ncx2', (1.2634981555275662, 2.1503143438355354,
                      -2.21123444897305e-31, 2.606409115395213), 0.0,
             494.0556831191233, 9.763622525936078, 12.915305068064065),
        'Chi2v': ('fisk', (5.416294252795936, -0.46711733318914683,
                           7.911730855234288), 0.0, 152.03415385974233,
                  7.97051218652611, 4.006279197425447),
        'SMR_VSA8': ('betaprime', (0.12979362790686721, 2.0084510281921832,
                                   -1.7874327445880742e-34, 0.4022005514286884),
                     0.0, 0.0, 0.0, 0.0),
        'ExactMolWt':
            ('mielke', (6.030507225812184, 6.081069808326847,
                        -3.1905715544779594, 393.79789827541134), 7.01545597009,
             7902.703267132, 413.2180535712111, 196.11660291127603),
        'fr_imidazole': ('wald', (-0.017711130586117518, 0.05908774988990952),
                         0, 11, 0.10202714189993299, 0.35905461339251266),
        'fr_aldehyde':
            ('halflogistic', (-2.2802084549638172e-10, 0.003260151958473212), 0,
             2, 0.0032602282159751184, 0.057529151986179125),
        'fr_Al_COO': ('beta', (0.6951891478660377, 401.3878921392054,
                               -1.8490162599417683e-28, 14.902552079575546), 0,
                      9, 0.05697398817917254, 0.2720833952111172),
        'NumAliphaticHeterocycles':
            ('alpha', (5.571361455216543e-09, -0.07477286399593108,
                       0.10965986221560856), 0, 22, 0.7546628263978479,
             0.9186038062617755),
        'fr_Ar_NH': ('wald', (-0.01991313112984635, 0.06651081784403591), 0, 13,
                     0.11327792945506185, 0.37297284960554855),
        'NumHAcceptors': ('logistic', (5.039523695264815, 1.2773064178194637),
                          0, 199, 5.285449981498705, 3.9293707904214674),
        'fr_lactam':
            ('halflogistic', (-1.9994772259051099e-10, 0.0019599976691355514),
             0, 2, 0.0019601372096046724, 0.04490322117641692),
        'fr_NH2': ('wald', (-0.025886531735103906, 0.08666379088388962), 0, 17,
                   0.14403008210574741, 0.5080894197040707),
        'fr_Ndealkylation1':
            ('wald', (-0.014170257884871005, 0.047146436951077536), 0, 8,
             0.08431590211314792, 0.3528598050157884),
        'SlogP_VSA7':
            ('recipinvgauss', (124626.03395996531, -5.039104162427062e-11,
                               0.4157942168226829), 0.0, 77.05538951387919,
             1.1814079834441082, 2.9446812598365155),
        'fr_Ar_N': ('halfgennorm', (0.32377704425495, -8.49433587779278e-22,
                                    0.010554705354503424), 0, 66,
                    1.3546848279379557, 1.7700301662397686),
        'NumSaturatedHeterocycles':
            ('halfgennorm', (0.39340463716320073, -3.084884622335595e-23,
                             0.007755844850523552), 0, 22, 0.545018151270589,
             0.8423690871097294),
        'NumAliphaticRings':
            ('gennorm', (0.16214999420806342, 1.0000000000000002,
                         1.2644170558638866e-06), 0, 24, 0.9787585130959167,
             1.0878564569993276),
        'SMR_VSA4':
            ('betaprime', (0.8177035716382387, 2.026000293708502,
                           -2.7076233813444817e-29, 0.8955328046106019), 0.0,
             169.22417394169068, 3.521664764716257, 6.367910012270387),
        'Chi0v': ('mielke', (5.8775750667785065, 5.969290153282742,
                             -0.051645183224216795, 16.255522569236142), 1.0,
                  310.3665679014367, 17.129538294841, 7.923870846730872),
        'qed': ('johnsonsb', (-0.537683817552717, 0.9438392221113977,
                              -0.05971660981816428, 1.0270014571751256),
                0.001610010104943233, 0.9484019712261345, 0.5707778636205341,
                0.21314724659491038),
        'fr_sulfonamd':
            ('betaprime', (0.6061868535729906, 2.47005272151398,
                           -1.7109734983680305e-30, 0.024136054923030247), 0, 2,
             0.09929695078655505, 0.3106727570704293),
        'fr_halogen':
            ('exponweib', (1.5936220372251417, 0.4773265592552294,
                           -6.305557196427465e-30, 0.11136163589207024), 0, 22,
             0.6656565959617173, 1.1538784657180654),
        'Chi4v': ('mielke', (3.641407704651825, 4.9160753250874905,
                             -0.19612721404766648, 4.272311768092421), 0.0,
                  80.31016831275534, 3.997197941292901, 2.1822145921791463),
        'MolLogP':
            ('nct', (5.423133497140618, -0.2505422147848311, 3.787125066469563,
                     1.447521060093181), -27.121900000000043,
             26.476990000000036, 3.357664737331615, 1.8518910248841818),
        'Chi2n': ('burr', (5.323167832131418, 0.9614449953883716,
                           -0.5182229173193543, 7.403200388112394), 0.0,
                  140.4224584835206, 7.320378282785918, 3.6830713407241156),
        'fr_Al_OH': ('pareto', (8.075644989366163, -0.6961711017351564,
                                0.6961711017065046), 0, 37, 0.18985328973028112,
                     0.5806908433990465),
        'LabuteASA': ('mielke', (5.9033344150609555, 6.093254767597408,
                                 -1.1647561264561102, 165.59359494140412),
                      19.375022664857827, 3138.810989711936, 172.78618537968595,
                      78.72241596326842),
        'SMR_VSA5': ('johnsonsu', (-6.770383826447828, 1.5639816052567266,
                                   -8.885737844117894, 0.8747218279195421), 0.0,
                     1059.7000355910607, 31.92088861776502, 31.701702832660054),
        'fr_guanido':
            ('halflogistic', (-2.8948632876909134e-11, 0.012390710083318518), 0,
             8, 0.012390867360715251, 0.1352692056473331),
        'SlogP_VSA6':
            ('dweibull', (1.2651196828193871, 44.8855208417171,
                          19.999416617652344), 0.0, 425.50532391912986,
             46.86150277566015, 23.994033852363053),
        'NumRadicalElectrons':
            ('halfnorm', (-3.3556536839857015e-09, 0.047012245813331466), 0, 10,
             0.0002900203014210995, 0.047011387972006546),
        'HeavyAtomCount': ('mielke', (5.542942710744559, 6.0129203920305345,
                                      -0.10475651052005365, 28.19145327714555),
                           1, 545, 29.187713139919794, 13.728021647131865),
        'fr_Ar_COO': ('pearson3', (2.284860216852766, 0.009533409939207087,
                                   0.010891254550521512), 0, 4,
                      0.019321352494674628, 0.14278703946614923),
        'fr_ester': ('wald', (-0.021345006038942453, 0.07132669030690274), 0, 9,
                     0.12089846289240247, 0.3816605577678055),
        'NumSaturatedCarbocycles':
            ('invweibull', (0.6897602654592729, -1.687045423250026e-28,
                            0.04100428545396396), 0, 9, 0.16625163761463302,
             0.5048961994063688),
        'MolMR': ('burr', (6.172170768729716, 0.8936060537538131,
                           -0.6260689145704982, 109.36170666360255), 0.0,
                  1943.4740000000081, 111.75295175062253, 49.21222833407792),
        'fr_SH':
            ('halflogistic', (-5.627308315330754e-10, 0.002940168145955494), 0,
             2, 0.0029402058144070084, 0.057719918625515),
        'fr_ketone_Topliss':
            ('invweibull', (0.7845359779027771, -5.104827196981351e-29,
                            0.013651958761797665), 0, 5, 0.052113647955356876,
             0.24762602874204098),
        'MolWt': ('burr', (6.292103975443265, 0.9448356544126613,
                           -2.576132139748636, 398.4147809834958), 6.941,
                  7906.685999999983, 413.6754726605188, 196.24518387547647),
        'Kappa1': ('fisk', (5.090249068250488, 1.5243046476343975,
                            17.457634086393984), 1.5974025974025972,
                   452.4780879471051, 20.511369181587643, 10.901150147628785),
        'fr_term_acetylene':
            ('tukeylambda', (1.56196184906287, 0.005212203460720155,
                             0.008141262955574484), 0, 2, 0.004260298220875461,
             0.06634877000162641),
        'Chi0n': ('mielke', (4.913702328256858, 5.56535100301131,
                             0.892757720674525, 15.151908382349568), 1.0,
                  310.3665679014367, 16.565702068330904, 7.789694974456289),
        'SMR_VSA9':
            ('betaprime', (0.6371524412800221, 0.1653001011937434,
                           -1.1096304834919027e-26, 0.008200829448013355), 0.0,
             68.99414199940686, 6.711933474546377, 7.898303955057197),
        'fr_hdrzine': ('genexpon', (2.2033965201556205, 4.581950155773536e-11,
                                    1.9344067808226306, -6.071473724440753e-12,
                                    0.021418241524934097), 0, 3,
                       0.009720680447631334, 0.10023082470639716),
        'PEOE_VSA11':
            ('betaprime', (0.5343466300982029, 2.0391606100747115,
                           -5.644268412961732e-28, 1.3724974434068684), 0.0,
             130.7949739456163, 3.9163487225567097, 6.085123241223355),
        'PEOE_VSA2': ('genlogistic', (1025.7536859090253, -32.00326592275044,
                                      5.426751909719114), 0.0,
                      436.5016359224807, 8.968372740727396, 10.688899230076526),
        'fr_C_O': ('dweibull', (0.799241338494272, 1.0000000000000002,
                                1.1376361682177123), 0, 79, 1.3462142349964497,
                   1.7126177063621493),
        'EState_VSA2': ('dgamma', (1.6208597357735999, 14.722163518402503,
                                   5.927601342234055), 0.0, 348.2229478088118,
                        16.17693847024966, 14.300245340171864),
        'fr_aryl_methyl': ('pareto', (2.2590951540541755, -0.03484369022048178,
                                      0.03484369021605811), 0, 10,
                           0.42467972758093064, 0.7334626175897146),
        'EState_VSA9':
            ('gompertz', (157762717330.9355, -8.297980815964564e-12,
                          1740272081706.4692), 0.0, 459.53012783207976,
             9.405433826279932, 11.29251583636432),
        'SlogP_VSA1':
            ('betaprime', (24.555424043054373, 8.676784435768978,
                           -8.426380550481028, 5.728625412480383), 0.0,
             409.4953592555066, 9.984737467652149, 9.779830593415184),
        'PEOE_VSA9': ('betaprime', (3.445175492809242, 31.144739014821592,
                                    -4.51163967848681, 170.01655826137412), 0.0,
                      811.1436563731141, 14.95011048070105, 13.041254445054955),
        'SMR_VSA2': ('wald', (-0.048767578220460746, 0.16186950174913545), 0.0,
                     43.27426884633218, 0.29905148573740525, 1.360442550951591),
        'fr_quatN':
            ('halflogistic', (-1.8398640615463982e-10, 0.0022802512521409345),
             0, 2, 0.002280159611172782, 0.05094097451844112),
        'fr_dihydropyridine':
            ('tukeylambda', (1.5508163529247363, 0.002932569691259215,
                             0.004547877033296321), 0, 1, 0.002350164511515806,
             0.04842149562213685),
        'MinPartialCharge':
            ('johnsonsu', (-2.7149539980911284, 1.0038367476615098,
                           -0.5139064061681397, 0.008495052829614172),
             -0.7539104058810929, 1.0, -0.4200780304196831, 0.07189165509088434
            ),
        'fr_ketone': ('wald', (-0.01054998963908324, 0.035036217635414556), 0,
                      6, 0.06421449501465103, 0.27790680727297107),
        'MaxAbsEStateIndex':
            ('t', (1.3108542430395254, 12.75352190648762, 0.7106673064513349),
             0.0, 18.093289294329608, 12.009719840202807, 2.3499823939671187),
        'MaxAbsPartialCharge':
            ('johnsonsu', (1.7917111012287341, 0.9884411996798824,
                           0.509455736564852, 0.019080208112214614),
             0.044672166080629815, 3.0, 0.4246995864209025, 0.07158969964653725
            ),
        'Chi1v': ('burr', (7.687954133980298, 1.4497345379886477,
                           -5.291458017141911, 14.007066785153341), 0.0,
                  193.52184470749225, 10.233294455235377, 4.9372866377359905),
        'fr_benzodiazepine':
            ('tukeylambda', (1.5687789926087756, 0.0024158367934644202,
                             0.0037899140207594472), 0, 1,
             0.0019601372096046724, 0.04423002455034584),
        'EState_VSA5':
            ('exponweib', (0.2671814730479748, 1.165636178704992,
                           -1.382245994068274e-31, 22.76769085633479), 0.0,
             352.5201628431787, 13.722297283129771, 14.369271762415691),
        'VSA_EState7':
            ('loggamma', (0.00016582407490511742, 4.690543031567471e-07,
                          5.38971779436177e-08), -0.23935820315780676, 0.0,
             -2.393749594049651e-06, 0.0007569398071209723),
        'fr_C_O_noCOO':
            ('gennorm', (0.3061307987512025, 1.0, 0.0046024831025323395), 0, 71,
             1.2699788985228966, 1.6247465287374963),
        'Chi3v': ('mielke', (4.392997381604184, 5.27028468604366,
                             -0.32791696488665867, 5.890409366481655), 0.0,
                  106.26777041156456, 5.646449010813653, 2.881905780196624),
        'PEOE_VSA5':
            ('gibrat', (-0.04993824025443838, 0.13664006325267003), 0.0,
             73.11939707590946, 2.0398397458425537, 4.669463461453408),
        'fr_epoxide':
            ('hypsecant', (9.775361773901804e-06, 0.004229322664644675), 0, 3,
             0.0015001050073505146, 0.041205202260121296),
        'fr_prisulfonamd':
            ('betaprime', (0.12979362790686721, 2.0084510281921832,
                           -1.7874327445880742e-34, 0.4022005514286884), 0, 0,
             0.0, 0.0),
        'fr_phenol': ('invweibull', (0.8135991555193818, -4.806156168974463e-28,
                                     0.016229148422267782), 0, 8,
                      0.05026351844629124, 0.2534132017880052),
        'fr_sulfide':
            ('gengamma', (1.5106281082392625, 0.5574261640341269,
                          -4.4413855396989186e-30, 0.0038351572717300027), 0, 6,
             0.08101567109697679, 0.2882927997299278),
        'fr_alkyl_halide': ('wald', (-0.04358050576953841, 0.1452321478197392),
                            0, 17, 0.25484783934875443, 0.9066667900753467),
        'NumAromaticHeterocycles':
            ('halfgennorm', (0.19057145504745865, -1.897689882032624e-17,
                             2.1261316374019246e-05), 0, 33, 0.9458862120348425,
             1.0262824322901387),
        'fr_Ar_OH': ('wald', (-0.0100135027386673, 0.03324263418577556), 0, 8,
                     0.06124428710009701, 0.2766130074506664),
        'fr_thiazole': ('wald', (-0.007257531762340398, 0.024068128495977677),
                        0, 6, 0.04502315162061344, 0.217339283746722),
        'fr_imide': ('pearson3', (2.2221583002327714, 0.018063449381832734,
                                  0.02006992198733707), 0, 6,
                     0.02752192653485744, 0.1778336591269578),
        'NumSaturatedRings':
            ('halfgennorm', (0.23246838885007082, -2.4267394888596534e-25,
                             0.00026458005932038795), 0, 22, 0.711269788885222,
             0.9971159900281824),
        'fr_hdrzone': ('wald', (-0.003105521778228028, 0.010274411296672493), 0,
                       2, 0.019951396597761843, 0.14280545219170362),
        'fr_lactone': ('tukeylambda', (1.527151454037369, 0.013585748455595358,
                                       0.020747495508148833), 0, 6,
                       0.011000770053903774, 0.11242720454031359),
        'FractionCSP3':
            ('gausshyper', (0.4771522405083861, 9.066071275571563,
                            -7.620494949081857, 3.1818106084013347,
                            -4.691817003086665e-28, 1.3215636376781599), 0.0,
             1.0, 0.3432847243844843, 0.19586440800503926),
        'HallKierAlpha': ('logistic', (-2.756871651517898, 0.6249482292466357),
                          -56.5600000000003, 3.0300000000000002,
                          -2.8236952794487813, 1.362245033459619),
        'fr_para_hydroxylation':
            ('gibrat', (-0.00874073035192809, 0.023988673379858362), 0, 7,
             0.2581980738651706, 0.5588920168597382),
        'HeavyAtomMolWt':
            ('burr', (6.148243331396738, 0.9539656352826273,
                      -2.0597306351675164, 373.86535994513247), 6.941,
             7542.798000000015, 389.1014159096795, 184.23918925686846),
        'SlogP_VSA12': ('lomax', (1.3526140858007119, -3.8436281815982077e-13,
                                  0.5306720480047502), 0.0, 199.36606648479187,
                        6.222469001946574, 9.773336061271038),
        'fr_allylic_oxid':
            ('wald', (-0.012125985576272109, 0.04021754130884261), 0, 12,
             0.07526526856879981, 0.43369194377652315),
        'fr_alkyl_carbamate':
            ('wald', (-0.0028043421468934076, 0.009276348457582401), 0, 3,
             0.018071264988549197, 0.13836470784532925),
        'fr_HOCCN': ('wald', (-0.0019826051357043533, 0.006555328648504729), 0,
                     2, 0.01286090026301841, 0.11373440248499654),
        'Chi1n': ('mielke', (4.960260518842145, 5.594233665274178,
                             -0.0629921400622475, 9.415745103559818), 0.0,
                  179.81742433151658, 9.668289027429788, 4.636496265706159),
        'PEOE_VSA4': ('pareto', (1.764490302170505, -1.0130190528730951,
                                 1.0130190526084408), 0.0, 74.63705581047195,
                      2.871044297114316, 5.252580084878072),
        'NOCount': ('dgamma', (0.9114891050741081, 5.999999999999998,
                               2.360844051557576), 0, 237, 6.576580360625244,
                    5.165410541429347),
        'EState_VSA4':
            ('foldnorm', (0.005911506485819916, -9.396020384268004e-10,
                          29.86848626045161), 0.0, 309.6802047001077,
             23.73312663187225, 18.134786070135654),
        'VSA_EState6': ('betaprime', (0.12979362790686721, 2.0084510281921832,
                                      -1.7874327445880742e-34,
                                      0.4022005514286884), 0.0, 0.0, 0.0, 0.0),
        'Chi3n': ('mielke', (3.8419002158163336, 5.021067627998188,
                             -0.12843494993055915, 5.26846058565802), 0.0,
                  89.41912553532535, 5.1024891606395135, 2.6233324773423217),
        'fr_barbitur':
            ('genhalflogistic', (0.0020825723750872178, -0.0014169372235491489,
                                 0.005286684885304822), 0, 2,
             0.0014100987069094837, 0.03805408163051986),
        'fr_Al_OH_noTert':
            ('gompertz', (35118115.876645096, -1.5683266541399722e-10,
                          5712177.117752606), 0, 37, 0.16987189103237227,
             0.5550559671673745),
        'fr_COO2': ('wald', (-0.012755896711276589, 0.04241946965978133), 0, 9,
                    0.07634534417409218, 0.3067871977652909),
        'fr_azo':
            ('genhalflogistic', (0.00040002675029834824, -2.796838513658869e-05,
                                 0.0031033294350019196), 0, 2,
             0.0006700469032832298, 0.027011148452530685),
        'FpDensityMorgan1':
            ('t', (7.1357369753127795, 1.0971430458813116,
                   0.17213544486837606), 0.07971014492753623, 2.111111111111111,
             1.0947677251482226, 0.20201147073315795),
        'fr_aniline':
            ('halfgennorm', (0.1049174054668825, -0.0004210957634456598,
                             2.0561338188621596e-11), 0, 17, 0.6701569109837688,
             0.8999932508248457),
        'SMR_VSA3': ('dgamma', (1.7525536696583415, 12.349493875194277,
                                4.09734805299208), 0.0, 394.2610115631847,
                     12.654567199384484, 11.238213406272283),
        'fr_tetrazole': ('wald', (-0.0018040357049599466, 0.005964316702018277),
                         0, 2, 0.011720820457432021, 0.1080900079828228),
        'VSA_EState10':
            ('gennorm', (0.325862160652744, -1.5857598255070421e-27,
                         0.009249443355545232), -22.789260764991006,
             74.77274927572002, 1.3375950132085297, 3.8197200206817126),
        'fr_phenol_noOrthoHbond':
            ('invweibull', (0.8136034301945807, -5.137954972709019e-29,
                            0.01610798208070398), 0, 8, 0.049623473643155024,
             0.2514798405151786),
        'PEOE_VSA8': ('dgamma', (1.49159666908677, 21.468624511796396,
                                 8.385920333308711), 0.0, 311.7903524102208,
                      25.24827589064641, 16.37842558375825),
        'EState_VSA8': ('genexpon', (0.8492139229831794, 0.898456899940423,
                                     1.8794539462657145, -7.998786049300654e-10,
                                     28.40401980183114), 0.0, 364.8615531780818,
                        21.03371597445418, 18.393077792682497),
        'BalabanJ':
            ('nct', (4.182658638749994, 2.0965263482828114, 1.1271767054584343,
                     0.2616489125636474), 0.0, 7.289359191119452,
             1.8039183288289355, 0.47846986656304485),
        'fr_C_S': ('tukeylambda', (1.373229786243365, 0.006142154117625993,
                                   0.008434588986021351), 0, 2,
                   0.017351214585020952, 0.13360466287284203),
        'fr_ArN': ('tukeylambda', (1.4424188577152934, 0.03329139576768536,
                                   0.04802013705497249), 0, 16,
                   0.05239366755672897, 0.33593599906289534),
        'NumAromaticRings': ('dgamma', (2.5851733541458035, 2.495448308226644,
                                        0.3726216029742541), 0, 34,
                             2.480923664656526, 1.2619073213233434),
        'fr_Imine': ('wald', (-0.004133581140781835, 0.013682691065125839), 0,
                     4, 0.02636184532917304, 0.17166002580284992),
        'NumAliphaticCarbocycles':
            ('halfgennorm', (0.39096958306392793, -4.260236992450893e-24,
                             0.0012095685496975065), 0, 9, 0.22409568669806887,
             0.5847294862795492),
        'fr_piperzine':
            ('invweibull', (0.7886440189451933, -2.4256731163377907e-29,
                            0.02564081626730841), 0, 5, 0.08231576210334723,
             0.28309044391366156),
        'fr_nitroso': ('genhalflogistic',
                       (2.1084981882574176e-07, -4.9280506479729835e-05,
                        0.0027110708397517554), 0, 2, 0.00014000980068604803,
                       0.012648778519674015),
        'FpDensityMorgan2':
            ('johnsonsu', (0.7111118816857295, 1.7569762969412164,
                           1.9820508520621796, 0.3566466873696017),
             0.13043478260869565, 2.75, 1.807249299497176, 0.26404333001263824),
        'SlogP_VSA3':
            ('genhalflogistic', (0.00150187910000416, -2.3112405176891454e-10,
                                 9.57718508471936), 0.0, 486.412075845355,
             13.207558092806261, 14.07198140237455),
        'fr_urea': ('wald', (-0.011156011934882477, 0.03708223885227959), 0, 4,
                    0.06708469592871501, 0.2580011979049133),
        'VSA_EState9':
            ('t', (2.778511535506783, 54.00468044177123,
                   14.841602340950482), -61.35386460040703, 1513.3333333333328,
             58.298150276013196, 32.97223516741629),
        'fr_nitro_arom':
            ('exponweib', (1.1347393331388103, 0.7642415443821742,
                           -9.540668167412522e-31, 0.00766605879556546), 0, 4,
             0.03152220655445881, 0.18587317845266496),
        'fr_amidine': ('gompertz', (1256395099718.369, -3.6322380930315925e-14,
                                    17278550801.355766), 0, 4,
                       0.0163211424799736, 0.13976768727132605),
        'fr_nitro_arom_nonortho':
            ('wald', (-0.002927735757054801, 0.009685114750495933), 0, 3,
             0.01884131889232246, 0.14002286241970868),
        'SlogP_VSA11':
            ('invweibull', (0.4147001922244138, -1.4325296088889312e-27,
                            0.4348351110420754), 0.0, 68.99414199940686,
             3.2168374575889174, 5.0279560653331945),
        'RingCount': ('dgamma', (2.186557671264268, 3.4836909786085286,
                                 0.49776105041563334), 0, 57,
                      3.4596821777524425, 1.5775840075847496),
        'fr_azide': ('hypsecant', (0.0011100250221481181, 0.002817992740312428),
                     0, 2, 0.0009900693048513396, 0.03238970929040505),
        'Ipc': ('ncf', (2.2307091334463722, 0.10899090116091759,
                        1.0000000000366653, -1.1981582241278056e+221,
                        0.5596425180505273), 0.0, 9.476131257211451e+221,
                1.0245870093128289e+217, inf),
        'fr_benzene': ('dgamma', (3.3192814002631734, 1.5061477493525466,
                                  0.23910537147076447), 0, 14,
                       1.5344274099186943, 0.9561701389956593),
        'fr_thiocyan':
            ('gengamma', (0.5076950581215369, 1.2254477892001665,
                          -1.4543304035389683e-31, 0.00801926074913513), 0, 2,
             0.0002900203014210995, 0.017605044440269665),
        'PEOE_VSA14': ('pareto', (1.7596722287597166, -2.239860392602096,
                                  2.239860392451776), 0.0, 416.4980987917504,
                       3.013749589704376, 6.883189641607028),
        'PEOE_VSA7': ('dgamma', (1.4806417259357412, 39.98160154845938,
                                 10.168406696469939), 0.0, 508.39601533563194,
                      41.78662552891026, 19.81428403790816),
        'VSA_EState5':
            ('genhalflogistic', (4.73389199695264e-09, -0.030429032584450635,
                                 0.09155583517487542), 0.0, 68.19071853741498,
             0.008147278492018148, 0.5030871156371451),
        'EState_VSA7':
            ('powerlaw', (0.2103250937511124, -1.031145487494383e-26,
                          231.78083977397324), 0.0, 225.3129516643085,
             25.741886434607192, 21.953593999938157),
        'fr_N_O': ('exponnorm', (5045.039118637744, -2.4181865220268798e-05,
                                 5.7188007391269775e-06), 0, 6,
                   0.028882021741521907, 0.23864172665999558),
        'VSA_EState4': ('betaprime', (0.12979362790686721, 2.0084510281921832,
                                      -1.7874327445880742e-34,
                                      0.4022005514286884), 0.0, 0.0, 0.0, 0.0),
        'EState_VSA6': ('chi', (0.5386485681683548, -2.640567062882436e-29,
                                30.844792175705116), 0.0, 298.3055333422921,
                        18.147758868775043, 16.4171251936251),
        'PEOE_VSA6': ('exponpow', (0.9016211226446249, -5.748343789415991e-27,
                                   53.824180129747546), 0.0, 415.6833365808712,
                      30.51404957069927, 23.580292452344),
        'fr_diazo':
            ('halfnorm', (-8.299739916759084e-10, 0.0031623214263548586), 0, 1,
             1.000070004900343e-05, 0.0031623725326093326),
        'MaxEStateIndex':
            ('t', (1.3108542430395254, 12.753521906487567, 0.7106673064513349),
             0.0, 18.093289294329608, 12.009719840202806, 2.3499823939671187),
        'fr_oxime': ('pearson3', (2.517850854435552, 0.009251874821651483,
                                  0.011647420462412978), 0, 3,
                     0.015401078075465282, 0.1267436305065821),
        'SlogP_VSA10':
            ('betaprime', (0.4375490925573042, 1.8760340999346696,
                           -1.050475196211794e-28, 1.367988164331459), 0.0,
             99.49189265982736, 6.345391750680324, 7.373907457789322),
        'fr_nitrile': ('invweibull', (0.813385754265167, -2.609225753111669e-30,
                                      0.010587145126146093), 0, 5,
                       0.04477313411938836, 0.2211084100854278),
        'fr_COO': ('wald', (-0.01274707189437067, 0.04238860957269042), 0, 9,
                   0.07629534067384718, 0.306587684326303),
        'VSA_EState8': ('cauchy', (0.4615208647039756, 1.4176964379023667),
                        -4.311579244789392, 610.9052623831149,
                        9.971522100585924, 18.488170207714543),
        'SlogP_VSA2': ('lognorm', (0.5117956692971417, -7.272585382702083,
                                   42.859086084122595), 0.0, 1181.0895715112954,
                       41.959432071647804, 34.15538345697471),
        'fr_priamide':
            ('exponweib', (1.0954611988211496, 0.7544974723445165,
                           -8.879663989913682e-32, 0.012058631113746431), 0, 6,
             0.03770263918474293, 0.21274815822293996),
        'SMR_VSA1': ('cauchy', (13.199049915613251, 5.44099244703784), 0.0,
                     611.7284512306612, 15.881198059745422, 15.324918915564249),
        'FpDensityMorgan3':
            ('johnsonsu', (1.054397168130864, 1.7348730697701296,
                           2.7630213444641853, 0.3798207348262541),
             0.18115942028985507, 3.5454545454545454, 2.4726320476023513,
             0.32057971250733736),
        'fr_bicyclic': ('beta', (0.5840060328350145, 474.4004348752039,
                                 -2.2007759769553093e-31, 286.9088171814874), 0,
                        31, 0.7892552478673507, 1.2222619740000107),
        'TPSA': ('johnsonsu', (-0.8579013079024209, 1.232465212393282,
                               51.250072658107214, 29.00917664083923), 0.0,
                 3183.790000000002, 83.57268198773914, 74.84752227939705),
        'NumHeteroatoms':
            ('genlogistic', (22.144409538152043, -1.4241236643022548,
                             2.462142139013319), 0, 259, 7.61976338343684,
             5.484367394332425),
        'fr_pyridine': ('gibrat', (-0.006105798295798184, 0.016715004503743934),
                        0, 5, 0.22085545988219174, 0.48191548762954484),
        'MinEStateIndex': ('cauchy', (-0.4065559618468691, 0.4370714902017909),
                           -9.858201857916416, 1.4242592592592593,
                           -1.1147863044659592, 1.550385821215674),
        'NumHDonors':
            ('johnsonsu', (-6.303785805420036, 1.2935688877340756,
                           -0.44063513396543863, 0.023159008664878106), 0, 104,
             1.6016621163481444, 2.2472252977995697),
        'NumValenceElectrons':
            ('t', (2.854251470602213, 144.3027466924646, 32.00913472185999), 0,
             2996, 153.57779044533117, 75.10240753520117),
        'Chi0': ('fisk', (5.941141785557418, -0.07374276109511947,
                          19.65464437888598), 0.0, 405.82501416452726,
                 20.82806399367901, 9.928368342000695),
        'Kappa2': ('fisk', (4.668432120081604, 0.2539009602536103,
                            7.822349728630153), 0.2713156408808583,
                   225.9686552554005, 8.912276904621457, 5.461845335100644),
        'NHOHCount': ('invgamma', (3.8797587021158746, -1.1307664222598288,
                                   8.227943634490117), 0, 117,
                      1.7429120038402688, 2.563177769965799),
        'SMR_VSA10': ('dweibull', (1.1539393051818632, 20.846729374180775,
                                   11.835976638334323), 0.0, 491.00242136254764,
                      23.906689304667765, 16.4741302013246),
        'PEOE_VSA12': ('alpha', (0.5429249279739217, -1.1871493007688236,
                                 2.145045390430375), 0.0, 419.4097607839573,
                       5.59007921551618, 9.564823801075985),
        'PEOE_VSA1':
            ('burr', (3.9268049096073985, 0.48995232102675695,
                      -1.6407085406832138, 19.294424729301284), 0.0,
             561.0838284951058, 14.999241834817811, 13.776374483985686),
        'fr_ether': ('fisk', (0.8051168725131534, -2.567380610408963e-22,
                              0.23753230483392967), 0, 60, 0.8167371716020121,
                     1.2247008120931449),
        'EState_VSA1':
            ('foldcauchy', (0.0067487067434300755, -2.744934012978528e-09,
                            4.288215504541297), 0.0, 1261.848165001823,
             12.539067008486557, 28.935889287596773),
        'VSA_EState3': ('betaprime', (0.12979362790686721, 2.0084510281921832,
                                      -1.7874327445880742e-34,
                                      0.4022005514286884), 0.0, 0.0, 0.0, 0.0),
        'SlogP_VSA9': ('betaprime', (0.12979362790686721, 2.0084510281921832,
                                     -1.7874327445880742e-34,
                                     0.4022005514286884), 0.0, 0.0, 0.0, 0.0),
        'MaxPartialCharge':
            ('dgamma', (0.8966427125022032, 0.26138963441733876,
                        0.06744166351965852), -0.03856361615904991, 3.0,
             0.27854710402529237, 0.08364524848258846),
        'BertzCT': ('beta', (18.188950676898635, 6538386706999.18,
                             -671.968519633414, 592927086280210.0), 0.0,
                    30826.22636467273, 987.5790655792254, 632.1534060688446),
        'fr_isocyan':
            ('genhalflogistic', (2.176451524678801e-10, -8.586837508181029e-13,
                                 0.003318792604734705), 0, 1,
             0.00011000770053903773, 0.010487878662763976),
        'fr_phos_ester':
            ('genhalflogistic', (3.2594653253264924e-11, -0.0012483281868383108,
                                 0.02952659923331233), 0, 22,
             0.007020491434400408, 0.3369253069398357),
        'fr_Nhpyrrole': ('wald', (-0.01991313112984635, 0.06651081784403544), 0,
                         13, 0.11327792945506185, 0.3729728496055485),
        'fr_sulfone': ('pearson3', (2.196096781694557, 0.009513879681788937,
                                    0.01044670027530296), 0, 3,
                       0.021361495304671328, 0.14807178403739918),
        'MinAbsPartialCharge':
            ('laplace', (0.26315857867884,
                         0.0561294628943012), 0.00017822665335796042, 1.0,
             0.2739606546909672, 0.07533771658554175),
        'SMR_VSA6': ('recipinvgauss', (0.5082650634594839, -6.086954132180281,
                                       8.524443845355009), 0.0,
                     817.1820035633156, 19.209113622421707, 17.41580422042883),
        'fr_thiophene':
            ('halfgennorm', (0.5766089104240131, -6.723634040192055e-19,
                             0.0018219011330323374), 0, 3, 0.050743552048643406,
             0.22669113213778155),
        'EState_VSA11':
            ('genhalflogistic', (0.0010872264517366357, -4.398093849911524e-07,
                                 0.31899977022818193), 0.0, 163.01426425844207,
             0.11426916104059913, 1.8841357303463615),
        'fr_NH0': ('foldnorm', (0.05865369911318355, -2.4259159741023723e-09,
                                2.811848848966825), 0, 59, 2.174432210254718,
                   1.785860796591467),
        'SlogP_VSA5': ('genexpon', (0.002852587331303075, 0.23353705964951516,
                                    0.2144337777778184, -2.3573191470339223,
                                    4.447211770059805), 0.0, 649.935656436369,
                       30.740541351785637, 23.895007786195226),
        'EState_VSA10': ('genexpon', (2.705707701965304, 3.771883657629245,
                                      5.375903761907104, -4.354556984551863e-07,
                                      54.16500289129278), 0.0, 322.991248574887,
                         11.740129013811245, 11.161898269805397),
        'fr_NH1': ('exponnorm', (2892.636726669909, -0.0012981492832112645,
                                 0.00037957863319105643), 0, 70,
                   1.1072875101257087, 1.5590446880501752),
        'SlogP_VSA4':
            ('halfgennorm', (0.13290288528045635, -2.16370553583873e-12,
                             4.03300081749346e-08), 0.0, 116.32363116174835,
             6.358583799897668, 7.800100182361395),
        'fr_Ndealkylation2':
            ('wald', (-0.031069214119069204, 0.10443777001623708), 0, 12,
             0.16626163831468202, 0.4316735416648144),
        'SMR_VSA7': ('dweibull', (1.215378550488066, 56.28465156519039,
                                  23.02916244099079), 0.0, 551.7863436604832,
                     58.73997506473642, 28.427064508331743),
        'fr_nitro': ('wald', (-0.006030983799621385, 0.019984499845659677), 0,
                     4, 0.03786265038552699, 0.20368952922550065),
        'SlogP_VSA8': ('ncf', (0.010351345715274272, 2.238220060195106,
                               2.3979254167173853, -2.0305306869043397e-08,
                               0.00093545475888877), 0.0, 133.9023781814824,
                       6.766619840611257, 8.671848120645365),
        'VSA_EState2':
            ('genlogistic', (1.415613774458108, -4.0305012625995696e-05,
                             0.001088572076970272), -0.7726899469689867,
             0.0802671757431348, -3.2739988469591856e-05, 0.004083178209614408),
        'NumAromaticCarbocycles':
            ('dgamma', (3.319688925350106, 1.5062285243564748,
                        0.23908502549666388), 0, 14, 1.5350374526216835,
             0.956147979224174),
        'fr_furan': ('halfgennorm', (0.5939855958815875, -5.726016095250532e-19,
                                     0.001778526530384881), 0, 2,
                     0.038802716190133306, 0.19747704781754222),
        'PEOE_VSA13':
            ('halfgennorm', (0.2726011593040699, -6.070628359122943e-24,
                             0.0023718323492315794), 0.0, 78.17764936502758,
             3.30010114873348, 4.358778167233281),
        'fr_oxazole': ('wald', (-0.0018806042805029088, 0.006217750403694874),
                       0, 2, 0.012210854759833188, 0.11073284872542345),
        'Kappa3': ('foldnorm', (0.7381901404044835, 0.07499999952842873,
                                4.410636124980368), 0.07500000000000012,
                   1760.9964197530858, 4.873556896466643, 6.732045653350472),
        'fr_morpholine':
            ('exponweib', (1.5683748664994073, 0.5053553495208175,
                           -2.1400950538163005e-30, 0.0033466480252323464), 0,
             3, 0.046923284629924095, 0.21977656974074022),
        'fr_unbrch_alkane':
            ('genhalflogistic', (0.0001735874860695949, -2.406745376333286e-08,
                                 0.2558409046054385), 0, 167,
             0.2555878911523807, 1.8216755193439447),
        'fr_amide':
            ('recipinvgauss', (220103.5665089948, -1.4778017808899836e-11,
                               0.7451439305459917), 0, 71, 1.1422999609972697,
             1.6398141260141192),
        'NumRotatableBonds':
            ('mielke', (2.902960148717509, 4.199193993037961,
                        -0.9968227721720102, 8.267126836374882), 0, 304,
             7.140309821687518, 6.233943765856294),
        'Chi1': ('mielke', (5.372526418714403, 6.068367649928466,
                            -0.06790886881111757, 13.725737593680664), 0.0,
                 256.9697300670324, 14.033526281218712, 6.54332057037178),
        'fr_phos_acid':
            ('genhalflogistic', (3.2594653253264924e-11, -0.0012483281868383108,
                                 0.02952659923331233), 0, 22,
             0.007130499134939446, 0.3375013082055841),
        'fr_piperdine':
            ('halfgennorm', (0.4238794287379706, -1.2796542653568352e-22,
                             0.0018770831152461857), 0, 6, 0.14572020041402897,
             0.39164760531240855),
        'fr_isothiocyan':
            ('genhalflogistic', (8.886662035991823e-09, -0.0008924947259535538,
                                 0.002685293974604489), 0, 2,
             0.0002300161011270789, 0.015810268619422887),
        'MinAbsEStateIndex':
            ('genpareto', (0.11654274500988507, -4.295406365103898e-11,
                           0.13245709790535898), 0.0, 2.5115740740740735,
             0.15016751055787822, 0.17292082606243664),
        'fr_methoxy': ('gibrat', (-0.01550907507397075, 0.042866902160229356),
                       0, 8, 0.31338193673557146, 0.6523713877037112)
    }
