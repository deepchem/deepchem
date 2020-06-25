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
from copy import deepcopy
from collections import Counter
from deepchem.utils import pdbqt_utils
from scipy.spatial.distance import cdist
from deepchem.utils import pdbqt_utils
from deepchem.utils.pdbqt_utils import convert_mol_to_pdbqt
from deepchem.utils.pdbqt_utils import convert_protein_to_pdbqt
from deepchem.utils.geometry_utils import angle_between
from deepchem.utils.geometry_utils import is_angle_within_cutoff
from deepchem.utils.geometry_utils import generate_random_rotation_matrix
from deepchem.utils.fragment_util import get_partial_charge

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)


class MoleculeLoadException(Exception):

  def __init__(self, *args, **kwargs):
    Exception.__init__(*args, **kwargs)


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
  return apply_pdbfixer(mol, hydrogenate=True)


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
    Molecule to hydrogenate
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
    from rdkit import Chem
    pdbblock = Chem.MolToPDBBlock(mol)
    pdb_stringio = StringIO()
    pdb_stringio.write(pdbblock)
    pdb_stringio.seek(0)
    import pdbfixer
    fixer = pdbfixer.PDBFixer(pdbfile=pdb_stringio)
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
    import simtk
    simtk.openmm.app.PDBFile.writeFile(fixer.topology, fixer.positions,
                                       hydrogenated_io)
    hydrogenated_io.seek(0)
    return Chem.MolFromPDBBlock(
        hydrogenated_io.read(), sanitize=False, removeHs=False)
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

  Params
  ------
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


def load_complex(molecular_complex,
                 add_hydrogens=True,
                 calc_charges=True,
                 sanitize=True):
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
    molecule_complex = [molecular_complex]
  fragments = []
  # TODO: How do I handle the is_protein here?
  for mol in molecular_complex:
    loaded = load_molecule(
        mol,
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
  from rdkit.Chem.rdchem import AtomValenceException
  from_pdb = False
  if ".mol2" in molecule_file:
    my_mol = Chem.MolFromMol2File(molecule_file, sanitize=False, removeHs=False)
  elif ".sdf" in molecule_file:
    suppl = Chem.SDMolSupplier(str(molecule_file), sanitize=False)
    # TODO: This is wrong. Should return all molecules
    my_mol = suppl[0]
  elif ".pdbqt" in molecule_file:
    pdb_block = pdbqt_utils.pdbqt_to_pdb(molecule_file)
    my_mol = Chem.MolFromPDBBlock(
        str(pdb_block), sanitize=False, removeHs=False)
    from_pdb = True
  elif ".pdb" in molecule_file:
    my_mol = Chem.MolFromPDBFile(
        str(molecule_file), sanitize=False, removeHs=False)
    from_pdb = True
  else:
    raise ValueError("Unrecognized file type for %s" % str(molecule_file))

  if my_mol is None:
    raise ValueError("Unable to read non None Molecule Object")

  if add_hydrogens or calc_charges:
    my_mol = apply_pdbfixer(
        my_mol, hydrogenate=add_hydrogens, is_protein=is_protein)
  if sanitize:
    try:
      Chem.SanitizeMol(my_mol)
    # Ideally we should catch AtomValenceException but Travis seems to choke on it for some reason.
    except:
      logger.warn("Mol %s failed sanitization" % Chem.MolToSmiles(my_mol))
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
      pdbqt_utils.convert_protein_to_pdbqt(mol, outfile)
    else:
      pdbqt_utils.convert_mol_to_pdbqt(mol, outfile)
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


def merge_molecular_fragments(molecules):
  """Helper method to merge two molecular fragments.

  Parameters
  ----------
  molecules: list
    List of `MolecularFragment` objects. 

  Returns
  -------
  merged: `MolecularFragment`
  """
  if len(molecules) == 0:
    return None
  if len(molecules) == 1:
    return molecules[0]
  else:
    all_atoms = []
    for mol_frag in molecules:
      all_atoms += mol_frag.GetAtoms()
    return MolecularFragment(all_atoms)


def strip_hydrogens(coords, mol):
  """Strip the hydrogens from input molecule

  Parameters
  ----------
  coords: Numpy ndarray
    Must be of shape (N, 3) and correspond to coordinates of mol.
  mol: Rdkit mol or `MolecularFragment`
    The molecule to strip

  Returns
  -------
  A tuple of (coords, mol_frag) where coords is a Numpy array of
  coordinates with hydrogen coordinates. mol_frag is a
  `MolecularFragment`. 
  """
  mol_atoms = mol.GetAtoms()
  atomic_numbers = [atom.GetAtomicNum() for atom in mol_atoms]
  atom_indices_to_keep = [
      ind for (ind, atomic_number) in enumerate(atomic_numbers)
      if (atomic_number != 1)
  ]
  return get_mol_subset(coords, mol, atom_indices_to_keep)


class MolecularFragment(object):
  """A class that represents a fragment of a molecule.

  It's often convenient to represent a fragment of a molecule. For
  example, if two molecules form a molecular complex, it may be useful
  to create two fragments which represent the subsets of each molecule
  that's close to the other molecule (in the contact region).

  Ideally, we'd be able to do this in RDKit direct, but manipulating
  molecular fragments doesn't seem to be supported functionality. 
  """

  def __init__(self, atoms):
    """Initialize this object.

    Parameters
    ----------
    atoms: list
      Each entry in this list should be an RdkitAtom
    """
    #self.atoms = [AtomShim(x) for x in atoms]
    self.atoms = [
        AtomShim(x.GetAtomicNum(), get_partial_charge(x)) for x in atoms
    ]
    #self.atoms = atoms

  def GetAtoms(self):
    """Returns the list of atoms

    Returns
    -------
    list of atoms in this fragment.
    """
    return self.atoms


class AtomShim(object):
  """This is a shim object wrapping an atom.

  We use this class instead of raw RDKit atoms since manipulating a
  large number of rdkit Atoms seems to result in segfaults. Wrapping
  the basic information in an AtomShim seems to avoid issues.
  """

  def __init__(self, atomic_num, partial_charge):
    """Initialize this object

    Parameters
    ----------
    atomic_num: int
      Atomic number for this atom.
    partial_charge: float
      The partial Gasteiger charge for this atom
    """
    self.atomic_num = atomic_num
    self.partial_charge = partial_charge

  def GetAtomicNum(self):
    return self.atomic_num

  def GetPartialCharge(self):
    return self.partial_charge


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
  mol_frag = MolecularFragment(atoms_to_keep)
  coords = coords[indexes_to_keep]
  return coords, mol_frag


def reduce_molecular_complex_to_contacts(fragments, cutoff=4.5):
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


def get_contact_atom_indices(fragments, cutoff=4.5):
  """Compute that atoms close to contact region.

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
  # indices to atoms to keep
  keep_inds = [set([]) for _ in fragments]
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
  keep_inds = [sorted(list(keep)) for keep in keep_inds]
  return keep_inds

  # Now extract atoms
  #atoms_to_keep = []
  #for i, frag_keep_inds in enumerate(keep_inds):
  #  frag = fragments[i]
  #  mol = frag[1]
  #  atoms = mol.GetAtoms()
  #  frag_keep = [atoms[keep_ind] for keep_ind in frag_keep_inds]
  #  atoms_to_keep.append(frag_keep)
  #return atoms_to_keep


def compute_contact_centroid(molecular_complex, cutoff=4.5):
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


def rotate_molecules(mol_coordinates_list):
  """Rotates provided molecular coordinates.

  Pseudocode:
  1. Generate random rotation matrix. This matrix applies a
     random transformation to any 3-vector such that, were the
     random transformation repeatedly applied, it would randomly
     sample along the surface of a sphere with radius equal to
     the norm of the given 3-vector cf.
     generate_random_rotation_matrix() for details
  2. Apply R to all atomic coordinates.
  3. Return rotated molecule

  Parameters
  ----------
  mol_coordinates_list: list
    Elements of list must be (N_atoms, 3) shaped arrays
  """
  R = generate_random_rotation_matrix()
  rotated_coordinates_list = []

  for mol_coordinates in mol_coordinates_list:
    coordinates = deepcopy(mol_coordinates)
    rotated_coordinates = np.transpose(np.dot(R, np.transpose(coordinates)))
    rotated_coordinates_list.append(rotated_coordinates)

  return (rotated_coordinates_list)


def compute_all_ecfp(mol, indices=None, degree=2):
  """Obtain molecular fragment for all atoms emanating outward to given degree.

  For each fragment, compute SMILES string (for now) and hash to
  an int. Return a dictionary mapping atom index to hashed
  SMILES.
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
