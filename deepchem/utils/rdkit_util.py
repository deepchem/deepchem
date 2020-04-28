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
from deepchem.utils.pdbqt_utils import pdbqt_to_pdb
from deepchem.utils.pdbqt_utils import convert_mol_to_pdbqt
from deepchem.utils.pdbqt_utils import convert_protein_to_pdbqt
from deepchem.utils.geometry_utils import angle_between
from deepchem.utils.geometry_utils import is_angle_within_cutoff
from deepchem.utils.geometry_utils import generate_random_rotation_matrix
from rdkit import Chem
from rdkit.Chem.rdchem import AtomValenceException

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)

class MoleculeLoadException(Exception):

  def __init__(self, *args, **kwargs):
    Exception.__init__(*args, **kwargs)


def compute_pairwise_distances(first_xyz, second_xyz):
  """Computes pairwise distances between two molecules.

  Takes an input (m, 3) and (n, 3) numpy arrays of 3D coords of
  two molecules respectively, and outputs an m x n numpy
  array of pairwise distances in Angstroms between the first and
  second molecule. entry (i,j) is dist between the i"th 
  atom of first molecule and the j"th atom of second molecule.

  Parameters
  ----------
  first_xyz: np.ndarray
    Of shape (m, 3)
  seocnd_xyz: np.ndarray
    Of shape (n, 3)

  Returns
  -------
  np.ndarray of shape (m, n)
  """

  pairwise_distances = cdist(first_xyz, second_xyz,
                             metric='euclidean')
  return pairwise_distances

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


def apply_pdbfixer(mol, add_missing=True, hydrogenate=True, pH=7.4,
                   remove_heterogens=True, is_protein=True):
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
  from_pdb = False
  if ".mol2" in molecule_file:
    my_mol = Chem.MolFromMol2File(molecule_file, sanitize=False, removeHs=False)
  elif ".sdf" in molecule_file:
    suppl = Chem.SDMolSupplier(str(molecule_file), sanitize=False)
    # TODO: This is wrong. Should return all molecules
    my_mol = suppl[0]
  elif ".pdbqt" in molecule_file:
    pdb_block = pdbqt_to_pdb(molecule_file)
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
      logger.warning("Mol %s failed sanitization" % Chem.MolToSmiles(my_mol))
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
  atom_indices_to_keep = [ind for (ind, atomic_number) in enumerate(atomic_numbers) if (atomic_number != 1)]
  return get_mol_subset(coords, mol, atom_indices_to_keep)

class MolecularFragment(object):
  """A class that represents a fragment of a molecule.

  It's often convenient to represent a fragment of a molecule. For
  example, if two molecules form a molecular complex, it may be useful
  to create two fragments which represent the subsets of each molecule
  that's close to the other molecule (in the contact region).
  """

  def __init__(self, atoms):
    """Initialize this object.

    Parameters
    ----------
    atoms: list
      Each entry in this list should be an rdkit Atom object.
    """
    self.atoms = [AtomShim(x) for x in atoms]
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

  def __init__(self, atomic_num):
    """Initialize this object

    Parameters
    ----------
    atomic_num: int
      Atomic number for this atom.
    """
    self.atomic_num = atomic_num

  def GetAtomicNum(self):
    return self.atomic_num

def get_mol_subset(coords, mol, atom_indices_to_keep):
  """Strip a subset of the atoms in this molecule

  Parameters
  ----------
  coords: Numpy ndarray
    Must be of shape (N, 3) and correspond to coordinates of mol.
  mol: Rdkit mol
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

  indexes_to_keep = []
  atoms_to_keep = []
  atoms = list(mol.GetAtoms())
  for index in atom_indices_to_keep:
    indexes_to_keep.append(index)
    #atomic_numbers.append(atom.GetAtomicNum())
    atoms_to_keep.append(atoms[index])
  #mol = MolecularFragment(atomic_numbers)
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


def compute_centroid(coordinates):
  """Compute the x,y,z centroid of provided coordinates

  coordinates: np.ndarray
    Shape (N, 3), where N is number atoms.
  """
  centroid = np.mean(coordinates, axis=0)
  return (centroid)

def subtract_centroid(xyz, centroid):
  """Subtracts centroid from each coordinate.

  Subtracts the centroid, a numpy array of dim 3, from all coordinates of all
  atoms in the molecule
  """
  xyz -= np.transpose(centroid)
  return (xyz)

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


def is_hydrogen_bond(protein_xyz,
                     protein,
                     ligand_xyz,
                     ligand,
                     contact,
                     hbond_distance_cutoff=4.0,
                     hbond_angle_cutoff=40.0):
  """
  Determine if a pair of atoms (contact = tuple of protein_atom_index, ligand_atom_index)
  between protein and ligand represents a hydrogen bond. Returns a boolean result.
  """
  protein_atom_xyz = protein_xyz[int(contact[0])]
  ligand_atom_xyz = ligand_xyz[int(contact[1])]
  protein_atom = protein.GetAtoms()[int(contact[0])]
  ligand_atom = ligand.GetAtoms()[int(contact[1])]

  # Nitrogen has atomic number 7, and oxygen 8.
  if ((ligand_atom.GetAtomicNum() == 7 or ligand_atom.GetAtomicNum() == 8) and (protein_atom.GetAtomicNum() == 7 or protein_atom.GetAtomicNum() == 8)):
    hydrogens = []

    for i, atom in enumerate(ligand.GetAtoms()):
      # If atom is a hydrogen
      if atom.GetAtomicNum() == 1:
        atom_xyz = ligand_xyz[i]
        dist = np.linalg.norm(atom_xyz-ligand_atom_xyz)
        # O-H distance is 0.96 A, N-H is 1.01 A. See http://www.science.uwaterloo.ca/~cchieh/cact/c120/bondel.html
        if dist < 1.3:
          hydrogens.append(atom_xyz)
    
    for j, atom in enumerate(protein.GetAtoms()):
      # If atom is a hydrogen
      if atom.GetAtomicNum() == 1:
        atom_xyz = protein_xyz[i]
        dist = np.linalg.norm(atom_xyz-protein_atom_xyz)
        # O-H distance is 0.96 A, N-H is 1.01 A. See http://www.science.uwaterloo.ca/~cchieh/cact/c120/bondel.html
        if dist < 1.3:
          hydrogens.append(atom_xyz)

    for hydrogen_xyz in hydrogens:
      hydrogen_to_ligand = ligand_atom_xyz - hydrogen_xyz
      hydrogen_to_protein = protein_atom_xyz - hydrogen_xyz
      if np.abs(180 - angle_between(hydrogen_to_protein, hydrogen_to_ligand) * 180.0 / np.pi) <= hbond_angle_cutoff:
        return True
  return False


def compute_hbonds_in_range(protein, protein_xyz, ligand, ligand_xyz,
                            pairwise_distances, hbond_dist_bin,
                            hbond_angle_cutoff):
  """
  Find all pairs of (protein_index_i, ligand_index_j) that hydrogen bond given

  a distance bin and an angle cutoff.
  """

  contacts = np.nonzero((pairwise_distances > hbond_dist_bin[0]) &
                        (pairwise_distances < hbond_dist_bin[1]))
  contacts = zip(contacts[0], contacts[1])
  hydrogen_bond_contacts = []
  for contact in contacts:
    if is_hydrogen_bond(protein_xyz, protein, ligand_xyz, ligand, contact,
                        hbond_angle_cutoff):
      hydrogen_bond_contacts.append(contact)
  return hydrogen_bond_contacts

def compute_hydrogen_bonds(protein_xyz, protein, ligand_xyz, ligand,
                           pairwise_distances, hbond_dist_bins,
                           hbond_angle_cutoffs):
  """Computes hydrogen bonds between proteins and ligands.

  Returns a list of sublists. Each sublist is a series of tuples
  of (protein_index_i, ligand_index_j) that represent a hydrogen
  bond. Each sublist represents a different type of hydrogen
  bond.
  """

  hbond_contacts = []
  for i, hbond_dist_bin in enumerate(hbond_dist_bins):
    hbond_angle_cutoff = hbond_angle_cutoffs[i]
    hbond_contacts.append(
        compute_hbonds_in_range(protein, protein_xyz, ligand, ligand_xyz,
                                pairwise_distances, hbond_dist_bin,
                                hbond_angle_cutoff))
  return (hbond_contacts)

def compute_salt_bridges(first,
                         second,
                         pairwise_distances,
                         cutoff=5.0):
  """Find salt bridge contacts between two molecules. 

  Parameters:
  -----------
  first: rdkit.rdchem.Mol
    Interacting molecules
  second: rdkit.rdchem.Mol
    Interacting molecules
  pairwise_distances: np.ndarray
    Array of pairwise interatomic distances between molecule atoms (Angstroms)
  cutoff: float
    Cutoff distance for contact consideration

  Returns:
  --------
  salt_bridge_contacts: list of tuples
    List of contacts. Tuple (i, j) indicates that atom i from
    first molecule interacts with atom j from second.
  """

  salt_bridge_contacts = []
  contacts = np.nonzero(pairwise_distances < cutoff)
  contacts = zip(contacts[0], contacts[1])
  for contact in contacts:
    first_atom = first.GetAtoms()[int(contact[0])]
    second_atom = second.GetAtoms()[int(contact[1])]
    if is_salt_bridge(first_atom, second_atom):
      salt_bridge_contacts.append(contact)
  return salt_bridge_contacts

def compute_cation_pi(mol1, mol2, charge_tolerance=0.01, **kwargs):
  """Finds aromatic rings in mo1 and cations in mol2 that interact with each other.

  Parameters:
  -----------
  mol1: rdkit.rdchem.Mol
    Molecule to look for interacting rings
  mol2: rdkit.rdchem.Mol
    Molecule to look for interacting cations
  charge_tolerance: float
    Atom is considered a cation if its formal charge is greater
    than 1 - charge_tolerance
  **kwargs:
    Arguments that are passed to is_cation_pi function

  Returns:
  --------
  mol1_pi: dict
    Dictionary that maps atom indices (from mol1) to the number of cations
    (in mol2) they interact with
  mol2_cation: dict
    Dictionary that maps atom indices (from mol2) to the number of aromatic
    atoms (in mol1) they interact with
  """
  mol1_pi = Counter()
  mol2_cation = Counter()
  conformer = mol2.GetConformer()

  aromatic_atoms = set(atom.GetIdx() for atom in mol1.GetAromaticAtoms())
  from rdkit import Chem
  rings = [list(r) for r in Chem.GetSymmSSSR(mol1)]

  for ring in rings:
    # if ring from mol1 is aromatic
    if set(ring).issubset(aromatic_atoms):
      ring_center = compute_ring_center(mol1, ring)
      ring_normal = compute_ring_normal(mol1, ring)

      for atom in mol2.GetAtoms():
        # ...and atom from mol2 is a cation
        if atom.GetFormalCharge() > 1.0 - charge_tolerance:
          cation_position = np.array(conformer.GetAtomPosition(atom.GetIdx()))
          # if angle and distance are correct
          if is_cation_pi(cation_position, ring_center, ring_normal, **kwargs):
            # count atoms forming a contact
            mol1_pi.update(ring)
            mol2_cation.update([atom.GetIndex()])
  return mol1_pi, mol2_cation

def is_cation_pi(cation_position,
                 ring_center,
                 ring_normal,
                 dist_cutoff=6.5,
                 angle_cutoff=30.0):
  """Check if a cation and an aromatic ring form contact.

  Parameters:
  -----------
  ring_center: np.ndarray
    Positions of ring center. Can be computed with the compute_ring_center
    function.
  ring_normal: np.ndarray
    Normal of ring. Can be computed with the compute_ring_normal function.
  dist_cutoff: float
    Distance cutoff. Max allowed distance between ring center
    and cation (in Angstroms).
  angle_cutoff: float
    Angle cutoff. Max allowed deviation from the ideal (0deg)
    angle between ring normal and vector pointing from ring
    center to cation (in degrees).
  """
  cation_to_ring_vec = cation_position - ring_center
  dist = np.linalg.norm(cation_to_ring_vec)
  angle = angle_between(cation_to_ring_vec, ring_normal) * 180. / np.pi
  if ((angle < angle_cutoff or angle > 180.0 - angle_cutoff) and
      (dist < dist_cutoff)):
    return True
  return False

def compute_pi_stack(mol1,
                     mol2,
                     pairwise_distances=None,
                     dist_cutoff=4.4,
                     angle_cutoff=30.):
  """Find aromatic rings in both molecules that form pi-pi contacts.
  For each atom in the contact, count number of atoms in the other molecule
  that form this contact.

  Pseudocode:

  for each aromatic ring in mol1:
    for each aromatic ring in mol2:
      compute distance between centers
      compute angle between normals
      if it counts as parallel pi-pi:
        count interacting atoms
      if it counts as pi-T:
        count interacting atoms

  Parameters:
  -----------
    mol1: rdkit.rdchem.Mol
      First molecule.
    mol2: rdkit.rdchem.Mol
      First molecule.
    pairwise_distances: np.ndarray (optional)
      Array of pairwise interatomic distances (Angstroms)
    dist_cutoff: float
      Distance cutoff. Max allowed distance between the ring center (Angstroms).
    angle_cutoff: float
      Angle cutoff. Max allowed deviation from the ideal angle between rings.

  Returns:
  --------
    mol1_pi_t, mol1_pi_parallel, mol2_pi_t, mol2_pi_parallel: dict
      Dictionaries mapping atom indices to number of atoms they interact with.
      Separate dictionary is created for each type of pi stacking (parallel and
      T-shaped) and each molecule (mol1 and mol2).
  """

  mol1_pi_parallel = Counter()
  mol1_pi_t = Counter()
  mol2_pi_parallel = Counter()
  mol2_pi_t = Counter()

  mol1_aromatic_rings = []
  mol2_aromatic_rings = []
  from rdkit import Chem
  for mol, ring_list in ((mol1, mol1_aromatic_rings),
                         (mol2, mol2_aromatic_rings)):
    aromatic_atoms = {atom.GetIdx() for atom in mol.GetAromaticAtoms()}
    for ring in Chem.GetSymmSSSR(mol):
      # if ring is aromatic
      if set(ring).issubset(aromatic_atoms):
        # save its indices, center, and normal
        ring_center = compute_ring_center(mol, ring)
        ring_normal = compute_ring_normal(mol, ring)
        ring_list.append((ring, ring_center, ring_normal))

  # remember mol1-mol2 pairs we already counted
  counted_pairs_parallel = set()
  counted_pairs_t = set()
  for prot_ring, prot_ring_center, prot_ring_normal in mol1_aromatic_rings:
    for lig_ring, lig_ring_center, lig_ring_normal in mol2_aromatic_rings:
      if is_pi_parallel(
          prot_ring_center,
          prot_ring_normal,
          lig_ring_center,
          lig_ring_normal,
          angle_cutoff=angle_cutoff,
          dist_cutoff=dist_cutoff):
        prot_to_update = set()
        lig_to_update = set()
        for prot_atom_idx in prot_ring:
          for lig_atom_idx in lig_ring:
            if (prot_atom_idx, lig_atom_idx) not in counted_pairs_parallel:
              # if this pair is new, count atoms forming a contact
              prot_to_update.add(prot_atom_idx)
              lig_to_update.add(lig_atom_idx)
              counted_pairs_parallel.add((prot_atom_idx, lig_atom_idx))

        mol1_pi_parallel.update(prot_to_update)
        mol2_pi_parallel.update(lig_to_update)

      if is_pi_t(
          prot_ring_center,
          prot_ring_normal,
          lig_ring_center,
          lig_ring_normal,
          angle_cutoff=angle_cutoff,
          dist_cutoff=dist_cutoff):
        prot_to_update = set()
        lig_to_update = set()
        for prot_atom_idx in prot_ring:
          for lig_atom_idx in lig_ring:
            if (prot_atom_idx, lig_atom_idx) not in counted_pairs_t:
              # if this pair is new, count atoms forming a contact
              prot_to_update.add(prot_atom_idx)
              lig_to_update.add(lig_atom_idx)
              counted_pairs_t.add((prot_atom_idx, lig_atom_idx))

        mol1_pi_t.update(prot_to_update)
        mol2_pi_t.update(lig_to_update)

  return (mol1_pi_t, mol1_pi_parallel, mol2_pi_t, mol2_pi_parallel)

def is_pi_t(ring1_center,
            ring1_normal,
            ring2_center,
            ring2_normal,
            dist_cutoff=5.5,
            angle_cutoff=30.0):
  """Check if two aromatic rings form a T-shaped pi-pi contact.

  Parameters:
  -----------
  ring1_center, ring2_center: np.ndarray
    Positions of centers of the two rings. Can be computed with the
    compute_ring_center function.
  ring1_normal, ring2_normal: np.ndarray
    Normals of the two rings. Can be computed with the compute_ring_normal
    function.
  dist_cutoff: float
    Distance cutoff. Max allowed distance between the ring center (Angstroms).
  angle_cutoff: float
    Angle cutoff. Max allowed deviation from the ideal (90deg) angle between
    the rings (in degrees).
  """
  dist = np.linalg.norm(ring1_center - ring2_center)
  angle = angle_between(ring1_normal, ring2_normal) * 180 / np.pi
  if ((90.0 - angle_cutoff < angle < 90.0 + angle_cutoff) and
      dist < dist_cutoff):
    return True
  return False

def is_pi_parallel(ring1_center,
                   ring1_normal,
                   ring2_center,
                   ring2_normal,
                   dist_cutoff=8.0,
                   angle_cutoff=30.0):
  """Check if two aromatic rings form a parallel pi-pi contact.

  Parameters:
  -----------
    ring1_center, ring2_center: np.ndarray
      Positions of centers of the two rings. Can be computed with the
      compute_ring_center function.
    ring1_normal, ring2_normal: np.ndarray
      Normals of the two rings. Can be computed with the compute_ring_normal
      function.
    dist_cutoff: float
      Distance cutoff. Max allowed distance between the ring center (Angstroms).
    angle_cutoff: float
      Angle cutoff. Max allowed deviation from the ideal (0deg) angle between
      the rings (in degrees).
  """

  dist = np.linalg.norm(ring1_center - ring2_center)
  angle = angle_between(ring1_normal, ring2_normal) * 180 / np.pi
  if ((angle < angle_cutoff or angle > 180.0 - angle_cutoff) and
      dist < dist_cutoff):
    return True
  return False

def compute_binding_pocket_cation_pi(mol1, mol2, **kwargs):
  """Finds cation-pi interactions between mol1 and mol2.

  Parameters:
  -----------
  mol1: rdkit.rdchem.Mol
    Interacting molecules
  mol2: rdkit.rdchem.Mol
    Interacting molecules
  **kwargs:
    Arguments that are passed to compute_cation_pi function

  Returns:
  --------
  mol1_cation_pi, mol2_cation_pi: dict
    Dictionaries that maps atom indices to the number of cations/aromatic
    atoms they interact with
  """
  # find interacting rings from mol1 and cations from mol2 
  mol1_pi, mol2_cation = compute_cation_pi(mol1, mol2, **kwargs)
  # find interacting cations from mol1 and rings from mol2 
  mol2_pi, mol1_cation = compute_cation_pi(mol2, mol1, **kwargs)

  # merge counters
  mol1_cation_pi = Counter()
  mol1_cation_pi.update(mol1_pi)
  mol1_cation_pi.update(mol1_cation)

  mol2_cation_pi = Counter()
  mol2_cation_pi.update(mol2_pi)
  mol2_cation_pi.update(mol2_cation)

  return mol1_cation_pi, mol2_cation_pi

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
