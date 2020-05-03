"""A collection of utilities for dealing with Molecular Fragments"""
import itertools
import numpy as np
from deepchem.utils.geometry_utils import compute_pairwise_distances

def get_partial_charge(atom):
  """Get partial charge of a given atom (rdkit Atom object)
  
  Parameters
  ----------
  atom: rdkit atom or `AtomShim` object
    Either an rdkit atom or `AtomShim`
  """
  from rdkit import Chem
  if isinstance(atom, Chem.Atom):
    try:
      value = atom.GetProp(str("_GasteigerCharge"))
      if value == '-nan':
        return 0
      return float(value)
    except KeyError:
      return 0
  else:
    return atom.GetPartialCharge()


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
    self.atoms = [AtomShim(x.GetAtomicNum(), get_partial_charge(x)) for x in atoms]

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
  is a tuple of `(coords, MolecularFragment)`. The coords is stripped
  down to `(N_contact_atoms, 3)` where `N_contact_atoms` is the number
  of contact atoms for this complex. `MolecularFragment` is used since
  it's tricky to make a RDKit sub-molecule. 
  """
  atoms_to_keep = get_contact_atom_indices(fragments, cutoff)
  reduced_complex = []
  for frag, keep in zip(fragments, atoms_to_keep):
    contact_frag = get_mol_subset(frag[0], frag[1], keep)
    reduced_complex.append(contact_frag)
  return reduced_complex
