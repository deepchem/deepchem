"""The functions in these utilities check that noncovalent interactions happen"""
import numpy as np
from collections import Counter
from deepchem.utils.fragment_utils import get_partial_charge
from deepchem.utils.rdkit_utils import compute_ring_center
from deepchem.utils.rdkit_utils import compute_ring_normal
from deepchem.utils.geometry_utils import angle_between
from deepchem.utils.geometry_utils import is_angle_within_cutoff


def is_salt_bridge(atom_i, atom_j):
  """Check if two atoms have correct charges to form a salt bridge"""
  if np.abs(2.0 - np.abs(
      get_partial_charge(atom_i) - get_partial_charge(atom_j))) < 0.01:
    return True
  return False


def compute_salt_bridges(first, second, pairwise_distances, cutoff=5.0):
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


def is_hydrogen_bond(frag1,
                     frag2,
                     contact,
                     hbond_distance_cutoff=4.0,
                     hbond_angle_cutoff=40.0):
  """
  Determine if a pair of atoms (contact = frag1_atom_index,
  frag2_atom_index) between two molecules represents a hydrogen
  bond. Returns a boolean result.

  Parameters
  ----------
  frag1: tuple
    Tuple of (coords, rdkit mol / MolecularFragment)
  frag2: tuple
    Tuple of (coords, rdkit mol / MolecularFragment)
  contact: Tuple
    Tuple of indices for (atom_i, atom_j) contact.
  hbond_distance_cutoff: float, optional
    Distance cutoff for hbond.
  hbond_angle_cutoff: float, optional
    Angle deviance cutoff for hbond
  """
  frag1_xyz, frag2_xyz = frag1[0], frag2[0]
  frag1_mol, frag2_mol = frag1[1], frag2[1]
  frag1_atom_xyz = frag1_xyz[int(contact[0])]
  frag2_atom_xyz = frag2_xyz[int(contact[1])]
  frag1_atom = frag1_mol.GetAtoms()[int(contact[0])]
  frag2_atom = frag2_mol.GetAtoms()[int(contact[1])]

  # Nitrogen has atomic number 7, and oxygen 8.
  if ((frag2_atom.GetAtomicNum() == 7 or frag2_atom.GetAtomicNum() == 8) and
      (frag1_atom.GetAtomicNum() == 7 or frag1_atom.GetAtomicNum() == 8)):
    hydrogens = []

    for i, atom in enumerate(frag2_mol.GetAtoms()):
      # If atom is a hydrogen
      if atom.GetAtomicNum() == 1:
        atom_xyz = frag2_xyz[i]
        dist = np.linalg.norm(atom_xyz - frag2_atom_xyz)
        # O-H distance is 0.96 A, N-H is 1.01 A. See http://www.science.uwaterloo.ca/~cchieh/cact/c120/bondel.html
        if dist < 1.3:
          hydrogens.append(atom_xyz)

    for j, atom in enumerate(frag1_mol.GetAtoms()):
      # If atom is a hydrogen
      if atom.GetAtomicNum() == 1:
        atom_xyz = frag1_xyz[i]
        dist = np.linalg.norm(atom_xyz - frag1_atom_xyz)
        # O-H distance is 0.96 A, N-H is 1.01 A. See http://www.science.uwaterloo.ca/~cchieh/cact/c120/bondel.html
        if dist < 1.3:
          hydrogens.append(atom_xyz)

    for hydrogen_xyz in hydrogens:
      hydrogen_to_frag2 = frag2_atom_xyz - hydrogen_xyz
      hydrogen_to_frag1 = frag1_atom_xyz - hydrogen_xyz
      return is_angle_within_cutoff(hydrogen_to_frag2, hydrogen_to_frag1,
                                    hbond_angle_cutoff)
  return False


def compute_hbonds_in_range(frag1, frag2, pairwise_distances, hbond_dist_bin,
                            hbond_angle_cutoff):
  """
  Find all pairs of (frag1_index_i, frag2_index_j) that hydrogen bond
  given a distance bin and an angle cutoff.

  Parameters
  ----------
  frag1: tuple
    Tuple of (coords, rdkit mol / MolecularFragment
  frag2: tuple
    Tuple of (coords, rdkit mol / MolecularFragment
  pairwise_distances:
    Matrix of shape `(N, M)` with pairwise distances between frag1/frag2.
  hbond_dist_bin: tuple
    Tuple of floats `(min_dist, max_dist)` in angstroms.
  hbond_angle_cutoffs: list[float]
    List of angles of deviances allowed for hbonds
  """

  contacts = np.nonzero((pairwise_distances > hbond_dist_bin[0]) &
                        (pairwise_distances < hbond_dist_bin[1]))
  contacts = zip(contacts[0], contacts[1])
  hydrogen_bond_contacts = []
  for contact in contacts:
    if is_hydrogen_bond(frag1, frag2, contact, hbond_angle_cutoff):
      hydrogen_bond_contacts.append(contact)
  return hydrogen_bond_contacts


def compute_hydrogen_bonds(frag1, frag2, pairwise_distances, hbond_dist_bins,
                           hbond_angle_cutoffs):
  """Computes hydrogen bonds between proteins and ligands.

  Returns a list of sublists. Each sublist is a series of tuples
  of (protein_index_i, ligand_index_j) that represent a hydrogen
  bond. Each sublist represents a different type of hydrogen
  bond.

  Parameters
  ----------
  frag1: tuple
    Tuple of (coords, rdkit mol / MolecularFragment
  frag2: tuple
    Tuple of (coords, rdkit mol / MolecularFragment
  pairwise_distances:
    Matrix of shape `(N, M)` with pairwise distances between frag1/frag2.
  hbond_dist_bins: list[tuple]
    List of tuples of hbond distance ranges.
  hbond_angle_cutoffs: list[float]
    List of angles of deviances allowed for hbonds

  Returns
  -------
  List
    A list of hydrogen bond contacts.
  """

  hbond_contacts = []
  for i, hbond_dist_bin in enumerate(hbond_dist_bins):
    hbond_angle_cutoff = hbond_angle_cutoffs[i]
    hbond_contacts.append(
        compute_hbonds_in_range(frag1, frag2, pairwise_distances,
                                hbond_dist_bin, hbond_angle_cutoff))
  return (hbond_contacts)


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
      Second molecule.
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
  for mol, ring_list in ((mol1, mol1_aromatic_rings), (mol2,
                                                       mol2_aromatic_rings)):
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


def is_pi_parallel(ring1_center: np.ndarray,
                   ring1_normal: np.ndarray,
                   ring2_center: np.ndarray,
                   ring2_normal: np.ndarray,
                   dist_cutoff: float = 8.0,
                   angle_cutoff: float = 30.0) -> bool:
  """Check if two aromatic rings form a parallel pi-pi contact.

  Parameters
  ----------
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

  Returns
  -------
  bool
    True if two aromatic rings form a parallel pi-pi.
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
