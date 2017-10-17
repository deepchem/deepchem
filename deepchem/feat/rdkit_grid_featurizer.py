from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

__author__ = "Bharath Ramsundar, Evan Feinberg, and Karl Leswing"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "MIT"

import os
import shutil
import time
import tempfile
import hashlib
from rdkit import Chem
from rdkit.Chem import AllChem
import deepchem.utils.rdkit_util as rdkit_util

import numpy as np
from copy import deepcopy
from functools import partial
from deepchem.feat import ComplexFeaturizer
from deepchem.utils.save import log
"""
TODO(LESWING) add sanitization with rdkit upgrade to 2017.*
"""


def get_ligand_filetype(ligand_filename):
  """Returns the filetype of ligand."""
  if ".mol2" in ligand_filename:
    return "mol2"
  elif ".sdf" in ligand_filename:
    return "sdf"
  elif ".pdbqt" in ligand_filename:
    return "pdbqt"
  elif ".pdb" in ligand_filename:
    return "pdb"
  else:
    raise ValueError("Unrecognized_filename")


def load_molecule(molecule_file, add_hydrogens=True, calc_charges=False):
  return rdkit_util.load_molecule(molecule_file, add_hydrogens, calc_charges)


def merge_two_dicts(x, y):
  """Given two dicts, merge them into a new dict as a shallow copy."""
  z = x.copy()
  z.update(y)
  return z


def compute_centroid(coordinates):
  """Compute compute the x,y,z centroid of provided coordinates

  coordinates: np.ndarray
    Shape (N, 3), where N is number atoms.
  """
  centroid = np.mean(coordinates, axis=0)
  return (centroid)


def generate_random__unit_vector():
  """generate a random unit vector on the 3-sphere
  citation:
  http://mathworld.wolfram.com/SpherePointPicking.html

  a. Choose random theta \element [0, 2*pi]
  b. Choose random z \element [-1, 1]
  c. Compute output: (x,y,z) = (sqrt(1-z^2)*cos(theta), sqrt(1-z^2)*sin(theta),z)
  d. output u
  """

  theta = np.random.uniform(low=0.0, high=2 * np.pi)
  z = np.random.uniform(low=-1.0, high=1.0)
  u = np.array(
      [np.sqrt(1 - z**2) * np.cos(theta),
       np.sqrt(1 - z**2) * np.sin(theta), z])
  return (u)


def generate_random_rotation_matrix():
  """
   1. Generate a random unit vector, i.e., randomly sampled from the unit
      3-sphere
    a. see function _generate_random__unit_vector() for details
    2. Generate a second random unit vector thru the algorithm in (1), output v
    a. If absolute value of u \dot v > 0.99, repeat.
       (This is important for numerical stability. Intuition: we want them to
        be as linearly independent as possible or else the orthogonalized
        version of v will be much shorter in magnitude compared to u. I assume
        in Stack they took this from Gram-Schmidt orthogonalization?)
    b. v" = v - (u \dot v)*u, i.e. subtract out the component of v that's in
       u's direction
    c. normalize v" (this isn"t in Stack but I assume it must be done)
    3. find w = u \cross v"
    4. u, v", and w will form the columns of a rotation matrix, R. The
       intuition is that u, v" and w are, respectively, what the standard basis
       vectors e1, e2, and e3 will be mapped to under the transformation.
  """
  u = generate_random__unit_vector()
  v = generate_random__unit_vector()
  while np.abs(np.dot(u, v)) >= 0.99:
    v = generate_random__unit_vector()

  vp = v - (np.dot(u, v) * u)
  vp /= np.linalg.norm(vp)

  w = np.cross(u, vp)

  R = np.column_stack((u, vp, w))
  return (R)


def rotate_molecules(mol_coordinates_list):
  """Rotates provided molecular coordinates.

  Pseudocode:
  1. Generate random rotation matrix. This matrix applies a random
     transformation to any 3-vector such that, were the random transformation
     repeatedly applied, it would randomly sample along the surface of a sphere
     with radius equal to the norm of the given 3-vector cf.
     _generate_random_rotation_matrix() for details
  2. Apply R to all atomic coordinatse.
  3. Return rotated molecule
  """
  R = generate_random_rotation_matrix()
  rotated_coordinates_list = []

  for mol_coordinates in mol_coordinates_list:
    coordinates = deepcopy(mol_coordinates)
    rotated_coordinates = np.transpose(np.dot(R, np.transpose(coordinates)))
    rotated_coordinates_list.append(rotated_coordinates)

  return (rotated_coordinates_list)


def compute_pairwise_distances(protein_xyz, ligand_xyz):
  """
  Takes an input m x 3 and n x 3 np arrays of 3d coords of protein and ligand,
  respectively, and outputs an m x n np array of pairwise distances in Angstroms
  between protein and ligand atoms. entry (i,j) is dist between the i"th protein
  atom and the j"th ligand atom
  """

  pairwise_distances = np.zeros((np.shape(protein_xyz)[0],
                                 np.shape(ligand_xyz)[0]))
  for j in range(0, np.shape(ligand_xyz)[0]):
    differences = protein_xyz - ligand_xyz[j, :]
    squared_differences = np.square(differences)
    pairwise_distances[:, j] = np.sqrt(np.sum(squared_differences, 1))

  return (pairwise_distances)


"""following two functions adapted from:
http://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python
"""


def unit_vector(vector):
  """ Returns the unit vector of the vector.  """
  return vector / np.linalg.norm(vector)


def angle_between(vector_i, vector_j):
  """Returns the angle in radians between vectors "vector_i" and "vector_j"::

  >>> print("%0.06f" % angle_between((1, 0, 0), (0, 1, 0)))
  1.570796
  >>> print("%0.06f" % angle_between((1, 0, 0), (1, 0, 0)))
  0.000000
  >>> print("%0.06f" % angle_between((1, 0, 0), (-1, 0, 0)))
  3.141593
  """
  vector_i_u = unit_vector(vector_i)
  vector_j_u = unit_vector(vector_j)
  angle = np.arccos(np.dot(vector_i_u, vector_j_u))
  if np.isnan(angle):
    if (vector_i_u == vector_j_u).all():
      return 0.0
    else:
      return np.pi
  return angle


def hash_sybyl(sybyl, sybyl_types):
  return (sybyl_types.index(sybyl))


def hash_ecfp(ecfp, power):
  """
  Returns an int of size 2^power representing that
  ECFP fragment. Input must be a string.
  """
  ecfp = ecfp.encode('utf-8')
  md5 = hashlib.md5()
  md5.update(ecfp)
  digest = md5.hexdigest()
  ecfp_hash = int(digest, 16) % (2**power)
  return (ecfp_hash)


def hash_ecfp_pair(ecfp_pair, power):
  """
  Returns an int of size 2^power representing that
  ECFP pair. Input must be a tuple of strings.
  """
  ecfp = "%s,%s" % (ecfp_pair[0], ecfp_pair[1])
  ecfp = ecfp.encode('utf-8')
  md5 = hashlib.md5()
  md5.update(ecfp)
  digest = md5.hexdigest()
  ecfp_hash = int(digest, 16) % (2**power)
  return (ecfp_hash)


def compute_all_ecfp(mol, indices=None, degree=2):
  """
  For each atom:
    Obtain molecular fragment for all atoms emanating outward to given degree.
    For each fragment, compute SMILES string (for now) and hash to an int.
    Return a dictionary mapping atom index to hashed SMILES.
  """

  ecfp_dict = {}
  for i in range(mol.GetNumAtoms()):
    if indices is not None and i not in indices:
      continue
    env = Chem.FindAtomEnvironmentOfRadiusN(mol, degree, i, useHs=True)
    submol = Chem.PathToSubmol(mol, env)
    smile = Chem.MolToSmiles(submol)
    ecfp_dict[i] = "%s,%s" % (mol.GetAtoms()[i].GetAtomicNum(), smile)

  return ecfp_dict


def compute_ecfp_features(mol, ecfp_degree, ecfp_power):
  """Computes ECFP features for provided rdkit molecule.

  Parameters:
  -----------
    mol: rdkit molecule
      Molecule to featurize.
    ecfp_degree: int
      ECFP radius
    ecfp_power: int
      Number of bits to store ECFP features (2^ecfp_power will be length of
      ECFP array)
  Returns:
  --------
    ecfp_array: np.ndarray
      Returns an array of size 2^ecfp_power where array at index i has a 1 if
      that ECFP fragment is found in the molecule and array at index j has a 0
      if ECFP fragment not in molecule.
  """
  bv = AllChem.GetMorganFingerprint(mol, 2)
  return [int(bv.GetBit(x)) for x in range(bv.GetNumBits())]


def featurize_binding_pocket_ecfp(protein_xyz,
                                  protein,
                                  ligand_xyz,
                                  ligand,
                                  pairwise_distances=None,
                                  cutoff=4.5,
                                  ecfp_degree=2):
  """Computes ECFP dicts for ligand and binding pocket of the protein.

  Parameters
  ----------
  protein_xyz: np.ndarray
    Of shape (N_protein_atoms, 3)
  protein: rdkit.rdchem.Mol
    Contains more metadata.
  ligand_xyz: np.ndarray
    Of shape (N_ligand_atoms, 3)
  ligand: rdkit.rdchem.Mol
    Contains more metadata
  pairwise_distances: np.ndarray 
    Array of pairwise protein-ligand distances (Angstroms) 
  cutoff: float
    Cutoff distance for contact consideration.
  """

  if pairwise_distances is None:
    pairwise_distances = compute_pairwise_distances(protein_xyz, ligand_xyz)
  contacts = np.nonzero((pairwise_distances < cutoff))
  protein_atoms = set([int(c) for c in contacts[0].tolist()])

  protein_ecfp_dict = compute_all_ecfp(
      protein, indices=protein_atoms, degree=ecfp_degree)
  ligand_ecfp_dict = compute_all_ecfp(ligand, degree=ecfp_degree)

  return (protein_ecfp_dict, ligand_ecfp_dict)


def compute_all_sybyl(mol, indices=None):
  """Computes Sybyl atom types for atoms in molecule."""
  raise ValueError("Not Yet Implemented")


def featurize_binding_pocket_sybyl(protein_xyz,
                                   protein,
                                   ligand_xyz,
                                   ligand,
                                   pairwise_distances=None,
                                   cutoff=7.0):
  """Computes Sybyl dicts for ligand and binding pocket of the protein.

  Parameters
  ----------
  protein_xyz: np.ndarray
    Of shape (N_protein_atoms, 3)
  protein: Rdkit Molecule
    Contains more metadata.
  ligand_xyz: np.ndarray
    Of shape (N_ligand_atoms, 3)
  ligand: Rdkit Molecule
    Contains more metadata
  pairwise_distances: np.ndarray 
    Array of pairwise protein-ligand distances (Angstroms) 
  cutoff: float
    Cutoff distance for contact consideration.
  """
  features_dict = {}

  if pairwise_distances is None:
    pairwise_distances = compute_pairwise_distances(protein_xyz, ligand_xyz)
  contacts = np.nonzero((pairwise_distances < cutoff))
  protein_atoms = set([int(c) for c in contacts[0].tolist()])

  protein_sybyl_dict = compute_all_sybyl(protein, indices=protein_atoms)
  ligand_sybyl_dict = compute_all_sybyl(ligand)
  return (protein_sybyl_dict, ligand_sybyl_dict)


def compute_splif_features_in_range(protein,
                                    ligand,
                                    pairwise_distances,
                                    contact_bin,
                                    ecfp_degree=2):
  """Computes SPLIF features for protein atoms close to ligand atoms.

  Find all protein atoms that are > contact_bin[0] and < contact_bin[1] away
  from ligand atoms.  Then, finds the ECFP fingerprints for the contacting
  atoms.  Returns a dictionary mapping (protein_index_i, ligand_index_j) -->
  (protein_ecfp_i, ligand_ecfp_j)
  """
  contacts = np.nonzero((pairwise_distances > contact_bin[0]) &
                        (pairwise_distances < contact_bin[1]))
  protein_atoms = set([int(c) for c in contacts[0].tolist()])
  contacts = zip(contacts[0], contacts[1])

  protein_ecfp_dict = compute_all_ecfp(
      protein, indices=protein_atoms, degree=ecfp_degree)
  ligand_ecfp_dict = compute_all_ecfp(ligand, degree=ecfp_degree)
  splif_dict = {
      contact: (protein_ecfp_dict[contact[0]], ligand_ecfp_dict[contact[1]])
      for contact in contacts
  }
  return (splif_dict)


def featurize_splif(protein_xyz, protein, ligand_xyz, ligand, contact_bins,
                    pairwise_distances, ecfp_degree):
  """Computes SPLIF featurization of protein-ligand binding pocket.

  For each contact range (i.e. 1 A to 2 A, 2 A to 3 A, etc.) compute a
  dictionary mapping (protein_index_i, ligand_index_j) tuples -->
  (protein_ecfp_i, ligand_ecfp_j) tuples.  return a list of such splif
  dictionaries.
  """
  if pairwise_distances is None:
    pairwise_distances = compute_pairwise_distances(protein_xyz, ligand_xyz)
  splif_dicts = []
  for i, contact_bin in enumerate(contact_bins):
    splif_dicts.append(
        compute_splif_features_in_range(protein, ligand, pairwise_distances,
                                        contact_bin, ecfp_degree))

  return (splif_dicts)


def compute_ring_center(mol, ring):
  ring_xyz = np.zeros((len(ring._path), 3))
  for i, atom_idx in enumerate(ring._path):
    atom = mol.GetAtom(int(atom_idx))
    ring_xyz[i, :] = [atom.x(), atom.y(), atom.z()]
  ring_centroid = compute_centroid(ring_xyz)
  return ring_centroid


def compute_ring_normal(mol, ring):
  points = np.zeros((3, 3))
  for i, atom_idx in enumerate(ring._path):
    if i == 3: break
    atom = mol.GetAtom(int(atom_idx))
    points[i, :] = [atom.x(), atom.y(), atom.z()]

  v1 = points[1, :] - points[0, :]
  v2 = points[2, :] - points[0, :]
  normal = np.cross(v1, v2)
  return normal


def is_pi_parallel(protein_ring_center, protein_ring_normal, ligand_ring_center,
                   ligand_ring_normal):
  dist = np.linalg.norm(protein_ring_center - ligand_ring_center)
  angle = angle_between(protein_ring_normal, ligand_ring_normal) * 180 / np.pi
  if ((np.abs(angle) < 30.0) or
      (np.abs(angle) > 150.0 and np.abs(angle) < 210.0) or
      (np.abs(angle) > 330.0 and np.abs(angle) < 360.0)):
    if dist < 8.0:
      return True

  return False


def is_pi_t(protein_ring_center, protein_ring_normal, ligand_ring_center,
            ligand_ring_normal):
  dist = np.linalg.norm(protein_ring_center - ligand_ring_center)
  angle = angle_between(protein_ring_normal, ligand_ring_normal) * 180 / np.pi
  if ((np.abs(angle) > 60.0 and np.abs(angle) < 120.0) or
      (np.abs(angle) > 240.0 and np.abs(angle) < 300.0)):
    if dist < 5.5:
      return True

  return False


def update_feature_dict(feature_dict, idxs=None, indices=None):
  if idxs is not None:
    indices = []
    for idx in idxs:
      indices.append(idx - 1)

  for index in indices:
    if index not in feature_dict.keys():
      feature_dict[index] = 1
    else:
      feature_dict[index] += 1

  return feature_dict


def compute_pi_stack(protein_xyz,
                     protein,
                     ligand_xyz,
                     ligand,
                     pairwise_distances=None,
                     dist_cutoff=4.4,
                     angle_cutoff=30.):
  """
  Pseudocode:

  for each ring in ligand:
    if it is aromatic:
      for each ring in protein:
        if it is aromatic:
          compute distance between centers
          compute angle.
          if it counts as parallel pi-pi:
            for each atom in ligand and in protein,
              add to list of atom indices
          if it counts as pi-T:
            for each atom in ligand and in protein:
              add to list of atom indices
  """
  raise ValueError("Not Yet Implemented")


def is_cation_pi(cation_position, ring_center, ring_normal):
  cation_to_ring_vec = cation_position - ring_center
  dist = np.linalg.norm(cation_to_ring_vec)
  angle = angle_between(cation_to_ring_vec, ring_normal) * 180. / np.pi
  if dist < 6.5:
    if ((np.abs(angle) < 30.0) or
        (np.abs(angle) > 150.0 and np.abs(angle) < 210.0) or
        (np.abs(angle) > 330.0 and np.abs(angle) < 360.0)):
      return True

  return False


def compute_cation_pi(protein, ligand, protein_cation_pi, ligand_cation_pi):
  raise ValueError("Not Yet Implemented")


def compute_binding_pocket_cation_pi(protein_xyz, protein, ligand_xyz, ligand):
  protein_cation_pi = {}
  ligand_cation_pi = {}

  (protein_cation_pi, ligand_cation_pi) = compute_cation_pi(
      protein, ligand, protein_cation_pi, ligand_cation_pi)
  (ligand_cation_pi, protein_cation_pi) = compute_cation_pi(
      ligand, protein, ligand_cation_pi, protein_cation_pi)
  return (protein_cation_pi, ligand_cation_pi)


def get_formal_charge(atom):
  try:
    value = atom.GetProp(str("_GasteigerCharge"))
    if value == '-nan':
      return 0
    return float(value)
  except KeyError:
    return 0


def is_salt_bridge(atom_i, atom_j):
  if np.abs(2.0 - np.abs(get_formal_charge(atom_i) - get_formal_charge(atom_j))
           ) < 0.01:
    return True
  else:
    return False


def compute_salt_bridges(protein_xyz, protein, ligand_xyz, ligand,
                         pairwise_distances):
  salt_bridge_contacts = []

  contacts = np.nonzero(pairwise_distances < 5.0)
  contacts = zip(contacts[0], contacts[1])
  for contact in contacts:
    protein_atom = protein.GetAtoms()[int(contact[0])]
    ligand_atom = ligand.GetAtoms()[int(contact[1])]
    if is_salt_bridge(protein_atom, ligand_atom):
      salt_bridge_contacts.append(contact)
  return salt_bridge_contacts


def is_angle_within_cutoff(vector_i, vector_j, hbond_angle_cutoff):
  angle = angle_between(vector_i, vector_j) * 180. / np.pi
  return (angle > (180 - hbond_angle_cutoff) and
          angle < (180. + hbond_angle_cutoff))


def is_hydrogen_bond(protein_xyz, protein, ligand_xyz, ligand, contact,
                     hbond_angle_cutoff):
  """
  Determine if a pair of atoms (contact = tuple of protein_atom_index, ligand_atom_index)
  between protein and ligand represents a hydrogen bond. Returns a boolean result.
  """

  # TODO(LESWING)
  return False


def compute_hbonds_in_range(protein, protein_xyz, ligand, ligand_xyz,
                            pairwise_distances, hbond_dist_bin,
                            hbond_angle_cutoff, ecfp_degree):
  """
  Find all pairs of (protein_index_i, ligand_index_j) that hydrogen bond given
  a distance bin and an angle cutoff.
  """

  contacts = np.nonzero((pairwise_distances > hbond_dist_bin[0]) &
                        (pairwise_distances < hbond_dist_bin[1]))
  protein_atoms = set([int(c) for c in contacts[0].tolist()])
  protein_ecfp_dict = compute_all_ecfp(
      protein, indices=protein_atoms, degree=ecfp_degree)
  ligand_ecfp_dict = compute_all_ecfp(ligand, degree=ecfp_degree)
  contacts = zip(contacts[0], contacts[1])
  hydrogen_bond_contacts = []
  for contact in contacts:
    if is_hydrogen_bond(protein_xyz, protein, ligand_xyz, ligand, contact,
                        hbond_angle_cutoff):
      hydrogen_bond_contacts.append(contact)
  return hydrogen_bond_contacts


def compute_hydrogen_bonds(protein_xyz, protein, ligand_xyz, ligand,
                           pairwise_distances, hbond_dist_bins,
                           hbond_angle_cutoffs, ecfp_degree):
  """Computes hydrogen bonds between proteins and ligands.

  Returns a list of sublists. Each sublist is a series of tuples of
  (protein_index_i, ligand_index_j) that represent a hydrogen bond. Each sublist
  represents a different type of hydrogen bond.
  """

  hbond_contacts = []
  for i, hbond_dist_bin in enumerate(hbond_dist_bins):
    hbond_angle_cutoff = hbond_angle_cutoffs[i]
    hbond_contacts.append(
        compute_hbonds_in_range(protein, protein_xyz, ligand, ligand_xyz,
                                pairwise_distances, hbond_dist_bin,
                                hbond_angle_cutoff, ecfp_degree))
  return (hbond_contacts)


def convert_atom_to_voxel(molecule_xyz,
                          atom_index,
                          box_width,
                          voxel_width,
                          verbose=False):
  """
  Converts an atom to an i,j,k grid index.
  """
  from warnings import warn

  indices = np.floor(
      (molecule_xyz[atom_index, :] + np.array([box_width, box_width, box_width]
                                             ) / 2.0) / voxel_width).astype(int)
  if ((indices < 0) | (indices >= box_width / voxel_width)).any():
    if verbose:
      warn(
          'Coordinates are outside of the box (atom id = %s, coords xyz = %s, coords in box = %s'
          % (atom_index, molecule_xyz[atom_index], indices))
  return ([indices])


def convert_atom_pair_to_voxel(molecule_xyz_tuple, atom_index_pair, box_width,
                               voxel_width):
  """
  Converts a pair of atoms to a list of i,j,k tuples.
  """
  indices_list = []
  indices_list.append(
      convert_atom_to_voxel(molecule_xyz_tuple[0], atom_index_pair[0],
                            box_width, voxel_width)[0])
  indices_list.append(
      convert_atom_to_voxel(molecule_xyz_tuple[1], atom_index_pair[1],
                            box_width, voxel_width)[0])
  return (indices_list)


def compute_charge_dictionary(molecule):
  """Create a dictionary with partial charges for each atom in the molecule.

  This function assumes that the charges for the molecule are already
  computed (it can be done with rdkit_util.compute_charges(molecule))
  """

  charge_dictionary = {}
  for i, atom in enumerate(molecule.GetAtoms()):
    charge_dictionary[i] = get_formal_charge(atom)
  return charge_dictionary


def subtract_centroid(xyz, centroid):
  """Subtracts centroid from each coordinate.

  Subtracts the centroid, a numpy array of dim 3, from all coordinates of all
  atoms in the molecule
  """

  xyz -= np.transpose(centroid)
  return (xyz)


class RdkitGridFeaturizer(ComplexFeaturizer):

  def __init__(self,
               box_x=16.0,
               box_y=16.0,
               box_z=16.0,
               nb_rotations=0,
               nb_reflections=0,
               feature_types="ecfp",
               ecfp_degree=2,
               ecfp_power=3,
               splif_power=3,
               save_intermediates=False,
               ligand_only=False,
               box_width=16.0,
               voxel_width=1.0,
               voxelize_features=True,
               voxel_feature_types=[],
               flatten=False,
               parallel=False,
               verbose=True,
               **kwargs):
    self.verbose = verbose
    self.parallel = parallel
    self.flatten = flatten

    self.box_x = float(box_x) / 10.0
    self.box_y = float(box_y) / 10.0
    self.box_z = float(box_z) / 10.0

    self.ecfp_degree = ecfp_degree
    self.ecfp_power = ecfp_power
    self.splif_power = splif_power

    self.nb_rotations = nb_rotations
    self.nb_reflections = nb_reflections
    self.feature_types = feature_types

    self.save_intermediates = save_intermediates
    self.ligand_only = ligand_only

    self.hbond_dist_bins = [(2.2, 2.5), (2.5, 3.2), (3.2, 4.0)]
    self.hbond_angle_cutoffs = [5, 50, 90]
    self.contact_bins = [(0, 2.0), (2.0, 3.0), (3.0, 4.5)]

    self.box_width = float(box_width)
    self.voxel_width = float(voxel_width)
    self.voxels_per_edge = self.box_width / self.voxel_width
    self.voxelize_features = voxelize_features
    self.voxel_feature_types = voxel_feature_types

    self.sybyl_types = [
        "C3", "C2", "C1", "Cac", "Car", "N3", "N3+", "Npl", "N2", "N1", "Ng+",
        "Nox", "Nar", "Ntr", "Nam", "Npl3", "N4", "O3", "O-", "O2", "O.co2",
        "O.spc", "O.t3p", "S3", "S3+", "S2", "So2", "Sox"
        "Sac"
        "SO", "P3", "P", "P3+", "F", "Cl", "Br", "I"
    ]

  def _featurize_complex(self, ligand_ext, ligand_lines, protein_pdb_lines):
    tempdir = tempfile.mkdtemp()

    ############################################################## TIMING
    time1 = time.time()
    ############################################################## TIMING
    ligand_file = os.path.join(tempdir, "ligand.%s" % ligand_ext)
    with open(ligand_file, "w") as mol_f:
      mol_f.writelines(ligand_lines)
    ############################################################## TIMING
    time2 = time.time()
    log("TIMING: Writing ligand took %0.3f s" % (time2 - time1), self.verbose)
    ############################################################## TIMING

    ############################################################## TIMING
    time1 = time.time()
    ############################################################## TIMING
    protein_pdb_file = os.path.join(tempdir, "protein.pdb")
    with open(protein_pdb_file, "w") as protein_f:
      protein_f.writelines(protein_pdb_lines)
    ############################################################## TIMING
    time2 = time.time()
    log("TIMING: Writing protein took %0.3f s" % (time2 - time1), self.verbose)
    ############################################################## TIMING

    features_dict = self._transform(protein_pdb_file, ligand_file)
    shutil.rmtree(tempdir)
    return features_dict.values()

  def featurize_complexes(self, mol_files, protein_pdbs, log_every_n=1000):
    """
    Calculate features for mol/protein complexes.

    Parameters
    ----------
    mols: list
      List of PDB filenames for molecules.
    protein_pdbs: list
      List of PDB filenames for proteins.
    """
    features = []
    for i, (mol_file, protein_pdb) in enumerate(zip(mol_files, protein_pdbs)):
      if i % log_every_n == 0:
        log("Featurizing %d / %d" % (i, len(mol_files)))
      ligand_ext = get_ligand_filetype(mol_file)
      with open(mol_file) as mol_f:
        mol_lines = mol_f.readlines()
      with open(protein_pdb) as protein_file:
        protein_pdb_lines = protein_file.readlines()
      features += self._featurize_complex(ligand_ext, mol_lines,
                                          protein_pdb_lines)
    features = np.asarray(features)
    return features

  def _transform(self, protein_pdb, ligand_file):
    """Computes featurization of protein/ligand complex.

    Takes as input files (strings) for pdb of the protein, pdb of the ligand,
    and a directory to save intermediate files.

    This function then computes the centroid of the ligand; decrements this
    centroid from the atomic coordinates of protein and ligand atoms, and then
    merges the translated protein and ligand. This combined system/complex is then
    saved.

    This function then computes a featurization with scheme specified by the user.
    """
    ############################################################## TIMING
    time1 = time.time()
    ############################################################## TIMING
    protein_name = str(protein_pdb).split("/")[len(str(protein_pdb).split("/"))
                                               - 2]

    if not self.ligand_only:
      protein_xyz, protein_ob = rdkit_util.load_molecule(
          protein_pdb, calc_charges=True)
    ############################################################## TIMING
    time2 = time.time()
    log("TIMING: Loading protein coordinates took %0.3f s" % (time2 - time1),
        self.verbose)
    ############################################################## TIMING
    ############################################################## TIMING
    time1 = time.time()
    ############################################################## TIMING
    ligand_xyz, ligand_ob = rdkit_util.load_molecule(
        ligand_file, calc_charges=True)
    ############################################################## TIMING
    time2 = time.time()
    log("TIMING: Loading ligand coordinates took %0.3f s" % (time2 - time1),
        self.verbose)
    ############################################################## TIMING

    if "ecfp" in self.feature_types:
      ecfp_array = compute_ecfp_features(ligand_ob, self.ecfp_degree,
                                         self.ecfp_power)
      return ({(0, 0): ecfp_array})

    ############################################################## TIMING
    time1 = time.time()
    ############################################################## TIMING
    centroid = compute_centroid(ligand_xyz)
    ligand_xyz = subtract_centroid(ligand_xyz, centroid)
    if not self.ligand_only:
      protein_xyz = subtract_centroid(protein_xyz, centroid)
    ############################################################## TIMING
    time2 = time.time()
    log("TIMING: Centroid processing took %0.3f s" % (time2 - time1),
        self.verbose)
    ############################################################## TIMING

    if "splif" in self.feature_types:
      splif_array = self._featurize_splif(protein_xyz, protein_ob, ligand_xyz,
                                          ligand_ob)
      return ({(0, 0): splif_array})

    if "flat_combined" in self.feature_types:
      return (self._compute_flat_features(protein_xyz, protein_ob, ligand_xyz,
                                          ligand_ob))

    pairwise_distances = compute_pairwise_distances(protein_xyz, ligand_xyz)
    if "ecfp" in self.voxel_feature_types:
      ############################################################## TIMING
      time1 = time.time()
      ############################################################## TIMING
      protein_ecfp_dict, ligand_ecfp_dict = (featurize_binding_pocket_ecfp(
          protein_xyz,
          protein_ob,
          ligand_xyz,
          ligand_ob,
          pairwise_distances,
          cutoff=4.5,
          ecfp_degree=self.ecfp_degree))
      ############################################################## TIMING
      time2 = time.time()
      log("TIMING: ecfp voxel computataion took %0.3f s" % (time2 - time1),
          self.verbose)
      ############################################################## TIMING
    if "splif" in self.voxel_feature_types:
      ############################################################## TIMING
      time1 = time.time()
      ############################################################## TIMING
      splif_dicts = featurize_splif(protein_xyz, protein_ob, ligand_xyz,
                                    ligand_ob, self.contact_bins,
                                    pairwise_distances, self.ecfp_degree)
      ############################################################## TIMING
      time2 = time.time()
      log("TIMING: splif voxel computataion took %0.3f s" % (time2 - time1),
          self.verbose)
      ############################################################## TIMING

    if "hbond" in self.voxel_feature_types:
      ############################################################## TIMING
      time1 = time.time()
      ############################################################## TIMING
      hbond_list = compute_hydrogen_bonds(
          protein_xyz, protein_ob, ligand_xyz, ligand_ob, pairwise_distances,
          self.hbond_dist_bins, self.hbond_angle_cutoffs, self.ecfp_degree)
      ############################################################## TIMING
      time2 = time.time()
      log("TIMING: hbond voxel computataion took %0.3f s" % (time2 - time1),
          self.verbose)
      ############################################################## TIMING

    if "sybyl" in self.voxel_feature_types:
      ############################################################## TIMING
      time1 = time.time()
      ############################################################## TIMING
      protein_sybyl_dict, ligand_sybyl_dict = featurize_binding_pocket_sybyl(
          protein_xyz,
          protein_ob,
          ligand_xyz,
          ligand_ob,
          pairwise_distances,
          cutoff=7.0)
      ############################################################## TIMING
      time2 = time.time()
      log("TIMING: sybyl voxel computataion took %0.3f s" % (time2 - time1),
          self.verbose)
      ############################################################## TIMING

    if "pi_stack" in self.voxel_feature_types:
      ############################################################## TIMING
      time1 = time.time()
      ############################################################## TIMING
      protein_pi_t, protein_pi_parallel, ligand_pi_t, ligand_pi_parallel = (
          compute_pi_stack(protein_xyz, protein_ob, ligand_xyz, ligand_ob,
                           pairwise_distances))
      ############################################################## TIMING
      time2 = time.time()
      log("TIMING: pi_stack voxel computataion took %0.3f s" % (time2 - time1),
          self.verbose)
      ############################################################## TIMING

    if "cation_pi" in self.voxel_feature_types:
      ############################################################## TIMING
      time1 = time.time()
      ############################################################## TIMING
      protein_cation_pi, ligand_cation_pi = (compute_binding_pocket_cation_pi(
          protein_xyz, protein_ob, ligand_xyz, ligand_ob))
      ############################################################## TIMING
      time2 = time.time()
      log("TIMING: cation_pi voxel computataion took %0.3f s" % (time2 - time1),
          self.verbose)
      ############################################################## TIMING

    if "salt_bridge" in self.voxel_feature_types:
      ############################################################## TIMING
      time1 = time.time()
      ############################################################## TIMING
      salt_bridge_list = compute_salt_bridges(
          protein_xyz, protein_ob, ligand_xyz, ligand_ob, pairwise_distances)
      ############################################################## TIMING
      time2 = time.time()
      log("TIMING: salt_bridge voxel computataion took %0.3f s" %
          (time2 - time1), self.verbose)
      ############################################################## TIMING

    if "charge" in self.voxel_feature_types:
      ############################################################## TIMING
      time1 = time.time()
      ############################################################## TIMING
      protein_charge_dictionary = compute_charge_dictionary(protein_ob)
      ligand_charge_dictionary = compute_charge_dictionary(ligand_ob)
      ############################################################## TIMING
      time2 = time.time()
      log("TIMING: charge voxel computataion took %0.3f s" % (time2 - time1),
          self.verbose)
      ############################################################## TIMING

    transformed_systems = {}
    transformed_systems[(0, 0)] = [protein_xyz, ligand_xyz]

    for i in range(0, int(self.nb_rotations)):
      rotated_system = rotate_molecules([protein_xyz, ligand_xyz])
      transformed_systems[(i + 1, 0)] = rotated_system
      for j in range(0, int(self.nb_reflections)):
        reflected_system = self._reflect_molecule(rotated_system)
        transformed_systems[(i + 1, j + 1)] = reflected_system

    if "voxel_combined" in self.feature_types:
      features = {}
      for system_id, system in transformed_systems.items():
        protein_xyz = system[0]
        ligand_xyz = system[1]
        feature_tensors = []
        if "ecfp" in self.voxel_feature_types:
          ecfp_tensor = self._voxelize(
              convert_atom_to_voxel,
              hash_ecfp,
              protein_xyz,
              feature_dict=protein_ecfp_dict,
              channel_power=self.ecfp_power)
          ecfp_tensor += self._voxelize(
              convert_atom_to_voxel,
              hash_ecfp,
              ligand_xyz,
              feature_dict=ligand_ecfp_dict,
              channel_power=self.ecfp_power)
          feature_tensors.append(ecfp_tensor)
          print("Completed ecfp tensor")

        if "splif" in self.voxel_feature_types:
          feature_tensors += [
              self._voxelize(
                  convert_atom_pair_to_voxel,
                  hash_ecfp_pair, (protein_xyz, ligand_xyz),
                  feature_dict=splif_dict,
                  channel_power=self.splif_power) for splif_dict in splif_dicts
          ]
          print("Completed splif tensor")

        if "hbond" in self.voxel_feature_types:
          feature_tensors += [
              self._voxelize(
                  convert_atom_pair_to_voxel,
                  None, (protein_xyz, ligand_xyz),
                  feature_list=hbond,
                  channel_power=0) for hbond in hbond_list
          ]
          print("Completed hbond tensor")

        if "sybyl" in self.voxel_feature_types:
          sybyl_partial = partial(hash_sybyl, sybyl_types=self.sybyl_types)
          sybyl_tensor = self._voxelize(
              convert_atom_to_voxel,
              hash_sybyl,
              protein_xyz,
              feature_dict=protein_sybyl_dict,
              nb_channel=len(self.sybyl_types))
          sybyl_tensor += self._voxelize(
              convert_atom_to_voxel,
              hash_sybyl,
              ligand_xyz,
              feature_dict=ligand_sybyl_dict,
              nb_channel=len(self.sybyl_types))
          feature_tensors.append(sybyl_tensor)
          print("Completed sybyl tensor")

        if "pi_stack" in self.voxel_feature_types:
          pi_parallel_tensor = self._voxelize(
              convert_atom_to_voxel,
              None,
              protein_xyz,
              feature_dict=protein_pi_parallel,
              nb_channel=1)
          pi_parallel_tensor += self._voxelize(
              convert_atom_to_voxel,
              None,
              ligand_xyz,
              feature_dict=ligand_pi_parallel,
              nb_channel=1)
          feature_tensors.append(pi_parallel_tensor)

          pi_t_tensor = self._voxelize(
              convert_atom_to_voxel,
              None,
              protein_xyz,
              feature_dict=protein_pi_t,
              nb_channel=1)
          pi_t_tensor += self._voxelize(
              convert_atom_to_voxel,
              None,
              ligand_xyz,
              feature_dict=ligand_pi_t,
              nb_channel=1)
          feature_tensors.append(pi_t_tensor)
          print("Completed pi_stack tensor")

        if "cation_pi" in self.voxel_feature_types:
          cation_pi_tensor = self._voxelize(
              convert_atom_to_voxel,
              None,
              protein_xyz,
              feature_dict=protein_cation_pi,
              nb_channel=1)
          cation_pi_tensor += self._voxelize(
              convert_atom_to_voxel,
              None,
              ligand_xyz,
              feature_dict=ligand_cation_pi,
              nb_channel=1)
          feature_tensors.append(cation_pi_tensor)
          print("Completed cation_pi tensor.")

        if "salt_bridge" in self.voxel_feature_types:
          salt_bridge_tensor = self._voxelize(
              convert_atom_pair_to_voxel,
              None, (protein_xyz, ligand_xyz),
              feature_list=salt_bridge_list,
              nb_channel=1)
          feature_tensors.append(salt_bridge_tensor)

          print("Completed salt_bridge tensor.")

        if "charge" in self.voxel_feature_types:
          charge_tensor = self._voxelize(
              convert_atom_to_voxel,
              None,
              protein_xyz,
              feature_dict=protein_charge_dictionary,
              nb_channel=1,
              dtype="np.float16")
          charge_tensor += self._voxelize(
              convert_atom_to_voxel,
              None,
              ligand_xyz,
              feature_dict=ligand_charge_dictionary,
              nb_channel=1,
              dtype="np.float16")
          feature_tensors.append(charge_tensor)

          print("Completed salt_bridge tensor.")

        if "charge" in self.voxel_feature_types:
          feature_tensor = np.concatenate(
              feature_tensors, axis=3).astype(np.float16)
        else:
          feature_tensor = np.concatenate(
              feature_tensors, axis=3).astype(np.int8)

        if self.flatten:
          feature_tensor = np.squeeze(feature_tensor)

        features[system_id] = feature_tensor

      return (features)

  def _voxelize(self,
                get_voxels,
                hash_function,
                coordinates,
                feature_dict=None,
                feature_list=None,
                channel_power=None,
                nb_channel=16,
                dtype="np.int8"):

    if channel_power is not None:
      if channel_power == 0:
        nb_channel = 1
      else:
        nb_channel = int(2**channel_power)
    if dtype == "np.int8":
      feature_tensor = np.zeros(
          (self.voxels_per_edge, self.voxels_per_edge, self.voxels_per_edge,
           nb_channel),
          dtype=np.int8)
    else:
      feature_tensor = np.zeros(
          (self.voxels_per_edge, self.voxels_per_edge, self.voxels_per_edge,
           nb_channel),
          dtype=np.float16)
    if feature_dict is not None:
      for key, features in feature_dict.items():
        voxels = get_voxels(coordinates, key, self.box_width, self.voxel_width)
        for voxel in voxels:
          if ((voxel >= 0) & (voxel < self.voxels_per_edge)).all():
            if hash_function is not None:
              feature_tensor[voxel[0], voxel[1], voxel[2],
                             hash_function(features, channel_power)] += 1.0
            else:
              feature_tensor[voxel[0], voxel[1], voxel[3], 0] += features
    elif feature_list is not None:
      for key in feature_list:
        voxels = get_voxels(coordinates, key, self.box_width, self.voxel_width)
        for voxel in voxels:
          if ((voxel >= 0) & (voxel < self.voxels_per_edge)).all():
            feature_tensor[voxel[0], voxel[1], voxel[2], 0] += 1.0

    return feature_tensor

  def _vectorize(self,
                 hash_function,
                 feature_dict=None,
                 feature_list=None,
                 channel_power=10):
    feature_vector = np.zeros(2**channel_power)
    if feature_dict is not None:
      on_channels = [
          hash_function(feature, channel_power)
          for key, feature in feature_dict.items()
      ]
      feature_vector[on_channels] += 1
    elif feature_list is not None:
      feature_vector[0] += len(feature_list)

    return feature_vector

  def _generate_box(self, mol):
    """Removes atoms outside box.

    Generate_box takes as input a molecule of class PDB and removes all atoms
    outside of the given box dims
    """

    molecule = deepcopy(mol)
    atoms_to_keep = []
    all_atoms = [a for a in molecule.topology.atoms]
    for atom in all_atoms:
      coords = np.abs(molecule.xyz[0][atom.index, :])
      if (coords[0] <= (self.box_x / 2.) and coords[1] <= (self.box_y / 2.) and
          coords[2] <= (self.box_z / 2.)):
        atoms_to_keep.append(atom.index)
    return (molecule.atom_slice(atoms_to_keep))

  def _compute_flat_features(self, protein_xyz, protein_ob, ligand_xyz,
                             ligand_ob):
    """Computes vectorial (as opposed to tensorial) featurization."""
    pairwise_distances = compute_pairwise_distances(protein_xyz, ligand_xyz)
    protein_ecfp_dict, ligand_ecfp_dict = featurize_binding_pocket_ecfp(
        protein_xyz,
        protein_ob,
        ligand_xyz,
        ligand_ob,
        pairwise_distances,
        cutoff=4.5,
        ecfp_degree=self.ecfp_degree)
    splif_dicts = featurize_splif(protein_xyz, protein_ob, ligand_xyz,
                                  ligand_ob, self.contact_bins,
                                  pairwise_distances, self.ecfp_degree)
    hbond_list = compute_hydrogen_bonds(
        protein_xyz, protein_ob, ligand_xyz, ligand_ob, pairwise_distances,
        self.hbond_dist_bins, self.hbond_angle_cutoffs, self.ecfp_degree)

    protein_ecfp_vector = [
        self._vectorize(
            hash_ecfp,
            feature_dict=protein_ecfp_dict,
            channel_power=self.ecfp_power)
    ]
    ligand_ecfp_vector = [
        self._vectorize(
            hash_ecfp,
            feature_dict=ligand_ecfp_dict,
            channel_power=self.ecfp_power)
    ]
    splif_vectors = [
        self._vectorize(
            hash_ecfp_pair,
            feature_dict=splif_dict,
            channel_power=self.splif_power) for splif_dict in splif_dicts
    ]
    hbond_vectors = [
        self._vectorize(
            hash_ecfp_pair, feature_list=hbond_list, channel_power=0)
        for hbond_class in hbond_list
    ]
    feature_vectors = protein_ecfp_vector + \
                      ligand_ecfp_vector + splif_vectors + hbond_vectors
    feature_vector = np.concatenate(feature_vectors, axis=0)

    return ({(0, 0): feature_vector})
