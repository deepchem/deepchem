"""
Computes physiochemical descriptors which summarize a 3D molecular complex.
"""
import logging
import os
import shutil
import time
import hashlib
import multiprocessing
from collections import Counter
import numpy as np
from warnings import warn
from collections import Counter
from deepchem.utils.rdkit_util import load_molecule
from deepchem.utils.rdkit_util import compute_centroid
from deepchem.utils.rdkit_util import subtract_centroid
from deepchem.utils.rdkit_util import compute_ring_center
from deepchem.utils.rdkit_util import rotate_molecules
from deepchem.utils.rdkit_util import get_partial_charge
from deepchem.utils.rdkit_util import compute_pairwise_distances
from deepchem.utils.rdkit_util import is_salt_bridge
from deepchem.utils.rdkit_util import compute_salt_bridges
from deepchem.utils.rdkit_util import compute_pi_stack
from deepchem.utils.rdkit_util import compute_binding_pocket_cation_pi
from deepchem.utils.rdkit_util import is_hydrogen_bond
from deepchem.utils.rdkit_util import MoleculeLoadException
from deepchem.feat import ComplexFeaturizer

logger = logging.getLogger(__name__)
"""following two functions adapted from:
http://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python
"""

sybyl_types = [
    "C3", "C2", "C1", "Cac", "Car", "N3", "N3+", "Npl", "N2", "N1", "Ng+",
    "Nox", "Nar", "Ntr", "Nam", "Npl3", "N4", "O3", "O-", "O2", "O.co2",
    "O.spc", "O.t3p", "S3", "S3+", "S2", "So2", "Sox"
    "Sac"
    "SO", "P3", "P", "P3+", "F", "Cl", "Br", "I"
]


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
  """Returns an int of size 2^power representing that ECFP pair. Input must be
  a tuple of strings.
  """
  ecfp = "%s,%s" % (ecfp_pair[0], ecfp_pair[1])
  ecfp = ecfp.encode('utf-8')
  md5 = hashlib.md5()
  md5.update(ecfp)
  digest = md5.hexdigest()
  ecfp_hash = int(digest, 16) % (2**power)
  return (ecfp_hash)


def _vectorize(hash_function,
               feature_dict=None,
               feature_list=None,
               channel_power=10):
  """Helper function to vectorize a spatial description."""
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


def compute_all_ecfp(mol, indices=None, degree=2):
  """Obtain molecular fragment for all atoms emanating outward to given degree.
  For each fragment, compute SMILES string (for now) and hash to an int.
  Return a dictionary mapping atom index to hashed SMILES.
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


# TODO(rbharath): Why not just use dc.feat.CircularFingerprint? This seems unne
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
  bv = AllChem.GetMorganFingerprintAsBitVect(
      mol, ecfp_degree, nBits=2**ecfp_power)
  return np.array(bv)


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
    Cutoff distance for contact consideration
  ecfp_degree: int
    ECFP radius
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
  raise NotImplementedError("This function is not implemented yet")


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

  Finds all protein atoms that are > contact_bin[0] and <
  contact_bin[1] away from ligand atoms. Then, finds the ECFP
  fingerprints for the contacting atoms. Returns a dictionary
  mapping (protein_index_i, ligand_index_j) --> (protein_ecfp_i,
  ligand_ecfp_j)
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


# TODO(rbharath): Should this be a featurizer class?
def featurize_splif(protein_xyz, protein, ligand_xyz, ligand, contact_bins,
                    pairwise_distances, ecfp_degree):
  """Computes SPLIF featurization of protein-ligand binding pocket.

  For each contact range (i.e. 1 A to 2 A, 2 A to 3 A, etc.)
  compute a dictionary mapping (protein_index_i, ligand_index_j)
  tuples --> (protein_ecfp_i, ligand_ecfp_j) tuples. Return a
  list of such splif dictionaries.
  """
  splif_dicts = []
  for i, contact_bin in enumerate(contact_bins):
    splif_dicts.append(
        compute_splif_features_in_range(protein, ligand, pairwise_distances,
                                        contact_bin, ecfp_degree))

  return (splif_dicts)


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
                                hbond_angle_cutoff))
  return (hbond_contacts)


def convert_atom_to_voxel(molecule_xyz, atom_index, box_width, voxel_width):
  """Converts atom coordinates to an i,j,k grid index.

  Parameters:
  -----------
  molecule_xyz: np.ndarray
    Array with coordinates of all atoms in the molecule, shape (N, 3)
  atom_index: int
    Index of an atom
  box_width: float
    Size of a box
  voxel_width: float
    Size of a voxel
  """

  indices = np.floor(
      (molecule_xyz[atom_index] + box_width / 2.0) / voxel_width).astype(int)
  if ((indices < 0) | (indices >= box_width / voxel_width)).any():
    logger.warning('Coordinates are outside of the box (atom id = %s,'
                   ' coords xyz = %s, coords in box = %s' %
                   (atom_index, molecule_xyz[atom_index], indices))

  return ([indices])


def convert_atom_pair_to_voxel(molecule_xyz_tuple, atom_index_pair, box_width,
                               voxel_width):
  """Converts a pair of atoms to a list of i,j,k tuples."""

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
    charge_dictionary[i] = get_partial_charge(atom)
  return charge_dictionary


def subtract_centroid(xyz, centroid):
  """Subtracts centroid from each coordinate.

  Subtracts the centroid, a numpy array of dim 3, from all coordinates of all
  atoms in the molecule
  """

  xyz -= np.transpose(centroid)
  return (xyz)


class RdkitGridFeaturizer(ComplexFeaturizer):
  """Featurizes protein-ligand complex using flat features or a 3D grid (in which
  each voxel is described with a vector of features).
  """

  def __init__(self,
               nb_rotations=0,
               feature_types=None,
               ecfp_degree=2,
               ecfp_power=3,
               splif_power=3,
               box_width=16.0,
               voxel_width=1.0,
               flatten=False,
               sanitize=True,
               **kwargs):
    """
    Parameters
    ----------
    nb_rotations: int, optional (default 0)
      Number of additional random rotations of a complex to generate.
    feature_types: list, optional (default ['ecfp'])
      Types of features to calculate. Available types are
        flat features -> 'ecfp_ligand', 'ecfp_hashed', 'splif_hashed', 'hbond_count'
        voxel features -> 'ecfp', 'splif', 'sybyl', 'salt_bridge', 'charge', 'hbond', 'pi_stack, 'cation_pi'
      There are also 3 predefined sets of features
        'flat_combined', 'voxel_combined', and 'all_combined'.
      Calculated features are concatenated and their order is preserved
      (features in predefined sets are in alphabetical order).
    ecfp_degree: int, optional (default 2)
      ECFP radius.
    ecfp_power: int, optional (default 3)
      Number of bits to store ECFP features (resulting vector will be
      2^ecfp_power long)
    splif_power: int, optional (default 3)
      Number of bits to store SPLIF features (resulting vector will be
      2^splif_power long)
    box_width: float, optional (default 16.0)
      Size of a box in which voxel features are calculated. Box is centered on a
      ligand centroid.
    voxel_width: float, optional (default 1.0)
      Size of a 3D voxel in a grid.
    flatten: bool, optional (defaul False)
      Indicate whether calculated features should be flattened. Output is always
      flattened if flat features are specified in feature_types.
    sanitize: bool, optional (default True)
      If set to True molecules will be sanitized. Note that
      calculating some features (e.g. aromatic interactions)
      require sanitized molecules.
    **kwargs: dict, optional
      Keyword arguments can be usaed to specify custom cutoffs and bins (see
      default values below).

    Default cutoffs and bins
    ------------------------
    hbond_dist_bins: [(2.2, 2.5), (2.5, 3.2), (3.2, 4.0)]
    hbond_angle_cutoffs: [5, 50, 90]
    splif_contact_bins: [(0, 2.0), (2.0, 3.0), (3.0, 4.5)]
    ecfp_cutoff: 4.5
    sybyl_cutoff: 7.0
    salt_bridges_cutoff: 5.0
    pi_stack_dist_cutoff: 4.4
    pi_stack_angle_cutoff: 30.0
    cation_pi_dist_cutoff: 6.5
    cation_pi_angle_cutoff: 30.0
    """

    ## check if user tries to set removed arguments
    #deprecated_args = [
    #    'box_x', 'box_y', 'box_z', 'save_intermediates', 'voxelize_features',
    #    'parallel', 'voxel_feature_types'
    #]

    # list of features that require sanitized molecules
    require_sanitized = ['pi_stack', 'cation_pi', 'ecfp_ligand']

    # not implemented featurization types
    not_implemented = ['sybyl']

    self.sanitize = sanitize
    self.flatten = flatten

    self.ecfp_degree = ecfp_degree
    self.ecfp_power = ecfp_power
    self.splif_power = splif_power

    self.nb_rotations = nb_rotations

    # default values
    self.cutoffs = {
        'hbond_dist_bins': [(2.2, 2.5), (2.5, 3.2), (3.2, 4.0)],
        'hbond_angle_cutoffs': [5, 50, 90],
        'splif_contact_bins': [(0, 2.0), (2.0, 3.0), (3.0, 4.5)],
        'ecfp_cutoff': 4.5,
        'sybyl_cutoff': 7.0,
        'salt_bridges_cutoff': 5.0,
        'pi_stack_dist_cutoff': 4.4,
        'pi_stack_angle_cutoff': 30.0,
        'cation_pi_dist_cutoff': 6.5,
        'cation_pi_angle_cutoff': 30.0,
    }

    # update with cutoffs specified by the user
    for arg, value in kwargs.items():
      if arg in self.cutoffs:
        self.cutoffs[arg] = value

    self.box_width = float(box_width)
    self.voxel_width = float(voxel_width)
    self.voxels_per_edge = int(self.box_width / self.voxel_width)

    self.FLAT_FEATURES = [
        'ecfp_ligand', 'ecfp_hashed', 'splif_hashed', 'hbond_count'
    ]
    self.VOXEL_FEATURES = [
        'ecfp', 'splif', 'sybyl', 'salt_bridge', 'charge', 'hbond', 'pi_stack',
        'cation_pi'
    ]

    if feature_types is None:
      feature_types = ['ecfp']

    # each entry is a tuple (is_flat, feature_name)
    self.feature_types = []

    # list of features that cannot be calculated with specified
    # parameters this list is used to define
    # <flat/voxel/all>_combined subset
    ignored_features = []
    if self.sanitize is False:
      ignored_features += require_sanitized
    ignored_features += not_implemented

    # parse provided feature types
    for feature_type in feature_types:
      if self.sanitize is False and feature_type in require_sanitized:
        logger.warning('sanitize is set to False, %s feature will be ignored' %
                       feature_type)
        continue
      if feature_type in not_implemented:
        logger.warning('%s feature is not implemented yet and will be ignored' %
                       feature_type)
        continue

      if feature_type in self.FLAT_FEATURES:
        self.feature_types.append((True, feature_type))
        if self.flatten is False:
          logger.warning(
              '%s feature is used, output will be flattened' % feature_type)
          self.flatten = True

      elif feature_type in self.VOXEL_FEATURES:
        self.feature_types.append((False, feature_type))

      elif feature_type == 'flat_combined':
        self.feature_types += [(True, ftype)
                               for ftype in sorted(self.FLAT_FEATURES)
                               if ftype not in ignored_features]
        if self.flatten is False:
          logger.warning('Flat features are used, output will be flattened')
          self.flatten = True

      elif feature_type == 'voxel_combined':
        self.feature_types += [(False, ftype)
                               for ftype in sorted(self.VOXEL_FEATURES)
                               if ftype not in ignored_features]
      elif feature_type == 'all_combined':
        self.feature_types += [(True, ftype)
                               for ftype in sorted(self.FLAT_FEATURES)
                               if ftype not in ignored_features]
        self.feature_types += [(False, ftype)
                               for ftype in sorted(self.VOXEL_FEATURES)
                               if ftype not in ignored_features]
        if self.flatten is False:
          logger.warning('Flat feature are used, output will be flattened')
          self.flatten = True
      else:
        logger.warning('Ignoring unknown feature %s' % feature_type)

  def _compute_feature(self, feature_name, prot_xyz, prot_rdk, lig_xyz, lig_rdk,
                       distances):
    if feature_name == 'ecfp_ligand':
      return [compute_ecfp_features(lig_rdk, self.ecfp_degree, self.ecfp_power)]
    if feature_name == 'ecfp_hashed':
      return [
          _vectorize(
              hash_ecfp, feature_dict=ecfp_dict, channel_power=self.ecfp_power)
          for ecfp_dict in featurize_binding_pocket_ecfp(
              prot_xyz,
              prot_rdk,
              lig_xyz,
              lig_rdk,
              distances,
              cutoff=self.cutoffs['ecfp_cutoff'],
              ecfp_degree=self.ecfp_degree)
      ]
    if feature_name == 'splif_hashed':
      return [
          _vectorize(
              hash_ecfp_pair,
              feature_dict=splif_dict,
              channel_power=self.splif_power) for splif_dict in featurize_splif(
                  prot_xyz, prot_rdk, lig_xyz, lig_rdk, self.cutoffs[
                      'splif_contact_bins'], distances, self.ecfp_degree)
      ]
    if feature_name == 'hbond_count':
      return [
          _vectorize(hash_ecfp_pair, feature_list=hbond_list, channel_power=0)
          for hbond_list in compute_hydrogen_bonds(
              prot_xyz, prot_rdk, lig_xyz, lig_rdk, distances, self.cutoffs[
                  'hbond_dist_bins'], self.cutoffs['hbond_angle_cutoffs'])
      ]
    if feature_name == 'ecfp':
      return [
          sum([
              self._voxelize(
                  convert_atom_to_voxel,
                  hash_ecfp,
                  xyz,
                  feature_dict=ecfp_dict,
                  channel_power=self.ecfp_power)
              for xyz, ecfp_dict in zip((prot_xyz, lig_xyz),
                                        featurize_binding_pocket_ecfp(
                                            prot_xyz,
                                            prot_rdk,
                                            lig_xyz,
                                            lig_rdk,
                                            distances,
                                            cutoff=self.cutoffs['ecfp_cutoff'],
                                            ecfp_degree=self.ecfp_degree))
          ])
      ]
    if feature_name == 'splif':
      return [
          self._voxelize(
              convert_atom_pair_to_voxel,
              hash_ecfp_pair, (prot_xyz, lig_xyz),
              feature_dict=splif_dict,
              channel_power=self.splif_power) for splif_dict in featurize_splif(
                  prot_xyz, prot_rdk, lig_xyz, lig_rdk, self.cutoffs[
                      'splif_contact_bins'], distances, self.ecfp_degree)
      ]
    if feature_name == 'sybyl':
      return [
          self._voxelize(
              convert_atom_to_voxel,
              lambda x: hash_sybyl(x, sybyl_types=sybyl_types),
              xyz,
              feature_dict=sybyl_dict,
              nb_channel=len(sybyl_types))
          for xyz, sybyl_dict in zip((prot_xyz, lig_xyz),
                                     featurize_binding_pocket_sybyl(
                                         prot_xyz,
                                         prot_rdk,
                                         lig_xyz,
                                         lig_rdk,
                                         distances,
                                         cutoff=self.cutoffs['sybyl_cutoff']))
      ]
    if feature_name == 'salt_bridge':
      return [
          self._voxelize(
              convert_atom_pair_to_voxel,
              None, (prot_xyz, lig_xyz),
              feature_list=compute_salt_bridges(
                  prot_rdk,
                  lig_rdk,
                  distances,
                  cutoff=self.cutoffs['salt_bridges_cutoff']),
              nb_channel=1)
      ]
    if feature_name == 'charge':
      return [
          sum([
              self._voxelize(
                  convert_atom_to_voxel,
                  None,
                  xyz,
                  feature_dict=compute_charge_dictionary(mol),
                  nb_channel=1,
                  dtype="np.float16")
              for xyz, mol in ((prot_xyz, prot_rdk), (lig_xyz, lig_rdk))
          ])
      ]
    if feature_name == 'hbond':
      return [
          self._voxelize(
              convert_atom_pair_to_voxel,
              None, (prot_xyz, lig_xyz),
              feature_list=hbond_list,
              channel_power=0) for hbond_list in compute_hydrogen_bonds(
                  prot_xyz, prot_rdk, lig_xyz, lig_rdk, distances, self.cutoffs[
                      'hbond_dist_bins'], self.cutoffs['hbond_angle_cutoffs'])
      ]
    if feature_name == 'pi_stack':
      return self._voxelize_pi_stack(prot_xyz, prot_rdk, lig_xyz, lig_rdk,
                                     distances)
    if feature_name == 'cation_pi':
      return [
          sum([
              self._voxelize(
                  convert_atom_to_voxel,
                  None,
                  xyz,
                  feature_dict=cation_pi_dict,
                  nb_channel=1) for xyz, cation_pi_dict in zip(
                      (prot_xyz, lig_xyz),
                      compute_binding_pocket_cation_pi(
                          prot_rdk,
                          lig_rdk,
                          dist_cutoff=self.cutoffs['cation_pi_dist_cutoff'],
                          angle_cutoff=self.cutoffs['cation_pi_angle_cutoff'],
                      ))
          ])
      ]
    raise ValueError('Unknown feature type "%s"' % feature_name)

  def _featurize(self, mol_pdb_file, protein_pdb_file):
    """Computes grid featurization of protein/ligand complex.

    Takes as input filenames pdb of the protein, pdb of the ligand.

    This function then computes the centroid of the ligand; decrements this
    centroid from the atomic coordinates of protein and ligand atoms, and then
    merges the translated protein and ligand. This combined system/complex is then
    saved.

    This function then computes a featurization with scheme specified by the user.
    Parameters
    ----------
    mol_pdb_file: Str 
      Filename for ligand pdb file. 
    protein_pdb_file: Str 
      Filename for protein pdb file. 
    """
    try:
      time1 = time.time()

      protein_xyz, protein_rdk = load_molecule(
          protein_pdb_file, calc_charges=True, sanitize=self.sanitize)
      time2 = time.time()
      logger.info(
          "TIMING: Loading protein coordinates took %0.3f s" % (time2 - time1))
      time1 = time.time()
      ligand_xyz, ligand_rdk = load_molecule(
          mol_pdb_file, calc_charges=True, sanitize=self.sanitize)
      time2 = time.time()
      logger.info(
          "TIMING: Loading ligand coordinates took %0.3f s" % (time2 - time1))
    except MoleculeLoadException:
      logger.warning("Some molecules cannot be loaded by Rdkit. Skipping")
      return None

    time1 = time.time()
    centroid = compute_centroid(ligand_xyz)
    ligand_xyz = subtract_centroid(ligand_xyz, centroid)
    protein_xyz = subtract_centroid(protein_xyz, centroid)
    time2 = time.time()
    logger.info("TIMING: Centroid processing took %0.3f s" % (time2 - time1))

    pairwise_distances = compute_pairwise_distances(protein_xyz, ligand_xyz)

    transformed_systems = {}
    transformed_systems[(0, 0)] = [protein_xyz, ligand_xyz]

    for i in range(self.nb_rotations):
      rotated_system = rotate_molecules([protein_xyz, ligand_xyz])
      transformed_systems[(i + 1, 0)] = rotated_system

    features_dict = {}
    for system_id, (protein_xyz, ligand_xyz) in transformed_systems.items():
      feature_arrays = []
      for is_flat, function_name in self.feature_types:

        result = self._compute_feature(
            function_name,
            protein_xyz,
            protein_rdk,
            ligand_xyz,
            ligand_rdk,
            pairwise_distances,
        )
        feature_arrays += result

        if self.flatten:
          features_dict[system_id] = np.concatenate(
              [feature_array.flatten() for feature_array in feature_arrays])
        else:
          features_dict[system_id] = np.concatenate(feature_arrays, axis=-1)

    # TODO(rbharath): Is this squeeze OK?
    features = np.squeeze(np.array(list(features_dict.values())))
    return features

  def _voxelize(self,
                get_voxels,
                hash_function,
                coordinates,
                feature_dict=None,
                feature_list=None,
                channel_power=None,
                nb_channel=16,
                dtype="np.int8"):
    """Private helper function to voxelize inputs.

    Parameters
    ----------
    get_voxels: function
      Function that voxelizes inputs
    hash_function: function
      Used to map feature choices to voxel channels.  
    coordinates: np.ndarray
      Contains the 3D coordinates of a molecular system.
    feature_dict: Dictionary
      Keys are atom indices.  
    feature_list: list
      List of available features. 
    channel_power: int
      If specified, nb_channel is set to 2**channel_power.
      TODO: This feels like a redundant parameter.
    nb_channel: int
      The number of feature channels computed per voxel 
    dtype: type
      The dtype of the numpy ndarray created to hold features.
    """

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
              feature_tensor[voxel[0], voxel[1], voxel[2], 0] += features
    elif feature_list is not None:
      for key in feature_list:
        voxels = get_voxels(coordinates, key, self.box_width, self.voxel_width)
        for voxel in voxels:
          if ((voxel >= 0) & (voxel < self.voxels_per_edge)).all():
            feature_tensor[voxel[0], voxel[1], voxel[2], 0] += 1.0

    return feature_tensor

  def _voxelize_pi_stack(self, prot_xyz, prot_rdk, lig_xyz, lig_rdk, distances):
    protein_pi_t, protein_pi_parallel, ligand_pi_t, ligand_pi_parallel = (
        compute_pi_stack(
            prot_rdk,
            lig_rdk,
            distances,
            dist_cutoff=self.cutoffs['pi_stack_dist_cutoff'],
            angle_cutoff=self.cutoffs['pi_stack_angle_cutoff']))
    pi_parallel_tensor = self._voxelize(
        convert_atom_to_voxel,
        None,
        prot_xyz,
        feature_dict=protein_pi_parallel,
        nb_channel=1)
    pi_parallel_tensor += self._voxelize(
        convert_atom_to_voxel,
        None,
        lig_xyz,
        feature_dict=ligand_pi_parallel,
        nb_channel=1)

    pi_t_tensor = self._voxelize(
        convert_atom_to_voxel,
        None,
        prot_xyz,
        feature_dict=protein_pi_t,
        nb_channel=1)
    pi_t_tensor += self._voxelize(
        convert_atom_to_voxel,
        None,
        lig_xyz,
        feature_dict=ligand_pi_t,
        nb_channel=1)
    return [pi_parallel_tensor, pi_t_tensor]
