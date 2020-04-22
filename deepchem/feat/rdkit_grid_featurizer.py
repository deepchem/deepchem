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
from deepchem.feat.fingerprints import CircularFingerprint
from deepchem.feat.contact_fingerprints import ContactCircularFingerprint
from deepchem.feat.contact_fingerprints import ContactCircularVoxelizer
from deepchem.feat.splif_fingerprint import SplifFingerprint
from deepchem.feat.splif_fingerprint import SplifVoxelizer
from deepchem.feat.grid_featurizers import compute_charge_dictionary
from deepchem.feat.grid_featurizers import ChargeVoxelizer
from deepchem.feat.grid_featurizers import SaltBridgeVoxelizer
from deepchem.feat.grid_featurizers import CationPiVoxelizer
from deepchem.feat.grid_featurizers import PiStackVoxelizer
from deepchem.feat.grid_featurizers import HydrogenBondCounter
from deepchem.feat.grid_featurizers import HydrogenBondVoxelizer
from deepchem.utils.rdkit_util import compute_hydrogen_bonds
from deepchem.utils.rdkit_util import load_molecule
from deepchem.utils.rdkit_util import compute_centroid
from deepchem.utils.rdkit_util import subtract_centroid
from deepchem.utils.rdkit_util import compute_ring_center
from deepchem.utils.rdkit_util import rotate_molecules
from deepchem.utils.rdkit_util import compute_pairwise_distances
from deepchem.utils.rdkit_util import is_salt_bridge
from deepchem.utils.rdkit_util import compute_salt_bridges
from deepchem.utils.rdkit_util import compute_pi_stack
from deepchem.utils.rdkit_util import compute_binding_pocket_cation_pi
from deepchem.utils.rdkit_util import is_hydrogen_bond
from deepchem.utils.rdkit_util import compute_all_ecfp
from deepchem.utils.rdkit_util import MoleculeLoadException
from deepchem.utils.hash_utils import hash_ecfp
from deepchem.utils.hash_utils import hash_ecfp_pair
from deepchem.utils.hash_utils import vectorize
from deepchem.utils.voxel_utils import voxelize
from deepchem.utils.voxel_utils import convert_atom_to_voxel
from deepchem.utils.voxel_utils import convert_atom_pair_to_voxel
from deepchem.feat import ComplexFeaturizer

logger = logging.getLogger(__name__)
"""following two functions adapted from:
http://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python
"""

# TODO(rbharath): Consider this comment on rdkit forums https://github.com/rdkit/rdkit/issues/1590 about sybyl featurization

FLAT_FEATURES = ['ecfp_ligand', 'ecfp_hashed', 'splif_hashed', 'hbond_count']

VOXEL_FEATURES = [
    'ecfp', 'splif', 'salt_bridge', 'charge', 'hbond', 'pi_stack', 'cation_pi'
]


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
      Size of a box in which voxel features are calculated. Box
      is centered on a ligand centroid.
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
    #sybyl_cutoff: 7.0
    salt_bridges_cutoff: 5.0
    pi_stack_dist_cutoff: 4.4
    pi_stack_angle_cutoff: 30.0
    cation_pi_dist_cutoff: 6.5
    cation_pi_angle_cutoff: 30.0
    """

    # list of features that require sanitized molecules
    require_sanitized = ['pi_stack', 'cation_pi', 'ecfp_ligand']

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

    # Intantiate Featurizers
    self.ecfp_featurizer = CircularFingerprint(
        radius=self.ecfp_degree, size=2**self.ecfp_power)
    self.contact_featurizer = ContactCircularFingerprint(
        cutoff=self.cutoffs['ecfp_cutoff'],
        radius=self.ecfp_degree,
        size=2**self.ecfp_power)
    self.contact_voxelizer = ContactCircularVoxelizer(
        cutoff=self.cutoffs['ecfp_cutoff'],
        radius=self.ecfp_degree,
        size=2**self.ecfp_power,
        box_width=self.box_width,
        voxel_width=self.voxel_width)
    self.splif_featurizer = SplifFingerprint(
        contact_bins=self.cutoffs['splif_contact_bins'],
        radius=self.ecfp_degree,
        size=2**self.splif_power)
    self.splif_voxelizer = SplifVoxelizer(
        contact_bins=self.cutoffs['splif_contact_bins'],
        radius=self.ecfp_degree,
        size=2**self.splif_power,
        box_width=self.box_width,
        voxel_width=self.voxel_width)
    self.charge_voxelizer = ChargeVoxelizer(
        box_width=self.box_width, voxel_width=self.voxel_width)
    self.salt_bridge_voxelizer = SaltBridgeVoxelizer(
        cutoff=self.cutoffs['salt_bridges_cutoff'],
        box_width=self.box_width,
        voxel_width=self.voxel_width)
    self.cation_pi_voxelizer = CationPiVoxelizer(
        distance_cutoff=self.cutoffs['cation_pi_dist_cutoff'],
        angle_cutoff=self.cutoffs['cation_pi_angle_cutoff'],
        box_width=self.box_width,
        voxel_width=self.voxel_width)
    self.pi_stack_voxelizer = PiStackVoxelizer(
        distance_cutoff=self.cutoffs['pi_stack_dist_cutoff'],
        angle_cutoff=self.cutoffs['pi_stack_angle_cutoff'],
        box_width=self.box_width,
        voxel_width=self.voxel_width)
    self.hbond_counter = HydrogenBondCounter(
        distance_bins=self.cutoffs['hbond_dist_bins'],
        angle_cutoffs=self.cutoffs['hbond_angle_cutoffs'])
    self.hbond_voxelizer = HydrogenBondVoxelizer(
        distance_bins=self.cutoffs['hbond_dist_bins'],
        angle_cutoffs=self.cutoffs['hbond_angle_cutoffs'],
        box_width=self.box_width,
        voxel_width=self.voxel_width)

    # parse provided feature types
    for feature_type in feature_types:
      if self.sanitize is False and feature_type in require_sanitized:
        logger.warning('sanitize is set to False, %s feature will be ignored' %
                       feature_type)
        continue

      if feature_type in FLAT_FEATURES:
        self.feature_types.append((True, feature_type))
        if self.flatten is False:
          logger.warning(
              '%s feature is used, output will be flattened' % feature_type)
          self.flatten = True

      elif feature_type in VOXEL_FEATURES:
        self.feature_types.append((False, feature_type))

      elif feature_type == 'flat_combined':
        self.feature_types += [(True, ftype)
                               for ftype in sorted(FLAT_FEATURES)
                               if ftype not in ignored_features]
        if self.flatten is False:
          logger.warning('Flat features are used, output will be flattened')
          self.flatten = True

      elif feature_type == 'voxel_combined':
        self.feature_types += [(False, ftype)
                               for ftype in sorted(VOXEL_FEATURES)
                               if ftype not in ignored_features]
      elif feature_type == 'all_combined':
        self.feature_types += [(True, ftype)
                               for ftype in sorted(FLAT_FEATURES)
                               if ftype not in ignored_features]
        self.feature_types += [(False, ftype)
                               for ftype in sorted(VOXEL_FEATURES)
                               if ftype not in ignored_features]
        if self.flatten is False:
          logger.warning('Flat feature are used, output will be flattened')
          self.flatten = True
      else:
        logger.warning('Ignoring unknown feature %s' % feature_type)

  def _compute_feature(self, feature_name, prot_xyz, prot_rdk, lig_xyz, lig_rdk,
                       distances):
    if feature_name == 'ecfp_ligand':
      return [self.ecfp_featurizer(lig_rdk)]
    if feature_name == 'ecfp_hashed':
      return self.contact_featurizer._featurize_complex((lig_xyz, lig_rdk),
                                                        (prot_xyz, prot_rdk))
    if feature_name == 'splif_hashed':
      return self.splif_featurizer._featurize_complex((lig_xyz, lig_rdk),
                                                      (prot_xyz, prot_rdk))
    if feature_name == 'hbond_count':
      return self.hbond_counter._featurize_complex((lig_xyz, lig_rdk),
                                                   (prot_xyz, prot_rdk))
    if feature_name == 'ecfp':
      return self.contact_voxelizer._featurize_complex((lig_xyz, lig_rdk),
                                                       (prot_xyz, prot_rdk))
    if feature_name == 'splif':
      return self.splif_voxelizer._featurize_complex((lig_xyz, lig_rdk),
                                                     (prot_xyz, prot_rdk))
    if feature_name == 'salt_bridge':
      return self.salt_bridge_voxelizer._featurize_complex((lig_xyz, lig_rdk),
                                                           (prot_xyz, prot_rdk))
    if feature_name == 'charge':
      return self.charge_voxelizer._featurize_complex((lig_xyz, lig_rdk),
                                                      (prot_xyz, prot_rdk))
    if feature_name == 'hbond':
      return self.hbond_voxelizer._featurize_complex((lig_xyz, lig_rdk),
                                                     (prot_xyz, prot_rdk))
    if feature_name == 'pi_stack':
      return self.pi_stack_voxelizer._featurize_complex((lig_xyz, lig_rdk),
                                                        (prot_xyz, prot_rdk))
    if feature_name == 'cation_pi':
      return self.cation_pi_voxelizer._featurize_complex((lig_xyz, lig_rdk),
                                                         (prot_xyz, prot_rdk))
    raise ValueError('Unknown feature type "%s"' % feature_name)

  def _featurize(self, mol_pdb_file, protein_pdb_file):
    """Computes grid featurization of protein/ligand complex.

    Takes as input filenames pdb of the protein, pdb of the
    ligand.

    This function then computes the centroid of the ligand;
    decrements this centroid from the atomic coordinates of
    protein and ligand atoms, and then merges the translated
    protein and ligand. This combined system/complex is then
    saved.

    This function then computes a featurization with scheme
    specified by the user.

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

    features = np.squeeze(np.array(list(features_dict.values())))
    return features
