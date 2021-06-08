import logging
import time

from deepchem.utils.rdkit_utils import MoleculeLoadException, load_molecule, compute_ecfp_features
from deepchem.utils.geometry_utils import rotate_molecules, compute_pairwise_distances, compute_centroid, subtract_centroid
from deepchem.utils.hash_utils import hash_ecfp, hash_ecfp_pair, hash_sybyl, vectorize
from deepchem.utils.noncovalent_utils import compute_hydrogen_bonds, compute_salt_bridges, compute_binding_pocket_cation_pi
from deepchem.utils.voxel_utils import convert_atom_to_voxel, convert_atom_pair_to_voxel, voxelize, voxelize_pi_stack

from deepchem.feat.complex_featurizers.contact_fingerprints import featurize_contacts_ecfp, featurize_binding_pocket_sybyl
from deepchem.feat.complex_featurizers.splif_fingerprints import featurize_splif
from deepchem.feat.complex_featurizers.grid_featurizers import compute_charge_dictionary

import numpy as np
from deepchem.feat import ComplexFeaturizer

logger = logging.getLogger(__name__)


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
               verbose=True,
               sanitize=False,
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
    verbose: bool, optional (defaul True)
      Verbolity for logging
    sanitize: bool, optional (defaul False)
      If set to True molecules will be sanitized. Note that calculating some
      features (e.g. aromatic interactions) require sanitized molecules.
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

    # check if user tries to set removed arguments
    deprecated_args = [
        'box_x', 'box_y', 'box_z', 'save_intermediates', 'voxelize_features',
        'parallel', 'voxel_feature_types'
    ]

    # list of features that require sanitized molecules
    require_sanitized = ['pi_stack', 'cation_pi', 'ecfp_ligand']

    # not implemented featurization types
    not_implemented = ['sybyl']

    for arg in deprecated_args:
      if arg in kwargs and verbose:
        logger.warning(
            '%s argument was removed and it is ignored,'
            ' using it will result in error in version 1.4' % arg,
            DeprecationWarning)

    self.verbose = verbose
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

    self.sybyl_types = [
        "C3", "C2", "C1", "Cac", "Car", "N3", "N3+", "Npl", "N2", "N1", "Ng+",
        "Nox", "Nar", "Ntr", "Nam", "Npl3", "N4", "O3", "O-", "O2", "O.co2",
        "O.spc", "O.t3p", "S3", "S3+", "S2", "So2", "Sox"
        "Sac"
        "SO", "P3", "P", "P3+", "F", "Cl", "Br", "I"
    ]

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

    # list of features that cannot be calculated with specified parameters
    # this list is used to define <flat/voxel/all>_combined subset
    ignored_features = []
    if self.sanitize is False:
      ignored_features += require_sanitized
    ignored_features += not_implemented

    # parse provided feature types
    for feature_type in feature_types:
      if self.sanitize is False and feature_type in require_sanitized:
        if self.verbose:
          logger.warning('sanitize is set to False, %s feature will be ignored'
                         % feature_type)
        continue
      if feature_type in not_implemented:
        if self.verbose:
          logger.warning('%s feature is not implemented yet and will be ignored'
                         % feature_type)
        continue

      if feature_type in self.FLAT_FEATURES:
        self.feature_types.append((True, feature_type))
        if self.flatten is False:
          if self.verbose:
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
          if self.verbose:
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
          if self.verbose:
            logger.warning('Flat feature are used, output will be flattened')
          self.flatten = True
      elif self.verbose:
        logger.warning('Ignoring unknown feature %s' % feature_type)

  def _compute_feature(self, feature_name, prot_xyz, prot_rdk, lig_xyz, lig_rdk,
                       distances):
    if feature_name == 'ecfp_ligand':
      return [compute_ecfp_features(lig_rdk, self.ecfp_degree, self.ecfp_power)]
    if feature_name == 'ecfp_hashed':
      return [
          vectorize(hash_ecfp, feature_dict=ecfp_dict, size=2**self.ecfp_power)
          for ecfp_dict in featurize_contacts_ecfp(
              (prot_xyz, prot_rdk), (lig_xyz, lig_rdk),
              distances,
              cutoff=self.cutoffs['ecfp_cutoff'],
              ecfp_degree=self.ecfp_degree)
      ]
    if feature_name == 'splif_hashed':
      return [
          vectorize(
              hash_ecfp_pair, feature_dict=splif_dict, size=2**self.splif_power)
          for splif_dict in featurize_splif((prot_xyz, prot_rdk), (
              lig_xyz, lig_rdk
          ), self.cutoffs['splif_contact_bins'], distances, self.ecfp_degree)
      ]
    if feature_name == 'hbond_count':
      return [
          vectorize(hash_ecfp_pair, feature_list=hbond_list, size=2**0)
          for hbond_list in compute_hydrogen_bonds((prot_xyz, prot_rdk), (
              lig_xyz, lig_rdk), distances, self.cutoffs[
                  'hbond_dist_bins'], self.cutoffs['hbond_angle_cutoffs'])
      ]
    if feature_name == 'ecfp':
      return [
          sum([
              voxelize(
                  convert_atom_to_voxel,
                  xyz,
                  box_width=self.box_width,
                  voxel_width=self.voxel_width,
                  hash_function=hash_ecfp,
                  feature_dict=ecfp_dict,
                  nb_channel=2**self.ecfp_power,
              )
              for xyz, ecfp_dict in zip((prot_xyz, lig_xyz),
                                        featurize_contacts_ecfp(
                                            (prot_xyz, prot_rdk), (lig_xyz,
                                                                   lig_rdk),
                                            distances,
                                            cutoff=self.cutoffs['ecfp_cutoff'],
                                            ecfp_degree=self.ecfp_degree))
          ])
      ]
    if feature_name == 'splif':
      return [
          voxelize(
              convert_atom_pair_to_voxel,
              (prot_xyz, lig_xyz),
              box_width=self.box_width,
              voxel_width=self.voxel_width,
              hash_function=hash_ecfp_pair,
              feature_dict=splif_dict,
              nb_channel=2**self.splif_power,
          ) for splif_dict in featurize_splif((prot_xyz, prot_rdk), (
              lig_xyz, lig_rdk
          ), self.cutoffs['splif_contact_bins'], distances, self.ecfp_degree)
      ]
    if feature_name == 'sybyl':

      def hash_sybyl_func(x):
        hash_sybyl(x, sybyl_types=self.sybyl_types)

      return [
          voxelize(
              convert_atom_to_voxel,
              xyz,
              box_width=self.box_width,
              voxel_width=self.voxel_width,
              hash_function=hash_sybyl_func,
              feature_dict=sybyl_dict,
              nb_channel=len(self.sybyl_types),
          ) for xyz, sybyl_dict in zip((prot_xyz, lig_xyz),
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
          voxelize(
              convert_atom_pair_to_voxel,
              (prot_xyz, lig_xyz),
              box_width=self.box_width,
              voxel_width=self.voxel_width,
              feature_list=compute_salt_bridges(
                  prot_rdk,
                  lig_rdk,
                  distances,
                  cutoff=self.cutoffs['salt_bridges_cutoff']),
              nb_channel=1,
          )
      ]
    if feature_name == 'charge':
      return [
          sum([
              voxelize(
                  convert_atom_to_voxel,
                  xyz,
                  box_width=self.box_width,
                  voxel_width=self.voxel_width,
                  feature_dict=compute_charge_dictionary(mol),
                  nb_channel=1,
                  dtype="np.float16")
              for xyz, mol in ((prot_xyz, prot_rdk), (lig_xyz, lig_rdk))
          ])
      ]
    if feature_name == 'hbond':
      return [
          voxelize(
              convert_atom_pair_to_voxel,
              (prot_xyz, lig_xyz),
              box_width=self.box_width,
              voxel_width=self.voxel_width,
              feature_list=hbond_list,
              nb_channel=2**0,
          ) for hbond_list in compute_hydrogen_bonds((prot_xyz, prot_rdk), (
              lig_xyz, lig_rdk), distances, self.cutoffs[
                  'hbond_dist_bins'], self.cutoffs['hbond_angle_cutoffs'])
      ]
    if feature_name == 'pi_stack':
      return voxelize_pi_stack(prot_xyz, prot_rdk, lig_xyz, lig_rdk, distances,
                               self.cutoffs['pi_stack_dist_cutoff'],
                               self.cutoffs['pi_stack_angle_cutoff'],
                               self.box_width, self.voxel_width)
    if feature_name == 'cation_pi':
      return [
          sum([
              voxelize(
                  convert_atom_to_voxel,
                  xyz,
                  box_width=self.box_width,
                  voxel_width=self.voxel_width,
                  feature_dict=cation_pi_dict,
                  nb_channel=1,
              ) for xyz, cation_pi_dict in zip(
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

  def _featurize(self, complex):
    """Computes grid featurization of protein/ligand complex.

    Takes as input filenames pdb of the protein, pdb of the ligand.

    This function then computes the centroid of the ligand; decrements this
    centroid from the atomic coordinates of protein and ligand atoms, and then
    merges the translated protein and ligand. This combined system/complex is
    then saved.

    This function then computes a featurization with scheme specified by the user.

    Parameters
    ----------
    complex: Tuple[str, str]
      Filenames for molecule and protein.
    """
    try:
      mol_pdb_file, protein_pdb_file = complex
      time1 = time.time()

      protein_xyz, protein_rdk = load_molecule(
          protein_pdb_file, calc_charges=True, sanitize=self.sanitize)
      time2 = time.time()
      logger.info(
          "TIMING: Loading protein coordinates took %0.3f s" % (time2 - time1),
          self.verbose)
      time1 = time.time()
      ligand_xyz, ligand_rdk = load_molecule(
          mol_pdb_file, calc_charges=True, sanitize=self.sanitize)
      time2 = time.time()
      logger.info(
          "TIMING: Loading ligand coordinates took %0.3f s" % (time2 - time1),
          self.verbose)
    except MoleculeLoadException:
      logger.warning("Some molecules cannot be loaded by Rdkit. Skipping")
      return None

    time1 = time.time()
    centroid = compute_centroid(ligand_xyz)
    ligand_xyz = subtract_centroid(ligand_xyz, centroid)
    protein_xyz = subtract_centroid(protein_xyz, centroid)
    time2 = time.time()
    logger.info("TIMING: Centroid processing took %0.3f s" % (time2 - time1),
                self.verbose)

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

    features = np.concatenate(list(features_dict.values()))
    return features
