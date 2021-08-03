"""
Compute various spatial fingerprints for macromolecular complexes.
"""
import itertools
import logging
import numpy as np
from deepchem.utils import rdkit_utils
from deepchem.feat import ComplexFeaturizer
from deepchem.utils.voxel_utils import voxelize
from deepchem.utils.voxel_utils import convert_atom_to_voxel
from deepchem.utils.voxel_utils import convert_atom_pair_to_voxel
from deepchem.utils.noncovalent_utils import compute_salt_bridges
from deepchem.utils.noncovalent_utils import compute_binding_pocket_cation_pi
from deepchem.utils.noncovalent_utils import compute_pi_stack
from deepchem.utils.noncovalent_utils import compute_hydrogen_bonds
from deepchem.utils.rdkit_utils import MoleculeLoadException
from deepchem.utils.rdkit_utils import compute_contact_centroid
from deepchem.utils.geometry_utils import compute_pairwise_distances
from deepchem.utils.geometry_utils import subtract_centroid
from deepchem.utils.fragment_utils import get_partial_charge
from deepchem.utils.fragment_utils import reduce_molecular_complex_to_contacts
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

HBOND_DIST_BINS = [(2.2, 2.5), (2.5, 3.2), (3.2, 4.0)]
HBOND_ANGLE_CUTOFFS = [5., 50., 90.]


def compute_charge_dictionary(molecule):
  """Create a dictionary with partial charges for each atom in the molecule.

  This function assumes that the charges for the molecule are
  already computed (it can be done with
  rdkit_util.compute_charges(molecule))
  """

  charge_dictionary = {}
  for i, atom in enumerate(molecule.GetAtoms()):
    charge_dictionary[i] = get_partial_charge(atom)
  return charge_dictionary


class ChargeVoxelizer(ComplexFeaturizer):
  """Localize partial charges of atoms in macromolecular complexes.

  Given a macromolecular complex made up of multiple
  constitutent molecules, compute the partial (Gasteiger
  charge) on each molecule. For each atom, localize this
  partial charge in the voxel in which it originated to create
  a local charge array. Sum contributions to get an effective
  charge at each voxel.

  Let `voxels_per_edge = int(box_width/voxel_width)`.  Creates a
  tensor output of shape `(voxels_per_edge, voxels_per_edge,
  voxels_per_edge, 1)` for each macromolecular complex that computes
  the effective charge at each voxel.
  """

  def __init__(self,
               cutoff: float = 4.5,
               box_width: float = 16.0,
               voxel_width: float = 1.0,
               reduce_to_contacts: bool = True):
    """
    Parameters
    ----------
    cutoff: float (default 4.5)
      Distance cutoff in angstroms for molecules in complex.
    box_width: float, optional (default 16.0)
      Size of a box in which voxel features are calculated. Box
      is centered on a ligand centroid.
    voxel_width: float, optional (default 1.0)
      Size of a 3D voxel in a grid.
    reduce_to_contacts: bool, optional
      If True, reduce the atoms in the complex to those near a contact
      region.
    """
    self.cutoff = cutoff
    self.box_width = box_width
    self.voxel_width = voxel_width
    self.reduce_to_contacts = reduce_to_contacts

  def _featurize(self, datapoint, **kwargs):  # -> Optional[np.ndarray]:
    """
    Compute featurization for a single mol/protein complex

    Parameters
    ----------
    datapoint: Tuple[str, str]
      Filenames for molecule and protein.
    """
    if 'complex' in kwargs:
      datapoint = kwargs.get("complex")
      raise DeprecationWarning(
          'Complex is being phased out as a parameter, please pass "datapoint" instead.'
      )

    try:
      fragments = rdkit_utils.load_complex(datapoint, add_hydrogens=False)

    except MoleculeLoadException:
      logger.warning("This molecule cannot be loaded by Rdkit. Returning None")
      return None
    pairwise_features = []
    # We compute pairwise contact fingerprints
    centroid = compute_contact_centroid(fragments, cutoff=self.cutoff)
    if self.reduce_to_contacts:
      fragments = reduce_molecular_complex_to_contacts(fragments, self.cutoff)
    # We compute pairwise contact fingerprints
    for (frag1_ind, frag2_ind) in itertools.combinations(
        range(len(fragments)), 2):
      frag1, frag2 = fragments[frag1_ind], fragments[frag2_ind]
      frag1_xyz = subtract_centroid(frag1[0], centroid)
      frag2_xyz = subtract_centroid(frag2[0], centroid)
      xyzs = [frag1_xyz, frag2_xyz]
      rdks = [frag1[1], frag2[1]]
      pairwise_features.append(
          sum([
              voxelize(
                  convert_atom_to_voxel,
                  hash_function=None,
                  coordinates=xyz,
                  box_width=self.box_width,
                  voxel_width=self.voxel_width,
                  feature_dict=compute_charge_dictionary(mol),
                  nb_channel=1,
                  dtype="np.float16") for xyz, mol in zip(xyzs, rdks)
          ]))
    # Features are of shape (voxels_per_edge, voxels_per_edge, voxels_per_edge, 1) so we should concatenate on the last axis.
    return np.concatenate(pairwise_features, axis=-1)


class SaltBridgeVoxelizer(ComplexFeaturizer):
  """Localize salt bridges between atoms in macromolecular complexes.

  Given a macromolecular complex made up of multiple
  constitutent molecules, compute salt bridges between atoms in
  the macromolecular complex. For each atom, localize this salt
  bridge in the voxel in which it originated to create a local
  salt bridge array. Note that if atoms in two different voxels
  interact in a salt-bridge, the interaction is double counted
  in both voxels.

  Let `voxels_per_edge = int(box_width/voxel_width)`.  Creates a
  tensor output of shape `(voxels_per_edge, voxels_per_edge,
  voxels_per_edge, 1)` for each macromolecular the number of salt
  bridges at each voxel.
  """

  def __init__(self,
               cutoff: float = 5.0,
               box_width: float = 16.0,
               voxel_width: float = 1.0,
               reduce_to_contacts: bool = True):
    """
    Parameters
    ----------
    cutoff: float, optional (default 5.0)
      The distance in angstroms within which atoms must be to
      be considered for a salt bridge between them.
    box_width: float, optional (default 16.0)
      Size of a box in which voxel features are calculated. Box
      is centered on a ligand centroid.
    voxel_width: float, optional (default 1.0)
      Size of a 3D voxel in a grid.
    reduce_to_contacts: bool, optional
      If True, reduce the atoms in the complex to those near a contact
      region.
    """
    self.cutoff = cutoff
    self.box_width = box_width
    self.voxel_width = voxel_width
    self.reduce_to_contacts = reduce_to_contacts

  def _featurize(self, datapoint, **kwargs):  # -> Optional[np.ndarray]:
    """
    Compute featurization for a single mol/protein complex

    Parameters
    ----------
    datapoint: Tuple[str, str]
      Filenames for molecule and protein.
    """
    if 'complex' in kwargs:
      datapoint = kwargs.get("complex")
      raise DeprecationWarning(
          'Complex is being phased out as a parameter, please pass "datapoint" instead.'
      )

    try:
      fragments = rdkit_utils.load_complex(datapoint, add_hydrogens=False)

    except MoleculeLoadException:
      logger.warning("This molecule cannot be loaded by Rdkit. Returning None")
      return None
    pairwise_features = []
    # We compute pairwise contact fingerprints
    centroid = compute_contact_centroid(fragments, cutoff=self.cutoff)
    if self.reduce_to_contacts:
      fragments = reduce_molecular_complex_to_contacts(fragments, self.cutoff)
    for (frag1_ind, frag2_ind) in itertools.combinations(
        range(len(fragments)), 2):
      frag1, frag2 = fragments[frag1_ind], fragments[frag2_ind]
      distances = compute_pairwise_distances(frag1[0], frag2[0])
      frag1_xyz = subtract_centroid(frag1[0], centroid)
      frag2_xyz = subtract_centroid(frag2[0], centroid)
      xyzs = [frag1_xyz, frag2_xyz]
      # rdks = [frag1[1], frag2[1]]
      pairwise_features.append(
          sum([
              voxelize(
                  convert_atom_pair_to_voxel,
                  hash_function=None,
                  coordinates=xyz,
                  box_width=self.box_width,
                  voxel_width=self.voxel_width,
                  feature_list=compute_salt_bridges(
                      frag1[1], frag2[1], distances, cutoff=self.cutoff),
                  nb_channel=1) for xyz in xyzs
          ]))
    # Features are of shape (voxels_per_edge, voxels_per_edge, voxels_per_edge, 1) so we should concatenate on the last axis.
    return np.concatenate(pairwise_features, axis=-1)


class CationPiVoxelizer(ComplexFeaturizer):
  """Localize cation-Pi interactions between atoms in macromolecular complexes.

  Given a macromolecular complex made up of multiple
  constitutent molecules, compute cation-Pi between atoms in
  the macromolecular complex. For each atom, localize this salt
  bridge in the voxel in which it originated to create a local
  cation-pi array.

  Let `voxels_per_edge = int(box_width/voxel_width)`.  Creates a
  tensor output of shape `(voxels_per_edge, voxels_per_edge,
  voxels_per_edge, 1)` for each macromolecular complex that counts the
  number of cation-pi interactions at each voxel.
  """

  def __init__(self,
               cutoff: float = 6.5,
               angle_cutoff: float = 30.0,
               box_width: float = 16.0,
               voxel_width: float = 1.0):
    """
    Parameters
    ----------
    cutoff: float, optional (default 6.5)
      The distance in angstroms within which atoms must be to
      be considered for a cation-pi interaction between them.
    angle_cutoff: float, optional (default 30.0)
      Angle cutoff. Max allowed deviation from the ideal (0deg)
      angle between ring normal and vector pointing from ring
      center to cation (in degrees).
    box_width: float, optional (default 16.0)
      Size of a box in which voxel features are calculated. Box
      is centered on a ligand centroid.
    voxel_width: float, optional (default 1.0)
      Size of a 3D voxel in a grid.
    """
    self.cutoff = cutoff
    self.angle_cutoff = angle_cutoff
    self.box_width = box_width
    self.voxel_width = voxel_width

  def _featurize(self, datapoint, **kwargs):  # -> Optional[np.ndarray]:
    """
    Compute featurization for a single mol/protein complex

    Parameters
    ----------
    datapoint: Tuple[str, str]
      Filenames for molecule and protein.
    """
    if 'complex' in kwargs:
      datapoint = kwargs.get("complex")
      raise DeprecationWarning(
          'Complex is being phased out as a parameter, please pass "datapoint" instead.'
      )

    try:
      fragments = rdkit_utils.load_complex(datapoint, add_hydrogens=False)

    except MoleculeLoadException:
      logger.warning("This molecule cannot be loaded by Rdkit. Returning None")
      return None
    pairwise_features = []
    # We compute pairwise contact fingerprints
    centroid = compute_contact_centroid(fragments, cutoff=self.cutoff)
    for (frag1_ind, frag2_ind) in itertools.combinations(
        range(len(fragments)), 2):
      frag1, frag2 = fragments[frag1_ind], fragments[frag2_ind]
      # distances = compute_pairwise_distances(frag1[0], frag2[0])
      frag1_xyz = subtract_centroid(frag1[0], centroid)
      frag2_xyz = subtract_centroid(frag2[0], centroid)
      xyzs = [frag1_xyz, frag2_xyz]
      # rdks = [frag1[1], frag2[1]]
      pairwise_features.append(
          sum([
              voxelize(
                  convert_atom_to_voxel,
                  hash_function=None,
                  box_width=self.box_width,
                  voxel_width=self.voxel_width,
                  coordinates=xyz,
                  feature_dict=cation_pi_dict,
                  nb_channel=1) for xyz, cation_pi_dict in zip(
                      xyzs,
                      compute_binding_pocket_cation_pi(
                          frag1[1],
                          frag2[1],
                          dist_cutoff=self.cutoff,
                          angle_cutoff=self.angle_cutoff,
                      ))
          ]))
    # Features are of shape (voxels_per_edge, voxels_per_edge, voxels_per_edge, 1) so we should concatenate on the last axis.
    return np.concatenate(pairwise_features, axis=-1)


class PiStackVoxelizer(ComplexFeaturizer):
  """Localize Pi stacking interactions between atoms in macromolecular complexes.

  Given a macromolecular complex made up of multiple
  constitutent molecules, compute pi-stacking interactions
  between atoms in the macromolecular complex. For each atom,
  localize this salt bridge in the voxel in which it originated
  to create a local pi-stacking array.

  Let `voxels_per_edge = int(box_width/voxel_width)`.  Creates a
  tensor output of shape `(voxels_per_edge, voxels_per_edge,
  voxels_per_edge, 2)` for each macromolecular complex. Each voxel has
  2 fields, with the first tracking the number of pi-pi parallel
  interactions, and the second tracking the number of pi-T
  interactions.
  """

  def __init__(self,
               cutoff: float = 4.4,
               angle_cutoff: float = 30.0,
               box_width: float = 16.0,
               voxel_width: float = 1.0):
    """
    Parameters
    ----------
    cutoff: float, optional (default 4.4)
      The distance in angstroms within which atoms must be to
      be considered for a cation-pi interaction between them.
    angle_cutoff: float, optional (default 30.0)
      Angle cutoff. Max allowed deviation from the ideal (0 deg)
      angle between ring normal and vector pointing from ring
      center to other ring center (in degrees).
    box_width: float, optional (default 16.0)
      Size of a box in which voxel features are calculated. Box
      is centered on a ligand centroid.
    voxel_width: float, optional (default 1.0)
      Size of a 3D voxel in a grid.
    """
    self.cutoff = cutoff
    self.angle_cutoff = angle_cutoff
    self.box_width = box_width
    self.voxel_width = voxel_width

  def _featurize(self, datapoint, **kwargs):  # -> Optional[np.ndarray]:
    """
    Compute featurization for a single mol/protein complex

    Parameters
    ----------
    datapoint: Tuple[str, str]
      Filenames for molecule and protein.
    """
    if 'complex' in kwargs:
      datapoint = kwargs.get("complex")
      raise DeprecationWarning(
          'Complex is being phased out as a parameter, please pass "datapoint" instead.'
      )

    try:
      fragments = rdkit_utils.load_complex(datapoint, add_hydrogens=False)

    except MoleculeLoadException:
      logger.warning("This molecule cannot be loaded by Rdkit. Returning None")
      return None
    pairwise_features = []
    # We compute pairwise contact fingerprints
    centroid = compute_contact_centroid(fragments, cutoff=self.cutoff)
    for (frag1_ind, frag2_ind) in itertools.combinations(
        range(len(fragments)), 2):
      frag1, frag2 = fragments[frag1_ind], fragments[frag2_ind]
      distances = compute_pairwise_distances(frag1[0], frag2[0])
      frag1_xyz = subtract_centroid(frag1[0], centroid)
      frag2_xyz = subtract_centroid(frag2[0], centroid)
      xyzs = [frag1_xyz, frag2_xyz]
      # rdks = [frag1[1], frag2[1]]
      protein_pi_t, protein_pi_parallel, ligand_pi_t, ligand_pi_parallel = (
          compute_pi_stack(
              frag1[1],
              frag2[1],
              distances,
              dist_cutoff=self.cutoff,
              angle_cutoff=self.angle_cutoff))
      pi_parallel_tensor = sum([
          voxelize(
              convert_atom_to_voxel,
              hash_function=None,
              box_width=self.box_width,
              voxel_width=self.voxel_width,
              coordinates=xyz,
              feature_dict=feature_dict,
              nb_channel=1)
          for (xyz, feature_dict
              ) in zip(xyzs, [ligand_pi_parallel, protein_pi_parallel])
      ])

      pi_t_tensor = sum([
          voxelize(
              convert_atom_to_voxel,
              hash_function=None,
              box_width=self.box_width,
              voxel_width=self.voxel_width,
              coordinates=frag1_xyz,
              feature_dict=protein_pi_t,
              nb_channel=1)
          for (xyz, feature_dict) in zip(xyzs, [ligand_pi_t, protein_pi_t])
      ])

      pairwise_features.append(
          np.concatenate([pi_parallel_tensor, pi_t_tensor], axis=-1))
    # Features are of shape (voxels_per_edge, voxels_per_edge, voxels_per_edge, 2) so we should concatenate on the last axis.
    return np.concatenate(pairwise_features, axis=-1)


class HydrogenBondCounter(ComplexFeaturizer):
  """Counts hydrogen bonds between atoms in macromolecular complexes.

  Given a macromolecular complex made up of multiple
  constitutent molecules, count the number of hydrogen bonds
  between atoms in the macromolecular complex.

  Creates a scalar output of shape `(3,)` (assuming the default value
  ofor `distance_bins` with 3 bins) for each macromolecular complex
  that computes the total number of hydrogen bonds.
  """

  def __init__(
      self,
      cutoff: float = 4.5,
      reduce_to_contacts: bool = True,
      distance_bins: Optional[List[Tuple[float, float]]] = None,
      angle_cutoffs: Optional[List[float]] = None,
  ):
    """
    Parameters
    ----------
    cutoff: float (default 4.5)
      Distance cutoff in angstroms for molecules in complex.
    reduce_to_contacts: bool, optional
      If True, reduce the atoms in the complex to those near a contact
      region.
    distance_bins: list[tuple]
      List of hydgrogen bond distance bins. If not specified is
      set to default
      `[(2.2, 2.5), (2.5, 3.2), (3.2, 4.0)]`.
    angle_cutoffs: list[float]
      List of hydrogen bond angle cutoffs. Max allowed
      deviation from the ideal (180 deg) angle between
      hydrogen-atom1, hydrogen-atom2 vectors.If not specified
      is set to default `[5, 50, 90]`
    """
    self.cutoff = cutoff
    if distance_bins is None:
      self.distance_bins = HBOND_DIST_BINS
    else:
      self.distance_bins = distance_bins
    if angle_cutoffs is None:
      self.angle_cutoffs = HBOND_ANGLE_CUTOFFS
    else:
      self.angle_cutoffs = angle_cutoffs
    self.reduce_to_contacts = reduce_to_contacts

  def _featurize(self, datapoint, **kwargs):  # -> Optional[np.ndarray]:
    """
    Compute featurization for a single mol/protein complex

    Parameters
    ----------
    datapoint: Tuple[str, str]
      Filenames for molecule and protein.
    """
    if 'complex' in kwargs:
      datapoint = kwargs.get("complex")
      raise DeprecationWarning(
          'Complex is being phased out as a parameter, please pass "datapoint" instead.'
      )
    try:
      fragments = rdkit_utils.load_complex(datapoint, add_hydrogens=False)

    except MoleculeLoadException:
      logger.warning("This molecule cannot be loaded by Rdkit. Returning None")
      return None
    pairwise_features = []
    # We compute pairwise contact fingerprints
    # centroid = compute_contact_centroid(fragments, cutoff=self.cutoff)
    if self.reduce_to_contacts:
      fragments = reduce_molecular_complex_to_contacts(fragments, self.cutoff)
    # We compute pairwise contact fingerprints
    for (frag1_ind, frag2_ind) in itertools.combinations(
        range(len(fragments)), 2):
      frag1, frag2 = fragments[frag1_ind], fragments[frag2_ind]
      distances = compute_pairwise_distances(frag1[0], frag2[0])
      # frag1_xyz = subtract_centroid(frag1[0], centroid)
      # frag2_xyz = subtract_centroid(frag2[0], centroid)
      # xyzs = [frag1_xyz, frag2_xyz]
      # rdks = [frag1[1], frag2[1]]
      pairwise_features.append(
          np.concatenate(
              [
                  np.array([len(hbond_list)])
                  for hbond_list in compute_hydrogen_bonds(
                      frag1, frag2, distances, self.distance_bins,
                      self.angle_cutoffs)
              ],
              axis=-1))
    # Features are of shape (voxels_per_edge, voxels_per_edge, voxels_per_edge, 1) so we should concatenate on the last axis.
    return np.concatenate(pairwise_features, axis=-1)


class HydrogenBondVoxelizer(ComplexFeaturizer):
  """Localize hydrogen bonds between atoms in macromolecular complexes.

  Given a macromolecular complex made up of multiple
  constitutent molecules, compute hydrogen bonds between atoms
  in the macromolecular complex. For each atom, localize this
  hydrogen bond in the voxel in which it originated to create a
  local hydrogen bond array. Note that if atoms in two
  different voxels interact in a hydrogen bond, the interaction
  is double counted in both voxels.

  Let `voxels_per_edge = int(box_width/voxel_width)`.  Creates a
  tensor output of shape `(voxels_per_edge, voxels_per_edge,
  voxels_per_edge, 3)` (assuming the default for `distance_bins` which
  has 3 bins) for each macromolecular complex that counts the number
  of hydrogen bonds at each voxel.
  """

  def __init__(
      self,
      cutoff: float = 4.5,
      box_width: float = 16.0,
      voxel_width: float = 1.0,
      reduce_to_contacts: bool = True,
      distance_bins: Optional[List[Tuple[float, float]]] = None,
      angle_cutoffs: Optional[List[float]] = None,
  ):
    """
    Parameters
    ----------
    cutoff: float (default 4.5)
      Distance cutoff in angstroms for contact atoms in complex.
    box_width: float, optional (default 16.0)
      Size of a box in which voxel features are calculated. Box
      is centered on a ligand centroid.
    voxel_width: float, optional (default 1.0)
      Size of a 3D voxel in a grid.
    reduce_to_contacts: bool, optional
      If True, reduce the atoms in the complex to those near a contact
      region.
    distance_bins: list[tuple]
      List of hydgrogen bond distance bins. If not specified is
      set to default
      `[(2.2, 2.5), (2.5, 3.2), (3.2, 4.0)]`.
    angle_cutoffs: list[float]
      List of hydrogen bond angle cutoffs. Max allowed
      deviation from the ideal (180 deg) angle between
      hydrogen-atom1, hydrogen-atom2 vectors.If not specified
      is set to default `[5, 50, 90]`
    """
    self.cutoff = cutoff
    if distance_bins is None:
      self.distance_bins = HBOND_DIST_BINS
    else:
      self.distance_bins = distance_bins
    if angle_cutoffs is None:
      self.angle_cutoffs = HBOND_ANGLE_CUTOFFS
    else:
      self.angle_cutoffs = angle_cutoffs
    self.box_width = box_width
    self.voxel_width = voxel_width
    self.reduce_to_contacts = reduce_to_contacts

  def _featurize(self, datapoint, **kwargs):  # -> Optional[np.ndarray]:
    """
    Compute featurization for a single mol/protein complex

    Parameters
    ----------
    datapoint: Tuple[str, str]
      Filenames for molecule and protein.
    """
    if 'complex' in kwargs:
      datapoint = kwargs.get("complex")
      raise DeprecationWarning(
          'Complex is being phased out as a parameter, please pass "datapoint" instead.'
      )
    try:
      fragments = rdkit_utils.load_complex(datapoint, add_hydrogens=False)

    except MoleculeLoadException:
      logger.warning("This molecule cannot be loaded by Rdkit. Returning None")
      return None
    pairwise_features = []
    # We compute pairwise contact fingerprints
    centroid = compute_contact_centroid(fragments, cutoff=self.cutoff)
    if self.reduce_to_contacts:
      fragments = reduce_molecular_complex_to_contacts(fragments, self.cutoff)
    for (frag1_ind, frag2_ind) in itertools.combinations(
        range(len(fragments)), 2):
      frag1, frag2 = fragments[frag1_ind], fragments[frag2_ind]
      distances = compute_pairwise_distances(frag1[0], frag2[0])
      frag1_xyz = subtract_centroid(frag1[0], centroid)
      frag2_xyz = subtract_centroid(frag2[0], centroid)
      xyzs = [frag1_xyz, frag2_xyz]
      # rdks = [frag1[1], frag2[1]]
      pairwise_features.append(
          np.concatenate(
              [
                  sum([
                      voxelize(
                          convert_atom_pair_to_voxel,
                          hash_function=None,
                          box_width=self.box_width,
                          voxel_width=self.voxel_width,
                          coordinates=xyz,
                          feature_list=hbond_list,
                          nb_channel=1) for xyz in xyzs
                  ]) for hbond_list in compute_hydrogen_bonds(
                      frag1, frag2, distances, self.distance_bins,
                      self.angle_cutoffs)
              ],
              axis=-1))
    # Features are of shape (voxels_per_edge, voxels_per_edge, voxels_per_edge, 1) so we should concatenate on the last axis.
    return np.concatenate(pairwise_features, axis=-1)
