"""
Topological fingerprints for macromolecular structures.
"""
import numpy as np
import logging
import itertools
from deepchem.utils.hash_utils import hash_ecfp
from deepchem.feat import ComplexFeaturizer
from deepchem.utils.rdkit_utils import load_complex
from deepchem.utils.hash_utils import vectorize
from deepchem.utils.voxel_utils import voxelize
from deepchem.utils.voxel_utils import convert_atom_to_voxel
from deepchem.utils.rdkit_utils import compute_all_ecfp
from deepchem.utils.rdkit_utils import compute_contact_centroid
from deepchem.utils.rdkit_utils import MoleculeLoadException
from deepchem.utils.geometry_utils import compute_pairwise_distances
from deepchem.utils.geometry_utils import subtract_centroid

from typing import Tuple, Dict, List

logger = logging.getLogger(__name__)


def featurize_contacts_ecfp(
    frag1: Tuple,
    frag2: Tuple,
    pairwise_distances: np.ndarray = None,
    cutoff: float = 4.5,
    ecfp_degree: int = 2) -> Tuple[Dict[int, str], Dict[int, str]]:
  """Computes ECFP dicts for pairwise interaction between two molecular fragments.

  Parameters
  ----------
  frag1: Tuple
    A tuple of (coords, mol) returned by `load_molecule`.
  frag2: Tuple
    A tuple of (coords, mol) returned by `load_molecule`.
  pairwise_distances: np.ndarray
    Array of pairwise fragment-fragment distances (Angstroms)
  cutoff: float
    Cutoff distance for contact consideration
  ecfp_degree: int
    ECFP radius

  Returns
  -------
  Tuple of dictionaries of ECFP contact fragments
  """
  if pairwise_distances is None:
    pairwise_distances = compute_pairwise_distances(frag1[0], frag2[0])
  # contacts is of form (x_coords, y_coords), a tuple of 2 lists
  contacts = np.nonzero((pairwise_distances < cutoff))
  # contacts[0] is the x_coords, that is the frag1 atoms that have
  # nonzero contact.
  frag1_atoms = set([int(c) for c in contacts[0].tolist()])
  # contacts[1] is the y_coords, the frag2 atoms with nonzero contacts
  frag2_atoms = set([int(c) for c in contacts[1].tolist()])

  frag1_ecfp_dict = compute_all_ecfp(
      frag1[1], indices=frag1_atoms, degree=ecfp_degree)
  frag2_ecfp_dict = compute_all_ecfp(
      frag2[1], indices=frag2_atoms, degree=ecfp_degree)

  return (frag1_ecfp_dict, frag2_ecfp_dict)


class ContactCircularFingerprint(ComplexFeaturizer):
  """Compute (Morgan) fingerprints near contact points of macromolecular complexes.

  Given a macromolecular complex made up of multiple
  constituent molecules, first compute the contact points where
  atoms from different molecules come close to one another. For
  atoms within "contact regions," compute radial "ECFP"
  fragments which are sub-molecules centered at atoms in the
  contact region.

  For a macromolecular complex, returns a vector of shape
  `(2*size,)`
  """

  def __init__(self, cutoff: float = 4.5, radius: int = 2, size: int = 8):
    """
    Parameters
    ----------
    cutoff: float (default 4.5)
      Distance cutoff in angstroms for molecules in complex.
    radius: int, optional (default 2)
      Fingerprint radius.
    size: int, optional (default 8)
      Length of generated bit vector.
    """
    self.cutoff = cutoff
    self.radius = radius
    self.size = size

  def _featurize(self, datapoint, **kwargs):
    """
    Compute featurization for a molecular complex

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
      fragments = load_complex(datapoint, add_hydrogens=False)

    except MoleculeLoadException:
      logger.warning("This molecule cannot be loaded by Rdkit. Returning None")
      return None
    pairwise_features = []
    # We compute pairwise contact fingerprints
    for (frag1, frag2) in itertools.combinations(fragments, 2):
      # Get coordinates
      distances = compute_pairwise_distances(frag1[0], frag2[0])
      vector = [
          vectorize(hash_ecfp, feature_dict=ecfp_dict, size=self.size)
          for ecfp_dict in featurize_contacts_ecfp(
              frag1,
              frag2,
              distances,
              cutoff=self.cutoff,
              ecfp_degree=self.radius)
      ]
      pairwise_features += vector

    pairwise_features = np.concatenate(pairwise_features)
    return pairwise_features


class ContactCircularVoxelizer(ComplexFeaturizer):
  """Computes ECFP fingerprints on a voxel grid.

  Given a macromolecular complex made up of multiple
  constituent molecules, first compute the contact points where
  atoms from different molecules come close to one another. For
  atoms within "contact regions," compute radial "ECFP"
  fragments which are sub-molecules centered at atoms in the
  contact region. Localize these ECFP fingeprints at the voxel
  in which they originated.

  Featurizes a macromolecular complex into a tensor of shape
  `(voxels_per_edge, voxels_per_edge, voxels_per_edge, size)` where
  `voxels_per_edge = int(box_width/voxel_width)`. If `flatten==True`,
  then returns a flattened version of this tensor of length
  `size*voxels_per_edge**3`
  """

  def __init__(self,
               cutoff: float = 4.5,
               radius: int = 2,
               size: int = 8,
               box_width: float = 16.0,
               voxel_width: float = 1.0,
               flatten: bool = False):
    """
    Parameters
    ----------
    cutoff: float (default 4.5)
      Distance cutoff in angstroms for molecules in complex.
    radius : int, optional (default 2)
      Fingerprint radius.
    size : int, optional (default 8)
      Length of generated bit vector.
    box_width: float, optional (default 16.0)
      Size of a box in which voxel features are calculated. Box
      is centered on a ligand centroid.
    voxel_width: float, optional (default 1.0)
      Size of a 3D voxel in a grid.
    flatten: bool, optional (default False)
      If True, then returns a flat feature vector rather than voxel grid. This
      feature vector is constructed by flattening the usual voxel grid.
    """
    self.cutoff = cutoff
    self.radius = radius
    self.size = size
    self.box_width = box_width
    self.voxel_width = voxel_width
    self.voxels_per_edge = int(self.box_width / self.voxel_width)
    self.flatten = flatten

  def _featurize(self, complex):
    """
    Compute featurization for a molecular complex

    Parameters
    ----------
    complex: Tuple[str, str]
      Filenames for molecule and protein.
    """
    try:
      fragments = load_complex(complex, add_hydrogens=False)

    except MoleculeLoadException:
      logger.warning("This molecule cannot be loaded by Rdkit. Returning None")
      return None
    pairwise_features: List[np.ndarray] = []
    # We compute pairwise contact fingerprints
    centroid = compute_contact_centroid(fragments, cutoff=self.cutoff)
    for (frag1, frag2) in itertools.combinations(fragments, 2):
      distances = compute_pairwise_distances(frag1[0], frag2[0])
      frag1_xyz = subtract_centroid(frag1[0], centroid)
      frag2_xyz = subtract_centroid(frag2[0], centroid)
      xyzs = [frag1_xyz, frag2_xyz]
      pairwise_features.append(
          sum([
              voxelize(
                  convert_atom_to_voxel,
                  xyz,
                  self.box_width,
                  self.voxel_width,
                  hash_function=hash_ecfp,
                  feature_dict=ecfp_dict,
                  nb_channel=self.size) for xyz, ecfp_dict in zip(
                      xyzs,
                      featurize_contacts_ecfp(
                          frag1,
                          frag2,
                          distances,
                          cutoff=self.cutoff,
                          ecfp_degree=self.radius))
          ]))
    if self.flatten:
      return np.concatenate(
          [features.flatten() for features in pairwise_features])
    else:
      # Features are of shape (voxels_per_edge, voxels_per_edge,
      # voxels_per_edge, num_feat) so we should concatenate on the last
      # axis.
      return np.concatenate(pairwise_features, axis=-1)


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

  if pairwise_distances is None:
    pairwise_distances = compute_pairwise_distances(protein_xyz, ligand_xyz)
  contacts = np.nonzero((pairwise_distances < cutoff))
  protein_atoms = set([int(c) for c in contacts[0].tolist()])

  protein_sybyl_dict = compute_all_sybyl(protein, indices=protein_atoms)
  ligand_sybyl_dict = compute_all_sybyl(ligand)
  return (protein_sybyl_dict, ligand_sybyl_dict)
