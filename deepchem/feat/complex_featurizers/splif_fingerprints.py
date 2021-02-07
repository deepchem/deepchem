"""
SPLIF Fingerprints for molecular complexes.
"""
import logging
import itertools
import numpy as np
from deepchem.utils.hash_utils import hash_ecfp_pair
from deepchem.utils.rdkit_utils import load_complex
from deepchem.utils.rdkit_utils import compute_all_ecfp
from deepchem.utils.rdkit_utils import MoleculeLoadException
from deepchem.utils.rdkit_utils import compute_contact_centroid
from deepchem.feat import ComplexFeaturizer
from deepchem.utils.hash_utils import vectorize
from deepchem.utils.voxel_utils import voxelize
from deepchem.utils.voxel_utils import convert_atom_pair_to_voxel
from deepchem.utils.geometry_utils import compute_pairwise_distances
from deepchem.utils.geometry_utils import subtract_centroid

from typing import Tuple, Dict, List

logger = logging.getLogger(__name__)

SPLIF_CONTACT_BINS = [(0, 2.0), (2.0, 3.0), (3.0, 4.5)]


def compute_splif_features_in_range(frag1: Tuple,
                                    frag2: Tuple,
                                    pairwise_distances: np.ndarray,
                                    contact_bin: List,
                                    ecfp_degree: int = 2) -> Dict:
  """Computes SPLIF features for close atoms in molecular complexes.

  Finds all frag1 atoms that are > contact_bin[0] and <
  contact_bin[1] away from frag2 atoms. Then, finds the ECFP
  fingerprints for the contacting atoms. Returns a dictionary
  mapping (frag1_index_i, frag2_index_j) --> (frag1_ecfp_i,
  frag2_ecfp_j)

  Parameters
  ----------
  frag1: Tuple
    A tuple of (coords, mol) returned by `load_molecule`.
  frag2: Tuple
    A tuple of (coords, mol) returned by `load_molecule`.
  contact_bins: np.ndarray
    Ranges of pair distances which are placed in separate bins.
  pairwise_distances: np.ndarray
    Array of pairwise fragment-fragment distances (Angstroms)
  ecfp_degree: int
    ECFP radius
  """
  contacts = np.nonzero((pairwise_distances > contact_bin[0]) &
                        (pairwise_distances < contact_bin[1]))
  frag1_atoms = set([int(c) for c in contacts[0].tolist()])
  contacts = zip(contacts[0], contacts[1])

  frag1_ecfp_dict = compute_all_ecfp(
      frag1[1], indices=frag1_atoms, degree=ecfp_degree)
  frag2_ecfp_dict = compute_all_ecfp(frag2[1], degree=ecfp_degree)
  splif_dict = {
      contact: (frag1_ecfp_dict[contact[0]], frag2_ecfp_dict[contact[1]])
      for contact in contacts
  }
  return splif_dict


def featurize_splif(frag1, frag2, contact_bins, pairwise_distances,
                    ecfp_degree):
  """Computes SPLIF featurization of fragment interactions binding pocket.

  For each contact range (i.e. 1 A to 2 A, 2 A to 3 A, etc.)
  compute a dictionary mapping (frag1_index_i, frag2_index_j)
  tuples --> (frag1_ecfp_i, frag2_ecfp_j) tuples. Return a
  list of such splif dictionaries.

  Parameters
  ----------
  frag1: Tuple
    A tuple of (coords, mol) returned by `load_molecule`.
  frag2: Tuple
    A tuple of (coords, mol) returned by `load_molecule`.
  contact_bins: np.ndarray
    Ranges of pair distances which are placed in separate bins.
  pairwise_distances: np.ndarray
    Array of pairwise fragment-fragment distances (Angstroms)
  ecfp_degree: int
    ECFP radius, the graph distance at which fragments are computed.

  Returns
  -------
  Dictionaries of SPLIF interactions suitable for `vectorize` or
  `voxelize`.
  """
  splif_dicts = []
  for i, contact_bin in enumerate(contact_bins):
    splif_dicts.append(
        compute_splif_features_in_range(frag1, frag2, pairwise_distances,
                                        contact_bin, ecfp_degree))

  return splif_dicts


class SplifFingerprint(ComplexFeaturizer):
  """Computes SPLIF Fingerprints for a macromolecular complex.

  SPLIF fingerprints are based on a technique introduced in the
  following paper.

  Da, C., and D. Kireev. "Structural protein–ligand interaction
  fingerprints (SPLIF) for structure-based virtual screening:
  method and benchmark study." Journal of chemical information
  and modeling 54.9 (2014): 2555-2561.

  SPLIF fingerprints are a subclass of `ComplexFeaturizer`. It
  requires 3D coordinates for a molecular complex. For each ligand
  atom, it identifies close pairs of atoms from different molecules.
  These atom pairs are expanded to 2D circular fragments and a
  fingerprint for the union is turned on in the bit vector. Note that
  we slightly generalize the original paper by not requiring the
  interacting molecules to be proteins or ligands.

  This is conceptually pretty similar to
  `ContactCircularFingerprint` but computes ECFP fragments only
  for direct contacts instead of the entire contact region.

  For a macromolecular complex, returns a vector of shape
  `(len(contact_bins)*size,)`
  """

  def __init__(self, contact_bins=None, radius=2, size=8):
    """
    Parameters
    ----------
    contact_bins: list[tuple]
      List of contact bins. If not specified is set to default
      `[(0, 2.0), (2.0, 3.0), (3.0, 4.5)]`.
    radius : int, optional (default 2)
        Fingerprint radius used for circular fingerprints.
    size: int, optional (default 8)
      Length of generated bit vector.
    """
    if contact_bins is None:
      self.contact_bins = SPLIF_CONTACT_BINS
    else:
      self.contact_bins = contact_bins
    self.size = size
    self.radius = radius

  def _featurize(self, complex: Tuple[str, str]):
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
    pairwise_features = []
    # We compute pairwise contact fingerprints
    for (frag1, frag2) in itertools.combinations(fragments, 2):
      # Get coordinates
      distances = compute_pairwise_distances(frag1[0], frag2[0])
      # distances = compute_pairwise_distances(prot_xyz, lig_xyz)
      vectors = [
          vectorize(hash_ecfp_pair, feature_dict=splif_dict,
                    size=self.size) for splif_dict in featurize_splif(
                        frag1, frag2, self.contact_bins, distances, self.radius)
      ]
      pairwise_features += vectors
    pairwise_features = np.concatenate(pairwise_features)
    return pairwise_features


class SplifVoxelizer(ComplexFeaturizer):
  """Computes SPLIF voxel grid for a macromolecular complex.

  SPLIF fingerprints are based on a technique introduced in the
  following paper [1]_.

  The SPLIF voxelizer localizes local SPLIF descriptors in
  space, by assigning features to the voxel in which they
  originated. This technique may be useful for downstream
  learning methods such as convolutional networks.

  Featurizes a macromolecular complex into a tensor of shape
  `(voxels_per_edge, voxels_per_edge, voxels_per_edge, size)`
  where `voxels_per_edge = int(box_width/voxel_width)`.

  References
  ----------
  .. [1] Da, C., and D. Kireev. "Structural protein–ligand interaction
  fingerprints (SPLIF) for structure-based virtual screening:
  method and benchmark study." Journal of chemical information
  and modeling 54.9 (2014): 2555-2561.
  """

  def __init__(self,
               cutoff: float = 4.5,
               contact_bins: List = None,
               radius: int = 2,
               size: int = 8,
               box_width: float = 16.0,
               voxel_width: float = 1.0):
    """
    Parameters
    ----------
    cutoff: float (default 4.5)
      Distance cutoff in angstroms for molecules in complex.
    contact_bins: list[tuple]
      List of contact bins. If not specified is set to default
      `[(0, 2.0), (2.0, 3.0), (3.0, 4.5)]`.
    radius : int, optional (default 2)
        Fingerprint radius used for circular fingerprints.
    size: int, optional (default 8)
      Length of generated bit vector.
    box_width: float, optional (default 16.0)
      Size of a box in which voxel features are calculated. Box
      is centered on a ligand centroid.
    voxel_width: float, optional (default 1.0)
      Size of a 3D voxel in a grid.
    """
    self.cutoff = cutoff
    if contact_bins is None:
      self.contact_bins = SPLIF_CONTACT_BINS
    else:
      self.contact_bins = contact_bins
    self.size = size
    self.radius = radius
    self.box_width = box_width
    self.voxel_width = voxel_width
    self.voxels_per_edge = int(self.box_width / self.voxel_width)

  def _featurize(self, complex: Tuple[str, str]):
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
    pairwise_features = []
    # We compute pairwise contact fingerprints
    centroid = compute_contact_centroid(fragments, cutoff=self.cutoff)
    for (frag1, frag2) in itertools.combinations(fragments, 2):
      distances = compute_pairwise_distances(frag1[0], frag2[0])
      frag1_xyz = subtract_centroid(frag1[0], centroid)
      frag2_xyz = subtract_centroid(frag2[0], centroid)
      xyzs = [frag1_xyz, frag2_xyz]
      pairwise_features.append(
          np.concatenate(
              [
                  voxelize(
                      convert_atom_pair_to_voxel,
                      hash_function=hash_ecfp_pair,
                      coordinates=xyzs,
                      box_width=self.box_width,
                      voxel_width=self.voxel_width,
                      feature_dict=splif_dict,
                      nb_channel=self.size)
                  for splif_dict in featurize_splif(
                      frag1, frag2, self.contact_bins, distances, self.radius)
              ],
              axis=-1))
    # Features are of shape (voxels_per_edge, voxels_per_edge, voxels_per_edge, 1) so we should concatenate on the last axis.
    return np.concatenate(pairwise_features, axis=-1)
