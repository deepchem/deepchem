"""
Topological fingerprints for macromolecular structures.
"""
import numpy as np
import logging
import itertools
from deepchem.utils.hash_utils import hash_ecfp
from deepchem.feat import ComplexFeaturizer
from deepchem.utils import rdkit_util
from deepchem.utils.hash_utils import vectorize
from deepchem.utils.voxel_utils import voxelize 
from deepchem.utils.voxel_utils import convert_atom_to_voxel
from deepchem.utils.rdkit_util import compute_all_ecfp
from deepchem.utils.rdkit_util import compute_contact_centroid
from deepchem.utils.rdkit_util import subtract_centroid
from deepchem.utils.rdkit_util import compute_pairwise_distances
from deepchem.utils.rdkit_util import MoleculeLoadException

logger = logging.getLogger(__name__)

def featurize_contacts_ecfp(frag1,
                            frag2,
                            pairwise_distances=None,
                            cutoff=4.5,
                            ecfp_degree=2):
  """Computes ECFP dicts for pairwise interaction between two molecular fragments.

  Parameters
  ----------
  frag1: Tuple
    A tuple of (coords, mol) returned by `rdkit_util.load_molecule`.
  frag2: Tuple
    A tuple of (coords, mol) returned by `rdkit_util.load_molecule`.
  pairwise_distances: np.ndarray
    Array of pairwise protein-ligand distances (Angstroms)
  cutoff: float
    Cutoff distance for contact consideration
  ecfp_degree: int
    ECFP radius
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
  frag2_ecfp_dict = compute_all_ecfp(frag2[1], indices=frag2_atoms, degree=ecfp_degree)

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

  def __init__(self, 
               cutoff=4.5,
               radius=2,
               size=8):
    """
    Parameters
    ----------
    cutoff: float (default 4.5)
      Distance cutoff in angstroms for molecules in complex.
    radius : int, optional (default 2)
        Fingerprint radius.
    size : int, optional (default 8)
      Length of generated bit vector.
    """
    self.cutoff = cutoff
    self.radius = radius
    self.size = size
      

  def _featurize_complex(self, molecular_complex):
    """
    Compute featurization for a single mol/protein complex

    Parameters
    ----------
    molecular_complex: Object
      Some representation of a molecular complex.
    """
    try:
      fragments = rdkit_util.load_complex(molecular_complex, add_hydrogens=False)

    except MoleculeLoadException:
      logger.warning("This molecule cannot be loaded by Rdkit. Returning None")
      return None
    pairwise_features = []
    # We compute pairwise contact fingerprints
    for (frag1, frag2) in itertools.combinations(fragments, 2):
      # Get coordinates
      distances = compute_pairwise_distances(frag1[0], frag2[0])
      vector = [vectorize(
            hash_ecfp, feature_dict=ecfp_dict, size=self.size)
        for ecfp_dict in featurize_contacts_ecfp(
            frag1,
            frag2,
            distances,
            cutoff=self.cutoff,
            ecfp_degree=self.radius)]
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
  `(voxels_per_edge, voxels_per_edge, voxels_per_edge, size)`
  where `voxels_per_edge = int(box_width/voxel_width)`.
  """

  def __init__(self, 
               cutoff=4.5,
               radius=2,
               size=8,
               box_width=16.0,
               voxel_width=1.0):
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
    """
    self.cutoff = cutoff
    self.radius = radius
    self.size = size
    self.box_width = box_width
    self.voxel_width = voxel_width
    self.voxels_per_edge = int(self.box_width / self.voxel_width)

  def _featurize_complex(self, molecular_complex):
    """
    Compute featurization for a single mol/protein complex

    Parameters
    ----------
    molecular_complex: Object
      A representation of a molecular complex, produced by
      `rdkit_util.load_complex`.
    """

    # TODO(rbharath): This is a little tricky in the generalized
    # regime, but we need to find a way to compute the centroid. My
    # idea is that we can compute the centroid of the contact
    # atoms.and use this to recenter all the fragments.
    try:
      fragments = rdkit_util.load_complex(molecular_complex, add_hydrogens=False)

    except MoleculeLoadException:
      logger.warning("This molecule cannot be loaded by Rdkit. Returning None")
      return None
    pairwise_features = []
    # We compute pairwise contact fingerprints
    centroid = compute_contact_centroid(fragments, cutoff=self.cutoff)
    ############################################
    #print("centroid")
    #print(centroid)
    ############################################
    for (frag1, frag2) in itertools.combinations(fragments, 2):
      distances = compute_pairwise_distances(frag1[0], frag2[0])
      ############################################
      #print("np.max(frag1[0])")
      #print(np.max(frag1[0]))
      #print("np.min(frag1[0])")
      #print(np.min(frag1[0]))
      ############################################
      frag1_xyz = subtract_centroid(frag1[0], centroid)
      frag2_xyz = subtract_centroid(frag2[0], centroid)
      xyzs = [frag1_xyz, frag2_xyz]
      #(lig_xyz, lig_rdk), (prot_xyz, prot_rdk) = mol, protein
      #distances = compute_pairwise_distances(prot_xyz, lig_xyz)
      ###########################################
      ##print("np.max(frag1[0])")
      ##print(np.max(frag1[0]))
      ##print("np.min(frag1[0])")
      ##print(np.min(frag1[0]))
      #print("np.max(frag1_xyz)")
      #print(np.max(frag1_xyz))
      #print("np.min(frag1_xyz)")
      #print(np.min(frag1_xyz))
      ###########################################
      # TODO(rbharath): I think the reason this isn't making errors is
      # that it's already computing contact map under the hood which
      # prunes out atoms outside the box
      pairwise_features.append(
          sum([
              voxelize(
                  convert_atom_to_voxel,
                  self.voxels_per_edge,
                  self.box_width,
                  self.voxel_width,
                  hash_ecfp,
                  xyz,
                  feature_dict=ecfp_dict,
                  nb_channel=self.size)
              for xyz, ecfp_dict in zip(xyzs,
                                        featurize_contacts_ecfp(
                                            frag1,
                                            frag2,
                                            distances,
                                            cutoff=self.cutoff,
                                            ecfp_degree=self.radius))
          ])
      )
    #############################################
    #print("[feat.shape for feat in pairwise_features]")
    #print([feat.shape for feat in pairwise_features])
    ############################################
    # Features are of shape (voxels_per_edge, voxels_per_edge, voxels_per_edge, num_feat) so we should concatenate on the last axis.
    return np.concatenate(pairwise_features, axis=-1)
