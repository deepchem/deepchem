"""
SPLIF Fingeprints for protein-ligand complexes.
"""
import logging
import numpy as np
from deepchem.utils.hash_utils import hash_ecfp_pair
from deepchem.utils.rdkit_util import compute_all_ecfp
from deepchem.feat import ComplexFeaturizer
from deepchem.utils.hash_utils import vectorize
from deepchem.utils.rdkit_util import compute_pairwise_distances

logger = logging.getLogger(__name__)

SPLIF_CONTACT_BINS = [(0, 2.0), (2.0, 3.0), (3.0, 4.5)]

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

def featurize_splif(protein_xyz, protein, ligand_xyz, ligand,
contact_bins,
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



class SplifFingerprint(ComplexFeaturizer):
  """Computes SPLIF Fingerprints for a macromolecular complex.

  SPLIF fingerprints are based on a technique introduced in the
  following paper. 

  Da, C., and D. Kireev. "Structural protein–ligand interaction
  fingerprints (SPLIF) for structure-based virtual screening:
  method and benchmark study." Journal of chemical information
  and modeling 54.9 (2014): 2555-2561.

  SPLIF fingerprints are a subclass of `ComplexFeaturizer`. It
  requires 3D coordinates for a protein-ligand complex. For
  each ligand atom, it identifies close protein atoms. These
  atom pairs are expanded to 2D circular fragments and a
  fingerprint for the union is turned on in the bit vector.
  """

  def __init__(self, 
               contact_bins=None,
               radius=2,
               size=8):
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

  def _featurize_complex(self, mol, protein):
    """
    Compute featurization for a single mol/protein complex

    TODO(rbharath): This is very not ergonomic. I'd much prefer
    returning an vector instead of a list of two vectors. In
    addition, there's a question of efficiency.
    RdkitGridFeaturizer caches rotated versions etc internally.
    To make things work out of box, we are accepting that
    kludgey input. This needs to be cleaned up before full
    merge.

    Parameters
    ----------
    mol: object
      Representation of the molecule
    protein: object
      Representation of the protein
    """
    (lig_xyz, lig_rdk), (prot_xyz, prot_rdk) = mol, protein
    distances = compute_pairwise_distances(prot_xyz, lig_xyz)
    return [
        vectorize(hash_ecfp_pair, feature_dict=splif_dict,
            size=self.size) for splif_dict in featurize_splif(
                prot_xyz, prot_rdk, lig_xyz, lig_rdk, self.contact_bins, distances, self.radius)
    ]
