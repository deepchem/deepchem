"""
Topological fingerprints for macromolecular structures.
"""
import numpy as np
import logging
from deepchem.utils.hash_utils import hash_ecfp
from deepchem.feat import ComplexFeaturizer
from deepchem.utils.hash_utils import vectorize
from deepchem.utils.rdkit_util import compute_all_ecfp
from deepchem.utils.rdkit_util import compute_pairwise_distances

logger = logging.getLogger(__name__)

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


class ContactCircularFingerprint(ComplexFeaturizer):
  """Compute (Morgan) fingerprints near contact points of macromolecular complexes.

  Given a macromolecular complex made up of multiple constituent molecules, first compute the contact points where atoms from different molecules come close to one another. For atoms within "contact regions," compute radial "ECFP" fragments which are sub-molecules centered at atoms in the contact region.
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
        vectorize(
            hash_ecfp, feature_dict=ecfp_dict, size=self.size)
        for ecfp_dict in featurize_binding_pocket_ecfp(
            prot_xyz,
            prot_rdk,
            lig_xyz,
            lig_rdk,
            distances,
            cutoff=self.cutoff,
            ecfp_degree=self.radius)
    ]
