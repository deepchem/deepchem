"""
Compute various spatial fingerprints for macromolecular complexes.
"""
import logging
from deepchem.utils.rdkit_util import get_partial_charge
from deepchem.feat import ComplexFeaturizer
from deepchem.utils.voxel_utils import voxelize 

logger = logging.getLogger(__name__)

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

  Given a macromolecular compelx made up of multiple
  constitutent molecules, compute the partial (Gasteiger
  charge) on each molecule. For each atom, localize this
  partial charge in the voxel in which it originated to create
  a local charge array. Sum contributes to get an effective
  charge at each voxel.

  Creates a tensor output of shape (box_width, box_width,
  box_width, 1) for each macromolecular complex that computes
  the effective charge at each voxel.
  """
  def __init__(self, 
               box_width=16.0,
               voxel_width=1.0):
    """
    Parameters
    ----------
    box_width: float, optional (default 16.0)
      Size of a box in which voxel features are calculated. Box
      is centered on a ligand centroid.
    voxel_width: float, optional (default 1.0)
      Size of a 3D voxel in a grid.
    """
    self.cutoff = cutoff
    self.box_width = box_width
    self.voxel_width = voxel_width
    self.voxels_per_edge = int(self.box_width / self.voxel_width)

  def _featurize_complex(self, mol, protein):
    """
    Compute featurization for a single mol/protein complex

    Parameters
    ----------
    mol: object
      Representation of the molecule
    protein: object
      Representation of the protein
    """
    (lig_xyz, lig_rdk), (prot_xyz, prot_rdk) = mol, protein
    return [
        sum([
            voxelize(
                convert_atom_to_voxel,
                self.voxels_per_edge,
                self.box_width,
                self.voxel_width,
                None,
                xyz,
                feature_dict=compute_charge_dictionary(mol),
                nb_channel=1,
                dtype="np.float16")
            for xyz, mol in ((prot_xyz, prot_rdk), (lig_xyz, lig_rdk))
        ])
    ]
