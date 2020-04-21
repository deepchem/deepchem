"""
Compute various spatial fingerprints for macromolecular complexes.
"""
import logging
from deepchem.utils.rdkit_util import get_partial_charge
from deepchem.feat import ComplexFeaturizer
from deepchem.utils.voxel_utils import voxelize 
from deepchem.utils.rdkit_util import compute_salt_bridges
from deepchem.utils.rdkit_util import compute_binding_pocket_cation_pi
from deepchem.utils.voxel_utils import convert_atom_to_voxel
from deepchem.utils.voxel_utils import convert_atom_pair_to_voxel
from deepchem.utils.rdkit_util import compute_pairwise_distances
from deepchem.utils.rdkit_util import compute_pi_stack

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

  Given a macromolecular complex made up of multiple
  constitutent molecules, compute the partial (Gasteiger
  charge) on each molecule. For each atom, localize this
  partial charge in the voxel in which it originated to create
  a local charge array. Sum contributions to get an effective
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

class SaltBridgeVoxelizer(ComplexFeaturizer):
  """Localize salt bridges between atoms in macromolecular complexes.

  Given a macromolecular complex made up of multiple
  constitutent molecules, compute salt bridges between atoms in
  the macromolecular complex. For each atom, localize this salt
  bridge in the voxel in which it originated to create a local
  salt bridge array. Note that if atoms in two different voxels
  interact in a salt-bridge, the interaction is double counted
  in both voxels.

  Creates a tensor output of shape (box_width, box_width,
  box_width, 1) for each macromolecular complex that computes
  the number of salt bridges at each voxel.
  """
  def __init__(self, 
               cutoff=5.0,
               box_width=16.0,
               voxel_width=1.0):
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
    distances = compute_pairwise_distances(prot_xyz, lig_xyz)
    return [
        voxelize(
            convert_atom_pair_to_voxel,
            self.voxels_per_edge,
            self.box_width,
            self.voxel_width,
            None, (prot_xyz, lig_xyz),
            feature_list=compute_salt_bridges(
                prot_rdk,
                lig_rdk,
                distances,
                cutoff=self.cutoff),
            nb_channel=1)
    ]

class CationPiVoxelizer(ComplexFeaturizer):
  """Localize cation-Pi interactions between atoms in macromolecular complexes.

  Given a macromolecular complex made up of multiple
  constitutent molecules, compute cation-Pi between atoms in
  the macromolecular complex. For each atom, localize this salt
  bridge in the voxel in which it originated to create a local
  cation-pi array.

  Creates a tensor output of shape (box_width, box_width,
  box_width, 1) for each macromolecular complex that computes
  the number of cation-pi interactions at each voxel.
  """
  def __init__(self, 
               distance_cutoff=6.5,
               angle_cutoff=30.0,
               box_width=16.0,
               voxel_width=1.0):
    """
    Parameters
    ----------
    distance_cutoff: float, optional (default 6.5)
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
    self.distance_cutoff = distance_cutoff
    self.angle_cutoff = angle_cutoff
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
    distances = compute_pairwise_distances(prot_xyz, lig_xyz)
    return [
        sum([
            voxelize(
                convert_atom_to_voxel,
                self.voxels_per_edge,
                self.box_width,
                self.voxel_width,
                None,
                xyz,
                feature_dict=cation_pi_dict,
                nb_channel=1) for xyz, cation_pi_dict in zip(
                    (prot_xyz, lig_xyz),
                    compute_binding_pocket_cation_pi(
                        prot_rdk,
                        lig_rdk,
                        dist_cutoff=self.distance_cutoff,
                        angle_cutoff=self.angle_cutoff,
                    ))
        ])
    ]

class PiStackVoxelizer(ComplexFeaturizer):
  """Localize Pi stacking interactions between atoms in macromolecular complexes.

  Given a macromolecular complex made up of multiple
  constitutent molecules, compute pi-stacking interactions
  between atoms in the macromolecular complex. For each atom,
  localize this salt bridge in the voxel in which it originated
  to create a local pi-stacking array.

  Creates a tensor output of shape (box_width, box_width,
  box_width, 2) for each macromolecular complex. Each voxel has
  2 fields, with the first tracking the number of pi-pi
  parallel interactions, and the second tracking the number of
  pi-T interactions.
  """
  def __init__(self, 
               distance_cutoff=4.4,
               angle_cutoff=30.0,
               box_width=16.0,
               voxel_width=1.0):
    """
    Parameters
    ----------
    distance_cutoff: float, optional (default 4.4)
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
    self.distance_cutoff = distance_cutoff
    self.angle_cutoff = angle_cutoff
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
    distances = compute_pairwise_distances(prot_xyz, lig_xyz)
    protein_pi_t, protein_pi_parallel, ligand_pi_t, ligand_pi_parallel = (
        compute_pi_stack(
             prot_rdk,
             lig_rdk,
             distances,
             dist_cutoff=self.distance_cutoff,
             angle_cutoff=self.angle_cutoff))
    pi_parallel_tensor = voxelize(
        convert_atom_to_voxel,
        self.voxels_per_edge,
        self.box_width,
        self.voxel_width,
        None,
        prot_xyz,
        feature_dict=protein_pi_parallel,
        nb_channel=1)
    pi_parallel_tensor += voxelize(
        convert_atom_to_voxel,
        self.voxels_per_edge,
        self.box_width,
        self.voxel_width,
        None,
        lig_xyz,
        feature_dict=ligand_pi_parallel,
        nb_channel=1)

    pi_t_tensor = voxelize(
        convert_atom_to_voxel,
        self.voxels_per_edge,
        self.box_width,
        self.voxel_width,
        None,
        prot_xyz,
        feature_dict=protein_pi_t,
        nb_channel=1)
    pi_t_tensor += voxelize(
        convert_atom_to_voxel,
        self.voxels_per_edge,
        self.box_width,
        self.voxel_width,
        None,
        lig_xyz,
        feature_dict=ligand_pi_t,
        nb_channel=1)
    return [pi_parallel_tensor, pi_t_tensor]
    
class HydrogenBondVoxelizer(ComplexFeaturizer):
  """Localize hydrogen bonds between atoms in macromolecular complexes.

  Given a macromolecular complex made up of multiple
  constitutent molecules, compute hydrogen bonds between atoms
  in the macromolecular complex. For each atom, localize this
  hydrogen bond in the voxel in which it originated to create a
  local hydrogen bond array. Note that if atoms in two
  different voxels interact in a hydrogen bond, the interaction
  is double counted in both voxels.

  Creates a tensor output of shape (box_width, box_width,
  box_width, 1) for each macromolecular complex that computes
  the number of hydrogen bonds at each voxel.
  """
  def __init__(self, 
               distance_cutoff=5.0,
               angle_cutoff=40.0,
               box_width=16.0,
               voxel_width=1.0):
    """
    Parameters
    ----------
    distance_cutoff: float, optional (default 4.0)
      The distance in angstroms within which atoms must be to
      be considered for a hydrogen bond interaction between
      them.
    angle_cutoff: float, optional (default 40.0)
      Angle cutoff. Max allowed deviation from the ideal (180
      deg) angle between hydrogen-atom1, hydrogen-atom2 vectors.
    box_width: float, optional (default 16.0)
      Size of a box in which voxel features are calculated. Box
      is centered on a ligand centroid.
    voxel_width: float, optional (default 1.0)
      Size of a 3D voxel in a grid.
    """
    self.distance_cutoff = distance_cutoff
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
    distances = compute_pairwise_distances(prot_xyz, lig_xyz)
    return [
        voxelize(
            convert_atom_pair_to_voxel,
            self.voxels_per_edge,
            self.box_width,
            self.voxel_width,
            None, (prot_xyz, lig_xyz),
            feature_list=hbond_list,
            nb_channel=1) for hbond_list in compute_hydrogen_bonds(
                prot_xyz, prot_rdk, lig_xyz, lig_rdk,
                distances, self.distance_cutoff,
                self.angle_cutoff)
    ]
