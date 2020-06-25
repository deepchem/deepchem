"""
Atomic coordinate featurizer.
"""
import logging
import numpy as np
import logging
from deepchem.feat import Featurizer
from deepchem.feat import ComplexFeaturizer
from deepchem.utils import rdkit_util, pad_array
from deepchem.utils.rdkit_util import MoleculeLoadException
from deepchem.utils.fragment_util import reduce_molecular_complex_to_contacts

logger = logging.getLogger(__name__)


class AtomicCoordinates(Featurizer):
  """Nx3 matrix of Cartesian coordinates.

  This featurizer featurizes molecules as arrays of atomic
  coordinates. A molecule with `N` atoms will be featurized as a `(N,
  3)` shape matrix.

  This class supports coordinates in both Angstrom and Bohr.  RDKit
  stores atomic coordinates in Angstrom. Atomic unit of length is the
  bohr (1 bohr = 0.529177 Angstrom). Converting units to Bohr makes
  gradient calculation consistent with most QM software packages.
  """
  name = ['atomic_coordinates']

  def __init__(self, use_bohr=True):
    """Initialize atomic coordinates.

    Parameters
    ----------
    use_bohr: bool, optional (Default True)
      If True return coordinates in Bohr, else in Angstrom.
    """
    self.use_bohr = use_bohr

  def _featurize(self, mol):
    """
    Calculate atomic coodinates.

    Parameters
    ----------
    mol : RDKit Mol
          Molecule.

    Returns
    -------
    A Numpy ndarray of shape `(N,3)` in Bohr.
    """
    N = mol.GetNumAtoms()
    coords = np.zeros((N, 3))

    if self.use_bohr:
      # RDKit stores atomic coordinates in Angstrom. Atomic unit of
      # length is the bohr (1 bohr = 0.529177 Angstrom). Converting
      # units makes gradient calculation consistent with most QM
      # software packages.
      coords_in_bohr = [
          mol.GetConformer(0).GetAtomPosition(i).__idiv__(0.52917721092)
          for i in range(N)
      ]

      for atom in range(N):
        coords[atom, 0] = coords_in_bohr[atom].x
        coords[atom, 1] = coords_in_bohr[atom].y
        coords[atom, 2] = coords_in_bohr[atom].z
    else:
      coords = np.array([
          mol.GetConformer(0).GetAtomPosition(i)
          for i in range(N)
      ])

    return coords


def compute_neighbor_list(coords, neighbor_cutoff, max_num_neighbors,
                           periodic_box_size):
  """Computes a neighbor list from atom coordinates.

  This function uses `mdtraj` to compute the neighborlist. You must
  have mdtraj installed to use it.

  Parameters
  ----------
  coords: Numpy array
    Of shape (N, 3) with all the atoms in this systems
  neighbor_cutoff: float
    The neighbor cutoff in angstroms
  max_num_neighbors: int
    The maximum number of neighbors per atom
  periodic_box_size: tuple
    With (x, y, z) box sizes per dimension

  Returns
  -------
  neighbor_list: list
    List of length `N` where `neighbor_list[i]` is the neighbors of
    atom `i`.
  """
  N = coords.shape[0]
  import mdtraj
  traj = mdtraj.Trajectory(coords.reshape((1, N, 3)), None)
  box_size = None
  if periodic_box_size is not None:
    box_size = np.array(periodic_box_size)
    traj.unitcell_vectors = np.array(
        [[[box_size[0], 0, 0], [0, box_size[1], 0], [0, 0, box_size[2]]]],
        dtype=np.float32)
  neighbors = mdtraj.geometry.compute_neighborlist(traj, neighbor_cutoff)
  neighbor_list = {}
  for i in range(N):
    if max_num_neighbors is not None and len(neighbors[i]) > max_num_neighbors:
      delta = coords[i] - coords.take(neighbors[i], axis=0)
      if box_size is not None:
        delta -= np.round(delta / box_size) * box_size
      dist = np.linalg.norm(delta, axis=1)
      sorted_neighbors = list(zip(dist, neighbors[i]))
      sorted_neighbors.sort()
      neighbor_list[i] = [
          sorted_neighbors[j][1] for j in range(max_num_neighbors)
      ]
    else:
      neighbor_list[i] = list(neighbors[i])
  return neighbor_list


def get_coords(mol):
  """
  Gets coordinates in Angstrom for RDKit mol.

  Parameters
  ----------
  mol: rdkit mol
    Molecule to get coordinates for
  """
  N = mol.GetNumAtoms()
  coords = np.zeros((N, 3))

  coords_raw = [mol.GetConformer(0).GetAtomPosition(i) for i in range(N)]
  for atom in range(N):
    coords[atom, 0] = coords_raw[atom].x
    coords[atom, 1] = coords_raw[atom].y
    coords[atom, 2] = coords_raw[atom].z
  return coords


class NeighborListAtomicCoordinates(Featurizer):
  """Featurizes molecules as coordinates and neighbor lists.

  This featurizer takes a molecule and featurizes it as a tuple of
  `(coords, nbr_list)` where `coords` is a `(N, 3)` shaped array of
  coordinates and `nbr_list` is a list of neighbor connectivity
  returned by `compute_neighbor_list`. Neighbors are determined by
  user-defined distance cutoff [in Angstrom].

  Notes
  -----
  <https://en.wikipedia.org/wiki/Cell_list>
  <http://www.cs.cornell.edu/ron/references/1989/Calculations%20of%20a%20List%20of%20Neighbors%20in%20Molecular%20Dynamics%20Si.pdf>
  """

  def __init__(self,
               max_num_neighbors=None,
               neighbor_cutoff=4,
               periodic_box_size=None,
               use_bohr=True):
    """Initializes Neighbor List Featurizer.

    Parameters
    ----------
    max_num_neighbors: int, optional (Default None)
      The maximum number of neighbors allowed for a given atom.
    neighbor_cutoff: float (Default 4.0)
      Threshold distance [Angstroms] for counting neighbors.
    periodic_box_size: np.ndarray, optional (Default None) 
      Dimensions of the periodic box in Angstroms (an `np.ndarray` of
      shape `(3,)`, or None to not use periodic boundary conditions
    use_bohr: bool, optional (Default True)
      If `True` return coordinates in Bohr, else in angstrom.
    """
    if neighbor_cutoff <= 0:
      raise ValueError("neighbor_cutoff must be positive value.")
    if max_num_neighbors is not None:
      if not isinstance(max_num_neighbors, int) or max_num_neighbors <= 0:
        raise ValueError("max_num_neighbors must be positive integer.")
    self.max_num_neighbors = max_num_neighbors
    self.neighbor_cutoff = neighbor_cutoff
    self.periodic_box_size = periodic_box_size
    # Type of data created by this featurizer
    self.dtype = object
    self.coordinates_featurizer = AtomicCoordinates(use_bohr=use_bohr)

  def _featurize(self, mol):
    """
    Compute neighbor list.

    Parameters
    ----------
    mol: rdkit Mol
      To be featurized.

    Returns
    -------
    featurization: Tuple
       A tuple with two elements. The first is the coordinates of the
       molecule, of shape `(N, 3)`. The second is the neighbor list.
    """
    N = mol.GetNumAtoms()
    coords = self.coordinates_featurizer._featurize(mol)[0]
    coords = get_coords(mol)
    neighbor_list = compute_neighbor_list(coords, self.neighbor_cutoff,
                                           self.max_num_neighbors,
                                           self.periodic_box_size)
    featurization = (coords, neighbor_list)
    return featurization 


class NeighborListComplexAtomicCoordinates(ComplexFeaturizer):
  """Featurizes a molecular complex as coordinates and adjacency list of neighbors.
  
  Computes the 3D coordinates of the system and an adjacency list of
  neighbors for protein-ligand complexes in 3-space. Neighbors are
  determined by user-defined distance cutoff.
  """

  def __init__(self, max_num_neighbors=None, neighbor_cutoff=4, use_bohr=True):
    """Initialize this featurizer.

    Parameters
    ----------
    max_num_neighbors: int, optional
      set the maximum number of neighbors
    neighbor_cutoff: int, optional
      The number of neighbors to store in the neighbor list.
    use_bohr: bool, optional (Default True)
      If True return coordinates in Bohr, else in Angstrom.
    """
    if neighbor_cutoff <= 0:
      raise ValueError("neighbor_cutoff must be positive value.")
    if max_num_neighbors is not None:
      if not isinstance(max_num_neighbors, int) or max_num_neighbors <= 0:
        raise ValueError("max_num_neighbors must be positive integer.")
    self.max_num_neighbors = max_num_neighbors
    self.neighbor_cutoff = neighbor_cutoff
    # Type of data created by this featurizer
    self.dtype = object
    self.coordinates_featurizer = AtomicCoordinates(use_bohr=use_bohr)

  def _featurize_complex(self, molecular_complex):
    """
    Compute neighbor list for complex.

    Parameters
    ----------
    molecular_complex: Object
      Some representation of a molecular complex.

    Returns
    -------
    featurization: Tuple
       A tuple with two elements. The first is the coordinates of the
       system, of shape `(N, 3)`. The second is the neighbor list.
    """
    fragments = rdkit_util.load_complex(molecular_complex, add_hydrogens=False)
    #mol_coords, ob_mol = rdkit_util.load_molecule(mol_pdb_file)
    #protein_coords, protein_mol = rdkit_util.load_molecule(protein_pdb_file)
    coords = [frag[0] for frag in fragments]
    mols = [frag[1] for frag in fragments]
    #system_coords = rdkit_util.merge_molecules_xyz(mol_coords, protein_coords)
    system_coords = rdkit_util.merge_molecules_xyz(coords)

    system_neighbor_list = compute_neighbor_list(
        system_coords, self.neighbor_cutoff, self.max_num_neighbors, None)

    return (system_coords, system_neighbor_list)


class AtomicConvFeaturizer(ComplexFeaturizer):
  """This class computes the featurization that corresponds to AtomicConvModel.

  This class computes featurizations needed for AtomicConvModel.
  Given a two molecular structures, it computes a number of
  useful geometric features. In particular, for each molecule
  and the global complex, it computes a coordinates matrix of
  size (N_atoms, 3) where N_atoms is the number of atoms. It
  also computes a neighbor-list, a dictionary with N_atoms
  elements where neighbor-list[i] is a list of the atoms the
  i-th atom has as neighbors. In addition, it computes a
  z-matrix for the molecule which is an array of shape
  (N_atoms,) that contains the atomic number of that atom.

  If `reduce_to_contacts` is set to `True`, only those atoms which are
  in a contact region of the molecular complex are considered. This
  can dramatically reduce the number of atoms that need to be
  featurized so this feature is set to True by default.

  Since the featurization computes these three quantities for
  each of the two molecules and the complex, a total of 9
  quantities are returned for each complex. Note that for
  efficiency, fragments of the molecules can be provided rather
  than the full molecules themselves.
  """

  def __init__(self,
               frag_max_atoms,
               max_num_neighbors,
               neighbor_cutoff=4.0,
               reduce_to_contacts=True,
               cutoff=4.5,
               strip_hydrogens=True):
    """Initialize an AtomicConvFeaturizer object.

    Parameters
    ----------
    frag_max_atoms: int or list[int]
      List of the max number of atoms in each fragment. If int,
      assumes you're setting the same bound for all fragments.
    max_num_neighbors: int
      The maximum number of neighbors allowed
    neighbor_cutoff: float, optional (Default 4.0)
      The distance in angstroms after which neighbors are cutoff.
    reduce_to_contacts: bool, optional
      If True, reduce the atoms in the complex to those near a contact
      region.
    cutoff: float
      The cutoff distance in angstroms. Only used if
      `reduce_to_contacts` is `True`.
    strip_hydrogens: bool, optional
      If True, remove hydrogens before featurizing.
    """
    self.frag_max_atoms = frag_max_atoms
    self.max_num_neighbors = max_num_neighbors
    self.neighbor_cutoff = neighbor_cutoff
    self.reduce_to_contacts = reduce_to_contacts
    self.cutoff = cutoff
    self.strip_hydrogens = strip_hydrogens

  def _featurize_complex(self, molecular_complex):
    """Featurize a single molecular complex.

    Parameters
    ----------
    molecular_complex: Object
      Some representation of a molecular complex.

    Returns
    -------
    """
    # If our upper bound is an int, expand it to a list for each
    # fragment
    if isinstance(self.frag_max_atoms, int):
      frag_max_atoms = [self.frag_max_atoms] * len(molecular_complex)
    else:
      frag_max_atoms = self.frag_max_atoms
    complex_max_atoms = sum(frag_max_atoms)
    frag_coords = []
    frag_mols = []
    try:
      fragments = rdkit_util.load_complex(
          molecular_complex, add_hydrogens=False)

    except MoleculeLoadException:
      logging.warning("This molecule cannot be loaded by Rdkit. Returning None")
      return None
    if self.strip_hydrogens:
      fragments = [
          rdkit_util.strip_hydrogens(frag[0], frag[1]) for frag in fragments
      ]
      #system_coords, system_mol = rdkit_util.strip_hydrogens(system_coords, system_mol)

    if self.reduce_to_contacts:
      fragments = list(reduce_molecular_complex_to_contacts(fragments, self.cutoff))
    ###########################################
    print("fragments")
    print(fragments)
    ###########################################
    coords = [frag[0] for frag in fragments]
    mols = [frag[1] for frag in fragments]
    #system_mol = rdkit_util.merge_molecules(mols)
    system_mol = rdkit_util.merge_molecular_fragments(mols)
    #system_coords = rdkit_util.get_xyz_from_mol(system_mol)
    system_coords = rdkit_util.merge_molecules_xyz(coords)

    try:
      frag_outputs = [
          self.featurize_mol(frag[0], frag[1], frag_max)
          for (frag, frag_max) in zip(fragments, frag_max_atoms)
      ]

      system_outputs = self.featurize_mol(system_coords, system_mol,
                                          complex_max_atoms)
    except ValueError as e:
      logging.warning(
          "max_atoms was set too low. Some complexes too large and skipped")
      return None

    return frag_outputs, system_outputs

  def get_Z_matrix(self, mol, max_atoms):
    """Helper function to make the matrix of atomic-numbers

    Parameters
    ----------
    mol: rdkit mol
      The molecule to featurize.
    max_atoms: int
      The max number of atoms allowed.

    Returns
    -------
    z: np.ndarray
      Numpy array of shape `(max_atoms,)`. The first
      `len(mol.GetAtoms())` entries will contain the atomic numbers
      with the remaining entries zero padded.
    """
    if len(mol.GetAtoms()) > max_atoms:
      raise ValueError("A molecule is larger than permitted by max_atoms. "
                       "Increase max_atoms and try again.")
    z = pad_array(
        np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()]), max_atoms)
    return z

  def featurize_mol(self, coords, mol, max_num_atoms):
    """Helper function to featurize each molecule in complex.

    Parameters
    ----------
    coords: Numpy array
      Shape `(N, 3)` for this molecule
    mol: rdkit mol
      Rdkit mol corresponding to `coords`
    max_num_atoms: int
      Max number of atoms for this molecules.

    Returns
    -------
    features: Tuple
      A tuple with 3 elements `(coords, neighbor_list, z)` where `coords` is a `(N, 3)` array of coordinates, `neighbor_list` is a neighbor-lsit as returned by `compute_neighbor_list` and `z` is returned by `get_Z_matrix()` and is a `np.ndarray` of shape `(max_atoms,)`.
    """
    logger.info("Featurizing molecule of size: %d", len(mol.GetAtoms()))
    neighbor_list = compute_neighbor_list(coords, self.neighbor_cutoff,
                                           self.max_num_neighbors, None)
    z = self.get_Z_matrix(mol, max_num_atoms)
    z = pad_array(z, max_num_atoms)
    coords = pad_array(coords, (max_num_atoms, 3))
    features = coords, neighbor_list, z
    return features


############################# Deprecation warning for old name of AtomicConvFeaturizer ###############################

DEPRECATION = "{} is deprecated and has been renamed to {} and will be removed in DeepChem 3.0."


class ComplexNeighborListFragmentAtomicCoordinates(AtomicConvFeaturizer):

  def __init__(self, *args, **kwargs):

    warnings.warn(
        DEPRECATION.format("ComplexNeighborListFragmentAtomicCoordinates",
                           "AtomicConvFeaturizer"), FutureWarning)

    super(ComplexNeighborListFragmentAtomicCoordinates, self).__init__(
        *args, **kwargs)
