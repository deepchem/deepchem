"""
Atomic coordinate featurizer.
"""
import logging
import numpy as np
from deepchem.feat import Featurizer
from deepchem.feat import ComplexFeaturizer
from deepchem.utils import pad_array
from deepchem.utils.rdkit_utils import MoleculeLoadException, get_xyz_from_mol, \
  load_molecule, merge_molecules_xyz, merge_molecules


class AtomicCoordinates(Featurizer):
  """
  Nx3 matrix of Cartesian coordinates [Angstrom]
  """
  name = ['atomic_coordinates']

  def _featurize(self, mol):
    """
    Calculate atomic coodinates.

    Parameters
    ----------
    mol : RDKit Mol
          Molecule.
    """

    N = mol.GetNumAtoms()
    coords = np.zeros((N, 3))

    # RDKit stores atomic coordinates in Angstrom. Atomic unit of length is the
    # bohr (1 bohr = 0.529177 Angstrom). Converting units makes gradient calculation
    # consistent with most QM software packages.
    coords_in_bohr = [
        mol.GetConformer(0).GetAtomPosition(i).__idiv__(0.52917721092)
        for i in range(N)
    ]

    for atom in range(N):
      coords[atom, 0] = coords_in_bohr[atom].x
      coords[atom, 1] = coords_in_bohr[atom].y
      coords[atom, 2] = coords_in_bohr[atom].z

    coords = [coords]
    return coords


def compute_neighbor_list(coords, neighbor_cutoff, max_num_neighbors,
                          periodic_box_size):
  """Computes a neighbor list from atom coordinates."""
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
  """
  Adjacency List of neighbors in 3-space

  Neighbors determined by user-defined distance cutoff [in Angstrom].

  https://en.wikipedia.org/wiki/Cell_list
  Ref: http://www.cs.cornell.edu/ron/references/1989/Calculations%20of%20a%20List%20of%20Neighbors%20in%20Molecular%20Dynamics%20Si.pdf

  Parameters
  ----------
  neighbor_cutoff: float
    Threshold distance [Angstroms] for counting neighbors.
  periodic_box_size: 3 element array
    Dimensions of the periodic box in Angstroms, or None to not use periodic boundary conditions
  """

  def __init__(self,
               max_num_neighbors=None,
               neighbor_cutoff=4,
               periodic_box_size=None):
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
    self.coordinates_featurizer = AtomicCoordinates()

  def _featurize(self, mol):
    """
    Compute neighbor list.

    Parameters
    ----------
      mol: rdkit Mol
        To be featurized.
    """
    N = mol.GetNumAtoms()
    # TODO(rbharath): Should this return a list?
    bohr_coords = self.coordinates_featurizer._featurize(mol)[0]
    coords = get_coords(mol)
    neighbor_list = compute_neighbor_list(coords, self.neighbor_cutoff,
                                          self.max_num_neighbors,
                                          self.periodic_box_size)
    return (bohr_coords, neighbor_list)


class NeighborListComplexAtomicCoordinates(ComplexFeaturizer):
  """
  Adjacency list of neighbors for protein-ligand complexes in 3-space.

  Neighbors dtermined by user-dfined distance cutoff.
  """

  def __init__(self, max_num_neighbors=None, neighbor_cutoff=4):
    if neighbor_cutoff <= 0:
      raise ValueError("neighbor_cutoff must be positive value.")
    if max_num_neighbors is not None:
      if not isinstance(max_num_neighbors, int) or max_num_neighbors <= 0:
        raise ValueError("max_num_neighbors must be positive integer.")
    self.max_num_neighbors = max_num_neighbors
    self.neighbor_cutoff = neighbor_cutoff
    # Type of data created by this featurizer
    self.dtype = object
    self.coordinates_featurizer = AtomicCoordinates()

  def _featurize(self, mol_pdb_file, protein_pdb_file):
    """
    Compute neighbor list for complex.

    Parameters
    ----------
    mol_pdb_file: Str 
      Filename for ligand pdb file. 
    protein_pdb_file: Str 
      Filename for protein pdb file. 
    """
    mol_coords, ob_mol = load_molecule(mol_pdb_file)
    protein_coords, protein_mol = load_molecule(protein_pdb_file)
    system_coords = merge_molecules_xyz([mol_coords, protein_coords])

    system_neighbor_list = compute_neighbor_list(
        system_coords, self.neighbor_cutoff, self.max_num_neighbors, None)

    return (system_coords, system_neighbor_list)


class ComplexNeighborListFragmentAtomicCoordinates(ComplexFeaturizer):
  """This class computes the featurization that corresponds to AtomicConvModel.

  This class computes featurizations needed for AtomicConvModel. Given a
  two molecular structures, it computes a number of useful geometric
  features. In particular, for each molecule and the global complex, it
  computes a coordinates matrix of size (N_atoms, 3) where N_atoms is the
  number of atoms. It also computes a neighbor-list, a dictionary with
  N_atoms elements where neighbor-list[i] is a list of the atoms the i-th
  atom has as neighbors. In addition, it computes a z-matrix for the
  molecule which is an array of shape (N_atoms,) that contains the atomic
  number of that atom.

  Since the featurization computes these three quantities for each of the
  two molecules and the complex, a total of 9 quantities are returned for
  each complex. Note that for efficiency, fragments of the molecules can be
  provided rather than the full molecules themselves.
  """

  def __init__(self,
               frag1_num_atoms,
               frag2_num_atoms,
               complex_num_atoms,
               max_num_neighbors,
               neighbor_cutoff,
               strip_hydrogens=True):
    self.frag1_num_atoms = frag1_num_atoms
    self.frag2_num_atoms = frag2_num_atoms
    self.complex_num_atoms = complex_num_atoms
    self.max_num_neighbors = max_num_neighbors
    self.neighbor_cutoff = neighbor_cutoff
    self.strip_hydrogens = strip_hydrogens
    self.neighborlist_featurizer = NeighborListComplexAtomicCoordinates(
        self.max_num_neighbors, self.neighbor_cutoff)

  def _featurize(self, mol_pdb_file, protein_pdb_file):
    try:
      frag1_coords, frag1_mol = load_molecule(
          mol_pdb_file, is_protein=False, sanitize=True, add_hydrogens=False)
      frag2_coords, frag2_mol = load_molecule(
          protein_pdb_file, is_protein=True, sanitize=True, add_hydrogens=False)
    except MoleculeLoadException:
      # Currently handles loading failures by returning None
      # TODO: Is there a better handling procedure?
      logging.warning("Some molecules cannot be loaded by Rdkit. Skipping")
      return None
    system_mol = merge_molecules([frag1_mol, frag2_mol])
    system_coords = get_xyz_from_mol(system_mol)

    frag1_coords, frag1_mol = self._strip_hydrogens(frag1_coords, frag1_mol)
    frag2_coords, frag2_mol = self._strip_hydrogens(frag2_coords, frag2_mol)
    system_coords, system_mol = self._strip_hydrogens(system_coords, system_mol)

    try:
      frag1_coords, frag1_neighbor_list, frag1_z = self.featurize_mol(
          frag1_coords, frag1_mol, self.frag1_num_atoms)

      frag2_coords, frag2_neighbor_list, frag2_z = self.featurize_mol(
          frag2_coords, frag2_mol, self.frag2_num_atoms)

      system_coords, system_neighbor_list, system_z = self.featurize_mol(
          system_coords, system_mol, self.complex_num_atoms)
    except ValueError as e:
      logging.warning(
          "max_atoms was set too low. Some complexes too large and skipped")
      return None

    return frag1_coords, frag1_neighbor_list, frag1_z, frag2_coords, frag2_neighbor_list, frag2_z, \
           system_coords, system_neighbor_list, system_z

  def get_Z_matrix(self, mol, max_atoms):
    if len(mol.GetAtoms()) > max_atoms:
      raise ValueError("A molecule is larger than permitted by max_atoms. "
                       "Increase max_atoms and try again.")
    return pad_array(
        np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()]), max_atoms)

  def featurize_mol(self, coords, mol, max_num_atoms):
    logging.info("Featurizing molecule of size: %d", len(mol.GetAtoms()))
    neighbor_list = compute_neighbor_list(coords, self.neighbor_cutoff,
                                          self.max_num_neighbors, None)
    z = self.get_Z_matrix(mol, max_num_atoms)
    z = pad_array(z, max_num_atoms)
    coords = pad_array(coords, (max_num_atoms, 3))
    return coords, neighbor_list, z

  def _strip_hydrogens(self, coords, mol):

    class MoleculeShim(object):
      """
      Shim of a Molecule which supports #GetAtoms()
      """

      def __init__(self, atoms):
        self.atoms = [AtomShim(x) for x in atoms]

      def GetAtoms(self):
        return self.atoms

    class AtomShim(object):

      def __init__(self, atomic_num):
        self.atomic_num = atomic_num

      def GetAtomicNum(self):
        return self.atomic_num

    if not self.strip_hydrogens:
      return coords, mol
    indexes_to_keep = []
    atomic_numbers = []
    for index, atom in enumerate(mol.GetAtoms()):
      if atom.GetAtomicNum() != 1:
        indexes_to_keep.append(index)
        atomic_numbers.append(atom.GetAtomicNum())
    mol = MoleculeShim(atomic_numbers)
    coords = coords[indexes_to_keep]
    return coords, mol
