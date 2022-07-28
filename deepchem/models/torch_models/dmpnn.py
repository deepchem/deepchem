import numpy as np
from deepchem.feat import GraphData
from typing import List, Sequence, Optional


class _MapperDMPNN:
  """
  This class is a helper class for DMPNNModel class to generate concatenated feature vector and mappings.

  `self.f_ini_atoms_bonds` is the concatenated feature vector which contains
  concatenation of initial atom and bond features.

  `self.atom_to_incoming_bonds` is mapping from atom index to list of indicies of incoming bonds.

  `self.mapping` is the mapping that maps bond index to 'array of indices of the bonds'
  incoming at the initial atom of the bond (excluding the reverse bonds)

  Example,
  Let the diagram given below represent a molecule containing 3 atoms (nodes) and 2 bonds (edges):-

  |   0 --- 1
  |   |
  |   2

  Here, atoms are => A0, A1 and A2 and their respective feature vectors are f0, f1, and f2.

  Let the bonds from atoms 0->1 ('B[01]') and 1->0 ('B[10]') be considered as 2 different bonds.
  Hence, by considering the same for all atoms, the total number of bonds = 4.

  Let:
    B[01] => B0
    B[10] => B1
    B[02] => B2
    B[20] => B3

  Hence repective bond features are fb0, fb1, fb2, and fb3.
  (Note: fb0 = fb1, fb2 = fb3)

  'f_ini_atoms_bonds' is the concatenated feature vector which contains
  concatenation of initial atom and bond features.

                   'B0'
    Example: 'A0' -----> A1 , concat feature = f0 + fb0

  Hence,
                            B0       B1       B2       B3     B(-1)
    f_ini_atoms_bonds = [ f0+fb0 , f1+fb1 , f0+fb2 , f2+fb3 , f(-1) ]

    (Note: f(-1) is a zero array of the same size as other concatenated features.)

  `atom_to_incoming_bonds` is mapping from atom index to list of indicies of incoming bonds.

                   B3          B1
    Example: 'A2' ----> 'A0' <---- 'A1', for A0 => [B1, B3]

  Hence,
                                 A0        A1        A2
    atom_to_incoming_bonds = [ [B1,B3] [B0,B(-1)] [B2,B(-1)] ]

    (Note: Here, maximum number of incoming bonds is 2. So, -1 index is added to all those cases
           where number of incoming bonds is less than maximum. In this case, its for A1 and A2.)

  To get mapping, first find indices of the bonds, incoming at the initial atom of the bond.

    Example: for bond B0, B1 and B3 are coming towards atom 0.

  |                    B0             B1
  |                 0 ----> 1  |  0 <---- 1
  |                            |  ^
  |                            |  | B3
  |                            |  2

                                     B0     B1    B2     B3
  mapping (with reverse bonds) = [ [B1,B3] [B0] [B1,B3] [B2] ]

  To get the required mapping, reverse bond indices are replaced with -1
  and extra space in the array elements is filled with -1, to get a uniform array.

  The mapping is also padded with -1 at the end, so that the length of `mapping` is
  equal to the length of `f_ini_atoms_bonds`.

  Hence,
                    B0          B1          B2            B3          B(-1)
    mapping = [ [B(-1),B3] [B(-1),B(-1)] [B1,B(-1)] [B(-1),B(-1)] [B(-1),B(-1)] ]

    OR

    mapping = [[-1, 3], [-1, -1], [1, -1], [-1, -1], [-1, -1]]
  """

  def __init__(self, graph: GraphData):
    """
    Parameters
    ----------
    graph: GraphData
      GraphData object.
    """
    self.num_atoms: int = graph.num_nodes
    self.num_atom_features: int = graph.num_node_features
    self.num_bonds: int = graph.num_edges
    self.num_bond_features: int = graph.num_edge_features
    self.atom_features: np.ndarray = graph.node_features
    self.bond_features: Optional[np.ndarray] = graph.edge_features
    self.bond_index: np.ndarray = graph.edge_index
    self.global_features: np.ndarray = graph.global_features  # type: ignore
    # mypy check is ignored for global_features as it is not a default attribute
    # of GraphData. It is created during runtime using **kwargs.

    # mapping from bond index to the index of the atom (where the bond is coming from)
    self.bond_to_ini_atom: np.ndarray

    # mapping from bond index to concat(in_atom, bond) features
    self.f_ini_atoms_bonds: np.ndarray = np.empty(0)

    # mapping from atom index to list of indicies of incoming bonds
    self.atom_to_incoming_bonds: np.ndarray

    # mapping which maps bond index to 'array of indices of the bonds' incoming at the initial atom of the bond (excluding the reverse bonds)
    self.mapping: np.ndarray = np.empty(0)

    if self.num_bonds == 0:
      self.bond_to_ini_atom = np.empty(0)
      self.f_ini_atoms_bonds = np.zeros(
          (1, self.num_atom_features + self.num_bond_features))

      self.atom_to_incoming_bonds = np.asarray([[-1]], dtype=int)
      self.mapping = np.asarray([[-1]], dtype=int)

    else:
      self.bond_to_ini_atom = self.bond_index[0]
      self._get_f_ini_atoms_bonds()  # its zero padded at the end

      self.atom_to_incoming_bonds = self._get_atom_to_incoming_bonds()
      self._generate_mapping()  # its padded with -1 at the end

  @property
  def values(self) -> Sequence[np.ndarray]:
    """
    Returns the required mappings:
    - atom features
    - concat features (atom + bond)
    - atom to incoming bonds mapping
    - mapping
    - global features
    """
    return self.atom_features, self.f_ini_atoms_bonds, self.atom_to_incoming_bonds, self.mapping, self.global_features

  def _get_f_ini_atoms_bonds(self):
    """
    Method to get `self.f_ini_atoms_bonds`
    """
    self.f_ini_atoms_bonds = np.hstack(
        (self.atom_features[self.bond_to_ini_atom], self.bond_features))

    # zero padded at the end
    self.f_ini_atoms_bonds = np.pad(self.f_ini_atoms_bonds, ((0, 1), (0, 0)))

  def _generate_mapping(self):
    """
    Generate mapping, which maps bond index to 'array of indices of the bonds'
    incoming at the initial atom of the bond (reverse bonds are not considered).

    Steps:
    - Get mapping based on `self.atom_to_incoming_bonds` and `self.bond_to_ini_atom`.
    - Replace reverse bond indices with -1.
    - Pad the mapping with -1.
    """

    # get mapping which maps bond index to 'array of indices of the bonds' incoming at the initial atom of the bond
    self.mapping = self.atom_to_incoming_bonds[self.bond_to_ini_atom]
    self._replace_rev_bonds()

    # padded with -1 at the end
    self.mapping = np.pad(self.mapping, ((0, 1), (0, 0)), constant_values=-1)

  def _get_atom_to_incoming_bonds(self) -> np.ndarray:
    """
    Method to get atom_to_incoming_bonds mapping
    """
    # mapping from bond index to the index of the atom (where the bond if going to)
    bond_to_final_atom: np.ndarray = self.bond_index[1]

    # mapping from atom index to list of indicies of incoming bonds
    a2b: List = []
    for i in range(self.num_atoms):
      a2b.append(list(np.where(bond_to_final_atom == i)[0]))

    # get maximum number of incoming bonds
    max_num_bonds: int = max(1,
                             max(len(incoming_bonds) for incoming_bonds in a2b))

    # Make number of incoming bonds equal to maximum number of bonds.
    # This is done by appending -1 to fill remaining space at each atom indices.
    a2b = [
        a2b[a] + [-1] * (max_num_bonds - len(a2b[a]))
        for a in range(self.num_atoms)
    ]

    return np.asarray(a2b, dtype=int)

  def _replace_rev_bonds(self):
    """
    Method to get b2revb and replace the reverse bond indices with -1 in mapping.
    """
    # mapping from bond index to the index of the reverse bond
    b2revb: np.ndarray = np.empty(self.num_bonds, dtype=int)
    for i in range(self.num_bonds):
      if i % 2 == 0:
        b2revb[i] = i + 1
      else:
        b2revb[i] = i - 1

    for count, i in enumerate(b2revb):
      self.mapping[count][np.where(self.mapping[count] == i)] = -1
