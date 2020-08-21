from typing import List, Optional, Sequence, Tuple, Union
import numpy as np

from deepchem.utils.typing import RDKitAtom, RDKitBond, RDKitMol
from deepchem.utils.graph_conv_utils import get_atom_type_one_hot, get_atomic_number, \
  construct_hydrogen_bonding_info, get_atom_hydrogen_bonding_one_hot, \
  get_atom_is_in_aromatic_one_hot, get_atom_hybridization_one_hot, \
  get_atom_total_num_Hs, get_atom_chirality_one_hot, get_atom_formal_charge, \
  get_atom_partial_charge, get_atom_ring_size_one_hot, get_bond_type_one_hot, \
  get_bond_is_in_same_ring_one_hot, get_bond_graph_distance_one_hot, \
  get_bond_euclidean_distance
from deepchem.feat.base_classes import MolecularFeaturizer
from deepchem.feat.graph_data import GraphData


def constrcut_atom_feature(
    atom: RDKitAtom,
    use_mpnn_style: bool,
    hydrogen_bonding: List[Tuple[int, str]],
    chiral_center: Optional[List[Tuple[int, str]]] = None,
    sssr: Optional[Sequence] = None) -> List[Union[int, float]]:
  """TODO: add docstring"""

  # common feature
  atom_type = get_atom_type_one_hot(atom)
  aromatic = get_atom_is_in_aromatic_one_hot(atom)
  hybridization = get_atom_hybridization_one_hot(atom)
  acceptor_donor_one_hot = get_atom_hydrogen_bonding_one_hot(
      atom, hydrogen_bonding)

  if use_mpnn_style:
    # MPNN style atom vecotor
    atomic_number = get_atomic_number(atom)
    num_Hs = get_atom_total_num_Hs(atom)
    return atom_type + atomic_number + acceptor_donor_one_hot + aromatic + \
      hybridization + num_Hs

  # Weave style atom vector
  if sssr is None or chiral_center is None:
    raise ValueError("Must set the values to `sssr` and `chiral_center`.")

  chirality = get_atom_chirality_one_hot(atom, chiral_center)
  formal_charge = get_atom_formal_charge(atom)
  partial_charge = get_atom_partial_charge(atom)
  ring_size = get_atom_ring_size_one_hot(atom, sssr)
  return atom_type + chirality + formal_charge + partial_charge + \
    ring_size + hybridization + acceptor_donor_one_hot + aromatic


def construct_bond_feature(
    bond: RDKitBond,
    use_mpnn_style: bool,
    graph_dist_matrix: Optional[np.ndarray] = None,
    euclidean_dist_matrix: Optional[np.ndarray] = None,
) -> List[Union[int, float]]:
  """TODO: add docstring"""

  # common feature
  bond_type = get_bond_type_one_hot(bond)

  if use_mpnn_style:
    # MPNN style bond vecotor
    if euclidean_dist_matrix is None:
      raise ValueError("Must set the value to `euclidean_dist_matrix`.")
    euclidean_distance = get_bond_euclidean_distance(bond,
                                                     euclidean_dist_matrix)
    return bond_type + euclidean_distance

  # Weave style atom vector
  if graph_dist_matrix is None:
    raise ValueError("Must set the value to `graph_dist_matrix`.")
  graph_distance = get_bond_graph_distance_one_hot(bond, graph_dist_matrix)
  same_ring = get_bond_is_in_same_ring_one_hot(bond)
  return bond_type + graph_distance + same_ring


class MolGraphConvFeaturizer(MolecularFeaturizer):
  """This class is a featurizer of gerneral graph convolution networks for molecules.

  The default featurization is based on WeaveNet style edge and node annotation.

  TODO: add more docstrings.

  Examples
  -------
  >>> smiles = ["C1CCC1", "C1=CC=CN=C1"]
  >>> featurizer = MolGraphConvFeaturizer()
  >>> out = featurizer.featurize(smiles)
  >>> type(out[0])
  <class 'deepchem.feat.graph_data.GraphData'>
  """

  def __init__(self, add_self_loop: bool = False, use_mpnn_style: bool = False):
    """
    Paramters
    ---------
    add_self_loop: bool, default False
      TODO: Docstring
    use_mpnn_style: bool, default False
      TODO: Docstring
    """
    self.add_self_loop = add_self_loop
    self.use_mpnn_style = use_mpnn_style

  def _featurize(self, mol: RDKitMol) -> GraphData:
    """Calculate molecule graph features from RDKit mol object.

    Parametrs
    ---------
    mol: rdkit.Chem.rdchem.Mol
      RDKit mol object.

    Returns
    -------
    graph: GraphData
      A molecule graph with some features.
    """
    try:
      from rdkit import Chem
      from rdkit.Chem import rdmolops, AllChem
    except ModuleNotFoundError:
      raise ValueError("This method requires RDKit to be installed.")

    # construct atom and bond features
    hydrogen_bonding = construct_hydrogen_bonding_info(mol)
    if self.use_mpnn_style:
      # MPNN style
      # compute 3D coordinate. Sometimes, this operation raise Error
      mol_for_coord = AllChem.AddHs(mol)
      conf_id = AllChem.EmbedMolecule(mol_for_coord)
      mol_for_coord = AllChem.RemoveHs(mol_for_coord)
      dist_matrix = rdmolops.Get3DDistanceMatrix(mol_for_coord, confId=conf_id)

      # construct atom (node) feature
      atom_features = np.array(
          [
              constrcut_atom_feature(atom, self.use_mpnn_style,
                                     hydrogen_bonding)
              for atom in mol.GetAtoms()
          ],
          dtype=np.float,
      )

      # construct edge (bond) information
      src, dist, bond_features = [], [], []
      for bond in mol.GetBonds():
        # add edge list considering a directed graph
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        src += [start, end]
        dist += [end, start]
        bond_features += 2 * [
            construct_bond_feature(
                bond, self.use_mpnn_style, euclidean_dist_matrix=dist_matrix)
        ]

      if self.add_self_loop:
        src += [i for i in range(mol.GetNumAtoms())]
        dist += [i for i in range(mol.GetNumAtoms())]
        bond_fea_length = len(bond_features[0])
        bond_features += 2 * [[0 for _ in range(bond_fea_length)]]

      return GraphData(
          node_features=atom_features,
          edge_index=np.array([src, dist], dtype=np.int),
          edge_features=np.array(bond_features, dtype=np.float))

    # Weave style
    # compute partial charges
    AllChem.ComputeGasteigerCharges(mol)
    dist_matrix = Chem.GetDistanceMatrix(mol)
    chiral_center = Chem.FindMolChiralCenters(mol)
    sssr = Chem.GetSymmSSSR(mol)

    # construct atom (node) feature
    atom_features = np.array(
        [
            constrcut_atom_feature(atom, self.use_mpnn_style, hydrogen_bonding,
                                   chiral_center, sssr)
            for atom in mol.GetAtoms()
        ],
        dtype=np.float,
    )

    # construct edge (bond) information
    src, dist, bond_features = [], [], []
    for bond in mol.GetBonds():
      # add edge list considering a directed graph
      start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
      src += [start, end]
      dist += [end, start]
      bond_features += 2 * [
          construct_bond_feature(
              bond, self.use_mpnn_style, graph_dist_matrix=dist_matrix)
      ]

    if self.add_self_loop:
      src += [i for i in range(mol.GetNumAtoms())]
      dist += [i for i in range(mol.GetNumAtoms())]
      bond_fea_length = len(bond_features[0])
      bond_features += 2 * [[0 for _ in range(bond_fea_length)]]

    return GraphData(
        node_features=atom_features,
        edge_index=np.array([src, dist], dtype=np.int),
        edge_features=np.array(bond_features, dtype=np.float))
