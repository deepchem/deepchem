from rdkit import Chem
import numpy as np
import logging
from typing import List, Tuple, Union, Dict, Set, Sequence, Optional
from deepchem.utils.typing import RDKitAtom, RDKitMol, RDKitBond

from deepchem.feat.base_classes import MolecularFeaturizer
from deepchem.feat.graph_data import GraphData
from deepchem.feat.molecule_featurizers.circular_fingerprint import CircularFingerprint
from deepchem.feat.molecule_featurizers.rdkit_descriptors import RDKitDescriptors

from deepchem.utils.molecule_feature_utils import one_hot_encode
from deepchem.utils.molecule_feature_utils import get_atom_total_degree_one_hot
from deepchem.utils.molecule_feature_utils import get_atom_formal_charge_one_hot
from deepchem.utils.molecule_feature_utils import get_atom_total_num_Hs_one_hot
from deepchem.utils.molecule_feature_utils import get_atom_hybridization_one_hot
from deepchem.utils.molecule_feature_utils import get_atom_is_in_aromatic_one_hot

from deepchem.feat.graph_features import bond_features as b_Feats

logger = logging.getLogger(__name__)


class GraphConvConstants(object):
  """
  A class for holding featurization parameters.
  """

  MAX_ATOMIC_NUM = 100
  ATOM_FEATURES: Dict[str, List[int]] = {
      'atomic_num': list(range(MAX_ATOMIC_NUM)),
      'degree': [0, 1, 2, 3, 4, 5],
      'formal_charge': [-1, -2, 1, 2, 0],
      'chiral_tag': [0, 1, 2, 3],
      'num_Hs': [0, 1, 2, 3, 4]
  }
  ATOM_FEATURES_HYBRIDIZATION: List[str] = ["SP", "SP2", "SP3", "SP3D", "SP3D2"]
  # Dimension of atom feature vector
  ATOM_FDIM = sum(len(choices) + 1 for choices in ATOM_FEATURES.values()) + len(
      ATOM_FEATURES_HYBRIDIZATION) + 1 + 2
  # len(choices) +1 and len(ATOM_FEATURES_HYBRIDIZATION) +1 to include room for unknown set
  # + 2 at end for is_in_aromatic and mass
  BOND_FDIM = 14

  # dictionary of available feature generators
  FEATURE_GENERATORS: Dict[str, MolecularFeaturizer] = {
      "morgan":
          CircularFingerprint(radius=2, size=2048, sparse=False),
      "morgan_count":
          CircularFingerprint(radius=2,
                              size=2048,
                              sparse=False,
                              is_counts_based=True),
      "rdkit_desc":
          RDKitDescriptors(use_bcut2d=False),
      "rdkit_desc_normalized":
          RDKitDescriptors(use_bcut2d=False, is_normalized=True)
  }


def get_atomic_num_one_hot(atom: RDKitAtom,
                           allowable_set: List[int],
                           include_unknown_set: bool = True) -> List[float]:
  """Get a one-hot feature about atomic number of the given atom.

  Parameters
  ---------
  atom: RDKitAtom
    RDKit atom object
  allowable_set: List[int]
    The range of atomic numbers to consider.
  include_unknown_set: bool, default False
    If true, the index of all types not in `allowable_set` is `len(allowable_set)`.

  Returns
  -------
  List[float]
    A one-hot vector of atomic number of the given atom.
    If `include_unknown_set` is False, the length is `len(allowable_set)`.
    If `include_unknown_set` is True, the length is `len(allowable_set) + 1`.
  """
  return one_hot_encode(atom.GetAtomicNum() - 1, allowable_set,
                        include_unknown_set)


def get_atom_chiral_tag_one_hot(
    atom: RDKitAtom,
    allowable_set: List[int],
    include_unknown_set: bool = True) -> List[float]:
  """Get a one-hot feature about chirality of the given atom.

  Parameters
  ---------
  atom: RDKitAtom
    RDKit atom object
  allowable_set: List[int]
    The list of chirality tags to consider.
  include_unknown_set: bool, default False
    If true, the index of all types not in `allowable_set` is `len(allowable_set)`.

  Returns
  -------
  List[float]
    A one-hot vector of chirality of the given atom.
    If `include_unknown_set` is False, the length is `len(allowable_set)`.
    If `include_unknown_set` is True, the length is `len(allowable_set) + 1`.
  """
  return one_hot_encode(atom.GetChiralTag(), allowable_set, include_unknown_set)


def get_atom_mass(atom: RDKitAtom) -> List[float]:
  """Get vector feature containing downscaled mass of the given atom.

  Parameters
  ---------
  atom: RDKitAtom
    RDKit atom object

  Returns
  -------
  List[float]
    A vector of downscaled mass of the given atom.
  """
  return [atom.GetMass() * 0.01]


def atom_features(
    atom: RDKitAtom,
    functional_groups: List[int] = None,
    only_atom_num: bool = False) -> Sequence[Union[bool, int, float]]:
  """Helper method used to compute atom feature vector.

  Deepchem already contains an atom_features function, however we are defining a new one here due to the need to handle features specific to DMPNN.

  Parameters
  ----------
  atom: RDKitAtom
    Atom to compute features on.
  functional_groups: List[int]
    A k-hot vector indicating the functional groups the atom belongs to.
    Default value is None
  only_atom_num: bool
    Toggle to build a feature vector for an atom containing only the atom number information.

  Returns
  -------
  features: Sequence[Union[bool, int, float]]
    A list of atom features.

  Examples
  --------
  >>> import deepchem as dc
  >>> from rdkit import Chem
  >>> mol = Chem.MolFromSmiles('C')
  >>> atom = mol.GetAtoms()[0]
  >>> features = dc.feat.molecule_featurizers.dmpnn_featurizer.atom_features(atom)
  >>> type(features)
  <class 'list'>
  >>> len(features)
  133
  """

  if atom is None:
    features: Sequence[Union[bool, int,
                             float]] = [0] * GraphConvConstants.ATOM_FDIM

  elif only_atom_num:
    features = []
    features += get_atomic_num_one_hot(
        atom, GraphConvConstants.ATOM_FEATURES['atomic_num'])
    features += [0] * (
        GraphConvConstants.ATOM_FDIM - GraphConvConstants.MAX_ATOMIC_NUM - 1
    )  # set other features to zero

  else:
    features = []
    features += get_atomic_num_one_hot(
        atom, GraphConvConstants.ATOM_FEATURES['atomic_num'])
    features += get_atom_total_degree_one_hot(
        atom, GraphConvConstants.ATOM_FEATURES['degree'])
    features += get_atom_formal_charge_one_hot(
        atom, GraphConvConstants.ATOM_FEATURES['formal_charge'])
    features += get_atom_chiral_tag_one_hot(
        atom, GraphConvConstants.ATOM_FEATURES['chiral_tag'])
    features += get_atom_total_num_Hs_one_hot(
        atom, GraphConvConstants.ATOM_FEATURES['num_Hs'])
    features += get_atom_hybridization_one_hot(
        atom, GraphConvConstants.ATOM_FEATURES_HYBRIDIZATION, True)
    features += get_atom_is_in_aromatic_one_hot(atom)
    features = [int(feature) for feature in features]
    features += get_atom_mass(atom)

    if functional_groups is not None:
      features += functional_groups
  return features


def bond_features(bond: RDKitBond) -> Sequence[Union[bool, int, float]]:
  """wrapper function for bond_features() already available in deepchem, used to compute bond feature vector.

  Parameters
  ----------
  bond: RDKitBond
    Bond to compute features on.

  Returns
  -------
  features: Sequence[Union[bool, int, float]]
    A list of bond features.

  Examples
  --------
  >>> import deepchem as dc
  >>> from rdkit import Chem
  >>> mol = Chem.MolFromSmiles('CC')
  >>> bond = mol.GetBondWithIdx(0)
  >>> b_features = dc.feat.molecule_featurizers.dmpnn_featurizer.bond_features(bond)
  >>> type(b_features)
  <class 'list'>
  >>> len(b_features)
  14
  """
  if bond is None:
    b_features: Sequence[Union[
        bool, int, float]] = [1] + [0] * (GraphConvConstants.BOND_FDIM - 1)

  else:
    b_features = [0] + b_Feats(bond, use_extended_chirality=True)
  return b_features


def map_reac_to_prod(
    mol_reac: RDKitMol,
    mol_prod: RDKitMol) -> Tuple[Dict[int, int], List[int], List[int]]:
  """
  Function to build a dictionary of mapping atom indices in the reactants to the products.

  Parameters
  ----------
  mol_reac: RDKitMol
  An RDKit molecule of the reactants.

  mol_prod: RDKitMol
  An RDKit molecule of the products.

  Returns
  -------
  mappings: Tuple[Dict[int,int],List[int],List[int]]
  A tuple containing a dictionary of corresponding reactant and product atom indices,
  list of atom ids of product not part of the mapping and
  list of atom ids of reactant not part of the mapping
  """
  only_prod_ids: List[int] = []
  prod_map_to_id: Dict[int, int] = {}
  mapnos_reac: Set[int] = set(
      [atom.GetAtomMapNum() for atom in mol_reac.GetAtoms()])
  for atom in mol_prod.GetAtoms():
    mapno = atom.GetAtomMapNum()
    if (mapno > 0):
      prod_map_to_id[mapno] = atom.GetIdx()
      if (mapno not in mapnos_reac):
        only_prod_ids.append(atom.GetIdx())
    else:
      only_prod_ids.append(atom.GetIdx())
  only_reac_ids: List[int] = []
  reac_id_to_prod_id: Dict[int, int] = {}
  for atom in mol_reac.GetAtoms():
    mapno = atom.GetAtomMapNum()
    if (mapno > 0):
      try:
        reac_id_to_prod_id[atom.GetIdx()] = prod_map_to_id[mapno]
      except KeyError:
        only_reac_ids.append(atom.GetIdx())
    else:
      only_reac_ids.append(atom.GetIdx())
  mappings: Tuple[Dict[int, int], List[int],
                  List[int]] = (reac_id_to_prod_id, only_prod_ids,
                                only_reac_ids)
  return mappings


class _MapperDMPNN:
  """
  This class is a helper class for DMPNN featurizer to generate concatenated feature vector and mapping.

  `self.f_ini_atoms_bonds_zero_padded` is the concatenated feature vector which contains
  concatenation of initial atom and bond features.

  `self.mapping` is the mapping which maps bond index to 'array of indices of the bonds'
  incoming at the initial atom of the bond (excluding the reverse bonds)
  """

  def __init__(self, datapoint: RDKitMol, concat_fdim: int,
               f_atoms_zero_padded: np.ndarray):
    """
    Parameters
    ----------
    datapoint: RDKitMol
      RDKit mol object.
    concat_fdim: int
      dimension of feature vector with concatenated atom (initial) and bond features
    f_atoms_zero_padded: np.ndarray
      mapping from atom index to atom features | initial input is a zero padding
    """
    self.datapoint = datapoint
    self.concat_fdim = concat_fdim
    self.f_atoms_zero_padded = f_atoms_zero_padded

    # number of atoms
    self.num_atoms: int = len(f_atoms_zero_padded) - 1

    # number of bonds
    self.num_bonds: int = 0

    # mapping from bond index to concat(in_atom, bond) features | initial input is a zero padding
    self.f_ini_atoms_bonds_zero_padded: np.ndarray = np.asarray(
        [[0] * (self.concat_fdim)], dtype=float)

    # mapping from atom index to list of indices of incoming bonds
    self.atom_to_incoming_bonds: List[List[int]] = [
        [] for i in range(self.num_atoms + 1)
    ]

    # mapping from bond index to the index of the atom the bond is coming from
    self.bond_to_ini_atom: List[int] = [0]

    # mapping from bond index to the index of the reverse bond
    self.b2revb: List[int] = [0]

    self.mapping: np.ndarray = np.empty(0)

    self._generate_mapping()

  def _generate_mapping(self):
    """
    Generate mapping which maps bond index to 'array of indices of the bonds'
    incoming at the initial atom of the bond (reverse bonds are not considered).

    Steps:
    - Iterate such that all bonds in the mol are considered.
      For each iteration: (if bond exists)
      - Update the `self.f_ini_atoms_bonds_zero_padded` concatenated feature vector.
      - Update secondary mappings.
    - Modify `self.atom_to_incoming_bonds` based on maximum number of bonds.
    - Get mapping based on `self.atom_to_incoming_bonds` and `self.bond_to_ini_atom`.
    - Replace reverse bond values with 0
    """
    for a1 in range(1, self.num_atoms + 1):
      for a2 in range(a1 + 1, self.num_atoms + 1):
        if not self._update_concat_vector(a1, a2):
          continue
        self._update_secondary_mappings(a1, a2)
        self.num_bonds += 2
    self._modify_based_on_max_bonds()

    # get mapping which maps bond index to 'array of indices of the bonds' incoming at the initial atom of the bond
    self.mapping = np.asarray(
        self.atom_to_incoming_bonds)[self.bond_to_ini_atom]

    self._replace_rev_bonds()

  def _extend_concat_feature(self, a1: int, bond_feature: np.ndarray):
    """
    Helper method to concatenate initial atom and bond features and append them to `self.f_ini_atoms_bonds_zero_padded`.

    Parameters
    ----------
    a1: int
      index of the atom where the bond starts
    bond_feature: np.ndarray
      array of bond features
    """
    concat_input: np.ndarray = np.concatenate(
        (self.f_atoms_zero_padded[a1], bond_feature),
        axis=0).reshape([1, self.concat_fdim])
    self.f_ini_atoms_bonds_zero_padded = np.concatenate(
        (self.f_ini_atoms_bonds_zero_padded, concat_input), axis=0)

  def _update_concat_vector(self, a1: int, a2: int):
    """
    Method to update `self.f_ini_atoms_bonds_zero_padded` with features of the bond between atoms `a1` and `a2`.

    Parameters
    ----------
    a1: int
      index of the atom 1
    a2: int
      index of the atom 2
    """
    bond: RDKitBond = self.datapoint.GetBondBetweenAtoms(a1 - 1, a2 - 1)
    if bond is None:
      return 0

    # get bond features
    f_bond: np.ndarray = np.asarray(bond_features(bond), dtype=float)

    self._extend_concat_feature(a1, f_bond)
    self._extend_concat_feature(a2, f_bond)
    return 1

  def _update_secondary_mappings(self, a1, a2):
    """
    Method to update `self.atom_to_incoming_bonds`, `self.bond_to_ini_atom` and `self.b2revb`
    with respect to the bond between atoms `a1` and `a2`.

    Parameters
    ----------
    a1: int
      index of the atom 1
    a2: int
      index of the atom 2
    """
    b1: int = self.num_bonds + 1  # bond index
    b2: int = self.num_bonds + 2  # reverse bond index

    self.atom_to_incoming_bonds[a2].append(b1)  # b1 = a1 --> 'a2'
    self.atom_to_incoming_bonds[a1].append(b2)  # b2 = a2 --> 'a1'

    self.bond_to_ini_atom.append(a1)  # b1 starts at a1
    self.bond_to_ini_atom.append(a2)  # b2 starts at a2 (remember, b2 =  b1+1)

    self.b2revb.append(b2)  # reverse bond of b1 is b2
    self.b2revb.append(b1)  # reverse bond of b2 is b1

  def _modify_based_on_max_bonds(self):
    """
    Method to make number of incoming bonds equal to maximum number of bonds.
    This is done by appending zeros to fill remaining space at each atom indices.
    """
    max_num_bonds: int = max(
        1,
        max(
            len(incoming_bonds)
            for incoming_bonds in self.atom_to_incoming_bonds))
    self.atom_to_incoming_bonds = [
        self.atom_to_incoming_bonds[a] + [0] *
        (max_num_bonds - len(self.atom_to_incoming_bonds[a]))
        for a in range(self.num_atoms + 1)
    ]

  def _replace_rev_bonds(self):
    """
    Method to replace the reverse bond indices with zeros.
    """
    for count, i in enumerate(self.b2revb):
      self.mapping[count][np.where(self.mapping[count] == i)] = 0


def generate_global_features(mol: RDKitMol,
                             features_generators: List[str]) -> np.ndarray:
  """
  Helper function for generating global features for a RDKit mol based on the given list of feature generators to be used.

  Parameters
  ----------
  mol: RDKitMol
    RDKit molecule to be featurized
  features_generators: List[str]
    List of names of the feature generators to be used featurization

  Returns
  -------
  global_features_array: np.ndarray
    Array of global features

  Examples
  --------
  >>> from rdkit import Chem
  >>> import deepchem as dc
  >>> mol = Chem.MolFromSmiles('C')
  >>> features_generators = ['morgan']
  >>> global_features = dc.feat.molecule_featurizers.dmpnn_featurizer.generate_global_features(mol, features_generators)
  >>> type(global_features)
  <class 'numpy.ndarray'>
  >>> len(global_features)
  2048
  >>> nonzero_features_indices = global_features.nonzero()[0]
  >>> nonzero_features_indices
  array([1264])
  >>> global_features[nonzero_features_indices[0]]
  1.0
  """
  global_features: List[np.ndarray] = []
  available_generators = GraphConvConstants.FEATURE_GENERATORS

  for generator in features_generators:
    if generator in available_generators:
      global_featurizer = available_generators[generator]
      if mol.GetNumHeavyAtoms() > 0:
        global_features.extend(global_featurizer.featurize(mol)[0])
      # for H2
      elif mol.GetNumHeavyAtoms() == 0:
        # not all features are equally long, so used methane as dummy molecule to determine length
        global_features.extend(
            np.zeros(
                len(global_featurizer.featurize(Chem.MolFromSmiles('C'))[0])))
    else:
      logger.warning(f"{generator} generator is not available in DMPNN")

  global_features_array: np.ndarray = np.asarray(global_features)

  # Fix nans in features
  replace_token = 0
  global_features_array = np.where(np.isnan(global_features_array),
                                   replace_token, global_features_array)

  return global_features_array


class DMPNNFeaturizer(MolecularFeaturizer):
  """
  This class is a featurizer for Directed Message Passing Neural Network (D-MPNN) implementation

  The default node(atom) and edge(bond) representations are based on
  `Analyzing Learned Molecular Representations for Property Prediction paper <https://arxiv.org/pdf/1904.01561.pdf>`_.

  The default node representation are constructed by concatenating the following values,
  and the feature length is 133.

  - Atomic num: A one-hot vector of this atom, in a range of first 100 atoms.
  - Degree: A one-hot vector of the degree (0-5) of this atom.
  - Formal charge: Integer electronic charge, -1, -2, 1, 2, 0.
  - Chirality: A one-hot vector of the chirality tag (0-3) of this atom.
  - Number of Hydrogens: A one-hot vector of the number of hydrogens (0-4) that this atom connected.
  - Hybridization: A one-hot vector of "SP", "SP2", "SP3", "SP3D", "SP3D2".
  - Aromatic: A one-hot vector of whether the atom belongs to an aromatic ring.
  - Mass: Atomic mass * 0.01

  The default edge representation are constructed by concatenating the following values,
  and the feature length is 14.

  - Bond type: A one-hot vector of the bond type, "single", "double", "triple", or "aromatic".
  - Same ring: A one-hot vector of whether the atoms in the pair are in the same ring.
  - Conjugated: A one-hot vector of whether this bond is conjugated or not.
  - Stereo: A one-hot vector of the stereo configuration (0-5) of a bond.

  If you want to know more details about features, please check the paper [1]_ and
  utilities in deepchem.utils.molecule_feature_utils.py.

  Examples
  --------
  >>> smiles = ["C1=CC=CN=C1", "C1CCC1"]
  >>> featurizer = DMPNNFeaturizer()
  >>> out = featurizer.featurize(smiles)
  >>> type(out[0])
  <class 'deepchem.feat.graph_data.GraphData'>
  >>> out[0].num_nodes
  6
  >>> out[0].num_node_features
  133
  >>> out[0].node_features.shape
  (6, 133)
  >>> out[0].num_edge_features
  14
  >>> out[0].num_edges
  12
  >>> out[0].edge_features.shape
  (12, 14)

  References
  ----------
  .. [1] Kearnes, Steven, et al. "Molecular graph convolutions: moving beyond fingerprints."
     Journal of computer-aided molecular design 30.8 (2016):595-608.

  Note
  ----
  This class requires RDKit to be installed.
  """

  def __init__(self,
               features_generators: Optional[List[str]] = None,
               is_adding_hs: bool = False,
               use_original_atom_ranks: bool = False):
    """
    Parameters
    ----------
    features_generator: List[str], default None
      List of global feature generators to be used.
    is_adding_hs: bool, default False
      Whether to add Hs or not.
    use_original_atom_ranks: bool, default False
      Whether to use original atom mapping or canonical atom mapping
    """
    self.features_generators = features_generators
    self.is_adding_hs = is_adding_hs
    super().__init__(use_original_atom_ranks)

  def _construct_bond_index(self, datapoint: RDKitMol) -> np.ndarray:
    """
    Construct edge (bond) index

    Parameters
    ----------
    datapoint: RDKitMol
      RDKit mol object.

    Returns
    -------
    edge_index: np.ndarray
      Edge (Bond) index
    """
    src: List[int] = []
    dest: List[int] = []
    for bond in datapoint.GetBonds():
      # add edge list considering a directed graph
      start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
      src += [start, end]
      dest += [end, start]
    return np.asarray([src, dest], dtype=int)

  def _get_bond_features(self, datapoint: RDKitMol) -> np.ndarray:
    """
    Construct bond(edge) features for the given datapoint

    For each bond index, 2 bond feature arrays are added to the main features array,
    for the current bond and its reverse bond respectively.

    Note: This method of generating bond features ensures that the shape of the bond features array
          is always equal to (number of bonds, number of bond features), even if the number of bonds
          is equal to 0.

    Parameters
    ----------
    datapoint: RDKitMol
      RDKit mol object.

    Returns
    -------
    f_bonds: np.ndarray
      Bond features array
    """
    bonds: Chem.rdchem._ROBondSeq = datapoint.GetBonds()

    bond_fdim: int = GraphConvConstants.BOND_FDIM
    number_of_bonds: int = len(
        bonds) * 2  # Note the value is doubled to account for reverse bonds
    f_bonds: np.ndarray = np.empty((number_of_bonds, bond_fdim))

    for index in range(0, number_of_bonds, 2):
      bond_id: int = index // 2
      bond_feature: np.ndarray = np.asarray(bond_features(bonds[bond_id]),
                                            dtype=float)
      f_bonds[index] = bond_feature  # bond
      f_bonds[index + 1] = bond_feature  # reverse bond
    return f_bonds

  def _featurize(self, datapoint: RDKitMol, **kwargs) -> GraphData:
    """
    Calculate molecule graph features from RDKit mol object.

    Parameters
    ----------
    datapoint: RDKitMol
      RDKit mol object.

    Returns
    -------
    graph: GraphData
      A molecule graph object with features:
      - node_features: Node feature matrix with shape [num_nodes, num_node_features]
      - edge_index: Graph connectivity in COO format with shape [2, num_edges]
      - edge_features: Edge feature matrix with shape [num_edges, num_edge_features]
      - global_features: Array of global molecular features
    """
    if isinstance(datapoint, Chem.rdchem.Mol):
      if self.is_adding_hs:
        datapoint = Chem.AddHs(datapoint)
    else:
      raise ValueError(
          "Feature field should contain smiles for DMPNN featurizer!")

    # get atom features
    f_atoms: np.ndarray = np.asarray(
        [atom_features(atom) for atom in datapoint.GetAtoms()], dtype=float)

    # get edge(bond) features
    f_bonds: np.ndarray = self._get_bond_features(datapoint)

    # get edge index
    edge_index: np.ndarray = self._construct_bond_index(datapoint)

    # get global features
    global_features: np.ndarray = np.empty(0)
    if self.features_generators is not None:
      global_features = generate_global_features(datapoint,
                                                 self.features_generators)

    return GraphData(node_features=f_atoms,
                     edge_index=edge_index,
                     edge_features=f_bonds,
                     global_features=global_features)
