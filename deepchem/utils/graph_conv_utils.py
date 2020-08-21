"""
Utilities for constructing node features or bond features.
Some functions are based on chainer-chemistry or dgl-lifesci.

Repositories:
- https://github.com/chainer/chainer-chemistry
- https://github.com/awslabs/dgl-lifesci
"""

import os
import logging
from typing import List, Union, Sequence, Tuple

import numpy as np

from deepchem.utils.typing import RDKitAtom, RDKitBond, RDKitMol

logger = logging.getLogger(__name__)

DEFAULT_ATOM_TYPE_SET = [
    "C",
    "N",
    "O",
    "F",
    "P",
    "S",
    "Br",
    "I",
]
DEFAULT_HYBRIDIZATION_SET = ["SP", "SP2", "SP3"]
DEFAULT_RING_SIZE_SET = [3, 4, 5, 6, 7, 8]
DEFAULT_BOND_TYPE_SET = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]
DEFAULT_GRAPH_DISTANCE_SET = [1, 2, 3, 4, 5, 6, 7]


class _ChemicalFeaturesFactory:
  """This is a singleton class for RDKit base features."""
  _instance = None

  @classmethod
  def get_instance(cls):
    try:
      from rdkit import RDConfig
      from rdkit.Chem import ChemicalFeatures
    except ModuleNotFoundError:
      raise ValueError("This class requires RDKit to be installed.")

    if not cls._instance:
      fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
      cls._instance = ChemicalFeatures.BuildFeatureFactory(fdefName)
    return cls._instance


def one_hot_encode(val: Union[int, str],
                   allowable_set: Union[List[str], List[int]],
                   include_unknown_set: bool = False) -> List[int]:
  """One hot encoder for elements of a provided set.

  Examples
  --------
  >>> one_hot_encode("a", ["a", "b", "c"])
  [1, 0, 0]
  >>> one_hot_encode(2, [0, 1, 2])
  [0, 0, 1]
  >>> one_hot_encode(3, [0, 1, 2])
  [0, 0, 0]
  >>> one_hot_encode(3, [0, 1, 2], True)
  [0, 0, 0, 1]

  Parameters
  ----------
  val: int or str
    The value must be present in `allowable_set`.
  allowable_set: List[int] or List[str]
    List of allowable quantities.
  include_unknown_set: bool, default False
    If true, the index of all values not in `allowable_set` is `len(allowable_set)`.

  Returns
  -------
  List[int]
    An one hot vector of val.
    If `include_unknown_set` is False, the length is `len(allowable_set)`.
    If `include_unknown_set` is True, the length is `len(allowable_set) + 1`.

  Raises
  ------
  `ValueError` if include_unknown_set is False and `val` is not in `allowable_set`.
  """
  if include_unknown_set is False:
    if val not in allowable_set:
      logger.warning("input {0} not in allowable set {1}:".format(
          val, allowable_set))

  if include_unknown_set is False:
    one_hot_legnth = len(allowable_set)
  else:
    one_hot_legnth = len(allowable_set) + 1
  one_hot = [0 for _ in range(one_hot_legnth)]

  try:
    one_hot[allowable_set.index(val)] = 1
  except:
    if include_unknown_set:
      # If include_unknown_set is True, set the last index is 1.
      one_hot[-1] = 1
    else:
      pass
  return one_hot


#################################################################
# atom (node) featurization
#################################################################


def get_atom_type_one_hot(atom: RDKitAtom,
                          allowable_set: List[str] = DEFAULT_ATOM_TYPE_SET,
                          include_unknown_set: bool = True) -> List[int]:
  """Get an one hot feature of an atom type.

  Paramters
  ---------
  atom: rdkit.Chem.rdchem.Atom
    RDKit atom object
  allowable_set: List[str]
    The atom types to consider. The default set is
    `["C", "N", "O", "F", "P", "S", "Br", "I"]`.
  include_unknown_set: bool, default True
    If true, the index of all atom not in `allowable_set` is `len(allowable_set)`.

  Returns
  -------
  List[int]
    An one hot vector of atom types.
    If `include_unknown_set` is False, the length is `len(allowable_set)`.
    If `include_unknown_set` is True, the length is `len(allowable_set) + 1`.
  """
  return one_hot_encode(atom.GetSymbol(), allowable_set, include_unknown_set)


def get_atomic_number(atom: RDKitAtom) -> List[int]:
  """Get an atomic number of an atom.

  Paramters
  ---------
  atom: rdkit.Chem.rdchem.Atom
    RDKit atom object

  Returns
  -------
  List[int]
    A vector of the atomic number.
  """
  return [atom.GetAtomicNum()]


def construct_hydrogen_bonding_info(mol: RDKitMol) -> List[Tuple[int, str]]:
  """Construct hydrogen bonding infos about a molecule.

  Paramters
  ---------
  mol: rdkit.Chem.rdchem.Mol
    RDKit mol object

  Returns
  -------
  List[Tuple[int, str]]
    A list of tuple `(atom_index, hydrogen_bonding_type)`.
    The `hydrogen_bonding_type` value is "Acceptor" or "Donor".
  """
  factory = _ChemicalFeaturesFactory.get_instance()
  feats = factory.GetFeaturesForMol(mol)
  hydrogen_bonding = []
  for f in feats:
    hydrogen_bonding.append((f.GetAtomIds()[0], f.GetFamily()))
  return hydrogen_bonding


def get_atom_hydrogen_bonding_one_hot(
    atom: RDKitAtom, hydrogen_bonding: List[Tuple[int, str]]) -> List[int]:
  """Get an one hot feat about whether an atom accepts electrons or donates electrons.

  Paramters
  ---------
  atom: rdkit.Chem.rdchem.Atom
    RDKit atom object
  hydrogen_bonding: List[Tuple[int, str]]
    The return value of `construct_hydrogen_bonding_info`.
    The value is a list of tuple `(atom_index, hydrogen_bonding)` like (1, "Acceptor").

  Returns
  -------
  List[int]
    A one hot vector of the ring size type. The first element
    indicates "Donor", and the second element indicates "Acceptor".
  """
  one_hot = [0, 0]
  atom_idx = atom.GetIdx
  for hydrogen_bonding_tuple in hydrogen_bonding:
    if hydrogen_bonding_tuple[0] == atom_idx:
      if hydrogen_bonding_tuple[1] == "Donor":
        one_hot[0] = 1
      elif hydrogen_bonding_tuple[1] == "Acceptor":
        one_hot[1] = 1
  return one_hot


def get_atom_is_in_aromatic_one_hot(atom: RDKitAtom) -> List[int]:
  """Get ans one hot feature about whether an atom is in aromatic system or not.

  Paramters
  ---------
  atom: rdkit.Chem.rdchem.Atom
    RDKit atom object

  Returns
  -------
  List[int]
    A vector of whether an atom is in aromatic system or not.
  """
  return [int(atom.GetIsAromatic())]


def get_atom_hybridization_one_hot(
    atom: RDKitAtom,
    allowable_set: List[str] = DEFAULT_HYBRIDIZATION_SET,
    include_unknown_set: bool = False) -> List[int]:
  """Get an one hot feature of hybridization type.

  Paramters
  ---------
  atom: rdkit.Chem.rdchem.Atom
    RDKit atom object
  allowable_set: List[str]
    The hybridization types to consider. The default set is `["SP1", "SP2", "SP3"]`
  include_unknown_set: bool, default False
    If true, the index of all types not in `allowable_set` is `len(allowable_set)`.

  Returns
  -------
  List[int]
    An one hot vector of the hybridization type.
    If `include_unknown_set` is False, the length is `len(allowable_set)`.
    If `include_unknown_set` is True, the length is `len(allowable_set) + 1`.
  """
  return one_hot_encode(
      str(atom.GetHybridization()), allowable_set, include_unknown_set)


def get_atom_total_num_Hs(atom: RDKitAtom) -> List[int]:
  """Get the number of hydrogen which an atom has.

  Paramters
  ---------
  atom: rdkit.Chem.rdchem.Atom
    RDKit atom object

  Returns
  -------
  List[int]
    A vector of the number of hydrogen which an atom has.
  """
  return [atom.GetTotalNumHs()]


def get_atom_chirality_one_hot(
    atom: RDKitAtom, chiral_center: List[Tuple[int, str]]) -> List[int]:
  """Get an one hot feature about an atom chirality type.

  Paramters
  ---------
  atom: rdkit.Chem.rdchem.Atom
    RDKit atom object
  chiral_center: List[Tuple[int, str]]
    The return value of `Chem.FindMolChiralCenters(mol)`.
    The value is a list of tuple `(atom_index, chirality)` like (1, 'S').

  Returns
  -------
  List[int]
    A one hot vector of the chirality type. The first element
    indicates "R", and the second element indicates "S".
  """
  one_hot = [0, 0]
  atom_idx = atom.GetIdx()
  for chiral_tuple in chiral_center:
    if chiral_tuple[0] == atom_idx:
      if chiral_tuple[1] == "R":
        one_hot[0] = 1
      elif chiral_tuple[1] == "S":
        one_hot[1] = 1
  return one_hot


def get_atom_formal_charge(atom: RDKitAtom) -> List[int]:
  """Get a formal charge of an atom.

  Paramters
  ---------
  atom: rdkit.Chem.rdchem.Atom
    RDKit atom object

  Returns
  -------
  List[int]
    A vector of the formal charge.
  """
  return [atom.GetFormalCharge()]


def get_atom_partial_charge(atom: RDKitAtom) -> List[float]:
  """Get a partial charge of an atom.

  Paramters
  ---------
  atom: rdkit.Chem.rdchem.Atom
    RDKit atom object

  Returns
  -------
  List[float]
    A vector of the parital charge.

  Notes
  -----
  Before using this function, you must calculate `GasteigerCharge`
  like `AllChem.ComputeGasteigerCharges(mol)`.
  """
  gasteiger_charge = atom.GetProp('_GasteigerCharge')
  if gasteiger_charge in ['-nan', 'nan', '-inf', 'inf']:
    gasteiger_charge = 0
  return [float(gasteiger_charge)]


def get_atom_ring_size_one_hot(atom: RDKitAtom,
                               sssr: Sequence,
                               allowable_set: List[int] = DEFAULT_RING_SIZE_SET,
                               include_unknown_set: bool = False) -> List[int]:
  """Get an one hot feature about the ring size if an atom is in a ring.

  Paramters
  ---------
  atom: rdkit.Chem.rdchem.Atom
    RDKit atom object
  sssr: Sequence
    The return value of `Chem.GetSymmSSSR(mol)`.
    The value is a sequence of rings.
  allowable_set: List[int]
    The ring size types to consider. The default set is `["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]`.
  include_unknown_set: bool, default False
    If true, the index of all types not in `allowable_set` is `len(allowable_set)`.

  Returns
  -------
  List[int]
    A one hot vector of the ring size type.
    If `include_unknown_set` is False, the length is `len(allowable_set)`.
    If `include_unknown_set` is True, the length is `len(allowable_set) + 1`.
  """
  one_hot = [0 for _ in range(len(allowable_set))]
  atom_index = atom.GetIdx()
  if atom.IsInRing():
    for ring in sssr:
      ring = list(ring)
      if atom_index in ring:
        ring_size = len(ring)
        try:
          one_hot[DEFAULT_RING_SIZE_SET.index(ring_size)] = 1
        except:
          pass
  return one_hot


#################################################################
# bond (edge) featurization
#################################################################


def get_bond_type_one_hot(bond: RDKitBond,
                          allowable_set: List[str] = DEFAULT_BOND_TYPE_SET,
                          include_unknown_set: bool = False) -> List[int]:
  """Get an one hot feature of bond type.

  Paramters
  ---------
  bond: rdkit.Chem.rdchem.Bond
    RDKit bond object
  allowable_set: List[str]
    The bond types to consider. The default set is `["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]`.
  include_unknown_set: bool, default False
    If true, the index of all types not in `allowable_set` is `len(allowable_set)`.

  Returns
  -------
  List[int]
    A one hot vector of the bond type.
    If `include_unknown_set` is False, the length is `len(allowable_set)`.
    If `include_unknown_set` is True, the length is `len(allowable_set) + 1`.
  """
  return one_hot_encode(
      str(bond.GetBondType()), allowable_set, include_unknown_set)


def get_bond_is_in_same_ring_one_hot(bond: RDKitBond) -> List[int]:
  """Get an one hot feature about whether atoms of a bond is in the same ring or not.

  Paramters
  ---------
  bond: rdkit.Chem.rdchem.Bond
    RDKit bond object

  Returns
  -------
  List[int]
    A one hot vector of whether a bond is in the same ring or not.
  """
  return [int(bond.IsInRing())]


def get_bond_graph_distance_one_hot(
    bond: RDKitBond,
    graph_dist_matrix: np.ndarray,
    allowable_set: List[int] = DEFAULT_GRAPH_DISTANCE_SET,
    include_unknown_set: bool = True) -> List[int]:
  """Get an one hot feature of graph distance.

  Paramters
  ---------
  bond: rdkit.Chem.rdchem.Bond
    RDKit bond object
  graph_dist_matrix: np.ndarray
    The return value of `Chem.GetDistanceMatrix(mol)`. The shape is `(num_atoms, num_atoms)`.
  allowable_set: List[str]
    The graph distance types to consider. The default set is `[1, 2, ..., 7]`.
  include_unknown_set: bool, default False
    If true, the index of all types not in `allowable_set` is `len(allowable_set)`.

  Returns
  -------
  List[int]
    A one hot vector of the graph distance.
    If `include_unknown_set` is False, the length is `len(allowable_set)`.
    If `include_unknown_set` is True, the length is `len(allowable_set) + 1`.
  """
  graph_dist = graph_dist_matrix[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
  return one_hot_encode(graph_dist, allowable_set, include_unknown_set)


def get_bond_euclidean_distance(
    bond: RDKitBond,
    euclidean_dist_matrix: np.ndarray) -> List[float]:
  """Get an one hot feature of euclidean distance.

  Paramters
  ---------
  bond: rdkit.Chem.rdchem.Bond
    RDKit bond object
  euclidean_dist_matrix: np.ndarray
    The return value of `Chem.GetDistanceMatrix(mol)`. The shape is `(num_atoms, num_atoms)`.

  Returns
  -------
  List[float]
    A vector of the euclidean distance.
  """
  euclidean_dist = euclidean_dist_matrix[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
  return [euclidean_dist]
