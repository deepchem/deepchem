from typing import List, Sequence, Tuple
import numpy as np

from deepchem.utils.typing import RDKitAtom, RDKitBond, RDKitMol
from deepchem.feat.graph_data import GraphData
from deepchem.feat.base_classes import MolecularFeaturizer
from deepchem.utils.molecule_feature_utils import get_atom_type_one_hot, \
  construct_hydrogen_bonding_info, get_atom_hydrogen_bonding_one_hot, \
  get_atom_is_in_aromatic_one_hot, get_atom_hybridization_one_hot, \
  get_atom_total_num_Hs_one_hot, get_atom_chirality_one_hot, get_atom_formal_charge, \
  get_atom_partial_charge, get_atom_ring_size_one_hot, get_atom_total_degree_one_hot, \
  get_bond_type_one_hot, get_bond_is_in_same_ring_one_hot, get_bond_is_conjugated_one_hot, \
  get_bond_stereo_one_hot


def _construct_atom_feature(atom: RDKitAtom,
                            h_bond_infos: List[Tuple[int, str]],
                            sssr: List[Sequence]) -> List[float]:
  """Construct an atom feature from a RDKit atom object.

  Parameters
  ----------
  atom: rdkit.Chem.rdchem.Atom
    RDKit atom object
  h_bond_infos: List[Tuple[int, str]]
    A list of tuple `(atom_index, hydrogen_bonding_type)`.
    Basically, it is expected that this value is the return value of
    `construct_hydrogen_bonding_info`. The `hydrogen_bonding_type`
    value is "Acceptor" or "Donor".
  sssr: List[Sequence]
    The return value of `Chem.GetSymmSSSR(mol)`.
    The value is a sequence of rings.

  Returns
  -------
  List[float]
    A one-hot vector of the atom feature.
  """
  atom_type = get_atom_type_one_hot(atom)
  chirality = get_atom_chirality_one_hot(atom)
  formal_charge = get_atom_formal_charge(atom)
  partial_charge = get_atom_partial_charge(atom)
  ring_size = get_atom_ring_size_one_hot(atom, sssr)
  hybridization = get_atom_hybridization_one_hot(atom)
  acceptor_donor = get_atom_hydrogen_bonding_one_hot(atom, h_bond_infos)
  aromatic = get_atom_is_in_aromatic_one_hot(atom)
  degree = get_atom_total_degree_one_hot(atom)
  total_num = get_atom_total_num_Hs_one_hot(atom)
  return atom_type + chirality + formal_charge + partial_charge + \
    ring_size + hybridization + acceptor_donor + aromatic + degree + total_num


def _construct_bond_feature(bond: RDKitBond) -> List[float]:
  """Construct a bond feature from a RDKit bond object.

  Parameters
  ---------
  bond: rdkit.Chem.rdchem.Bond
    RDKit bond object

  Returns
  -------
  List[float]
    A one-hot vector of the bond feature.
  """
  bond_type = get_bond_type_one_hot(bond)
  same_ring = get_bond_is_in_same_ring_one_hot(bond)
  conjugated = get_bond_is_conjugated_one_hot(bond)
  stereo = get_bond_stereo_one_hot(bond)
  return bond_type + same_ring + conjugated + stereo


class MolGraphConvFeaturizer(MolecularFeaturizer):
  """This class is a featurizer of general graph convolution networks for molecules.

  The default node(atom) and edge(bond) representations are based on
  `WeaveNet paper <https://arxiv.org/abs/1603.00856>`_. If you want to use your own representations,
  you could use this class as a guide to define your original Featurizer. In many cases, it's enough
  to modify return values of `construct_atom_feature` or `construct_bond_feature`.

  The default node representation are constructed by concatenating the following values,
  and the feature length is 39.

  - Atom type: A one-hot vector of this atom, "C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "other atoms".
  - Chirality: A one-hot vector of the chirality, "R" or "S".
  - Formal charge: Integer electronic charge.
  - Partial charge: Calculated partial charge.
  - Ring sizes: A one-hot vector of the size (3-8) of rings that include this atom.
  - Hybridization: A one-hot vector of "sp", "sp2", "sp3".
  - Hydrogen bonding: A one-hot vector of whether this atom is a hydrogen bond donor or acceptor.
  - Aromatic: A one-hot vector of whether the atom belongs to an aromatic ring.
  - Degree: A one-hot vector of the degree (0-5) of this atom.
  - Number of Hydrogens: A one-hot vector of the number of hydrogens (0-4) that this atom connected.

  The default edge representation are constructed by concatenating the following values,
  and the feature length is 11.

  - Bond type: A one-hot vector of the bond type, "single", "double", "triple", or "aromatic".
  - Same ring: A one-hot vector of whether the atoms in the pair are in the same ring.
  - Conjugated: A one-hot vector of whether this bond is conjugated or not.
  - Stereo: A one-hot vector of the stereo configuration of a bond.

  If you want to know more details about features, please check the paper [1]_ and
  utilities in deepchem.utils.molecule_feature_utils.py.

  Examples
  --------
  >>> smiles = ["C1CCC1", "C1=CC=CN=C1"]
  >>> featurizer = MolGraphConvFeaturizer()
  >>> out = featurizer.featurize(smiles)
  >>> type(out[0])
  <class 'deepchem.feat.graph_data.GraphData'>
  >>> out[0].num_node_features
  39
  >>> out[0].num_edge_features
  11

  References
  ----------
  .. [1] Kearnes, Steven, et al. "Molecular graph convolutions: moving beyond fingerprints."
     Journal of computer-aided molecular design 30.8 (2016):595-608.

  Notes
  -----
  This class requires RDKit to be installed.
  """

  def __init__(self, add_self_edges: bool = False):
    """
    Parameters
    ----------
    add_self_edges: bool, default False
      Whether to add self-connected edges or not. If you want to use DGL,
      you sometimes need to add explict self-connected edges.
    """
    self.add_self_edges = add_self_edges

  def _featurize(self, mol: RDKitMol) -> GraphData:
    """Calculate molecule graph features from RDKit mol object.

    Parameters
    ----------
    mol: rdkit.Chem.rdchem.Mol
      RDKit mol object.

    Returns
    -------
    graph: GraphData
      A molecule graph with some features.
    """
    try:
      from rdkit import Chem
      from rdkit.Chem import AllChem
    except ModuleNotFoundError:
      raise ValueError("This method requires RDKit to be installed.")

    # construct atom and bond features
    try:
      mol.GetAtomWithIdx(0).GetProp('_GasteigerCharge')
    except:
      # If partial charges were not computed
      AllChem.ComputeGasteigerCharges(mol)

    h_bond_infos = construct_hydrogen_bonding_info(mol)
    sssr = Chem.GetSymmSSSR(mol)

    # construct atom (node) feature
    atom_features = np.array(
        [
            _construct_atom_feature(atom, h_bond_infos, sssr)
            for atom in mol.GetAtoms()
        ],
        dtype=np.float,
    )

    # construct edge (bond) information
    src, dest, bond_features = [], [], []
    for bond in mol.GetBonds():
      # add edge list considering a directed graph
      start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
      src += [start, end]
      dest += [end, start]
      bond_features += 2 * [_construct_bond_feature(bond)]

    if self.add_self_edges:
      num_atoms = mol.GetNumAtoms()
      src += [i for i in range(num_atoms)]
      dest += [i for i in range(num_atoms)]
      # add dummy edge features
      bond_fea_length = len(bond_features[0])
      bond_features += num_atoms * [[0 for _ in range(bond_fea_length)]]

    return GraphData(
        node_features=atom_features,
        edge_index=np.array([src, dest], dtype=np.int),
        edge_features=np.array(bond_features, dtype=np.float))
