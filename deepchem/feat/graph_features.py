from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import numpy as np
from rdkit import Chem
import itertools, operator

from deepchem.feat import Featurizer
from deepchem.feat.mol_graphs import ConvMol, WeaveMol


def one_of_k_encoding(x, allowable_set):
  if x not in allowable_set:
    raise Exception(
        "input {0} not in allowable set{1}:".format(x, allowable_set))
  return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
  """Maps inputs not in the allowable set to the last element."""
  if x not in allowable_set:
    x = allowable_set[-1]
  return list(map(lambda s: x == s, allowable_set))


def get_intervals(l):
  """For list of lists, gets the cumulative products of the lengths"""
  intervals = len(l) * [0]
  # Initalize with 1
  intervals[0] = 1
  for k in range(1, len(l)):
    intervals[k] = (len(l[k]) + 1) * intervals[k - 1]

  return intervals


def safe_index(l, e):
  """Gets the index of e in l, providing an index of len(l) if not found"""
  try:
    return l.index(e)
  except:
    return len(l)


possible_atom_list = [
    'C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Mg', 'Na', 'Br', 'Fe', 'Ca', 'Cu',
    'Mc', 'Pd', 'Pb', 'K', 'I', 'Al', 'Ni', 'Mn'
]
possible_numH_list = [0, 1, 2, 3, 4]
possible_valence_list = [0, 1, 2, 3, 4, 5, 6]
possible_formal_charge_list = [-3, -2, -1, 0, 1, 2, 3]
possible_hybridization_list = [
    Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2
]
possible_number_radical_e_list = [0, 1, 2]

reference_lists = [
    possible_atom_list, possible_numH_list, possible_valence_list,
    possible_formal_charge_list, possible_number_radical_e_list,
    possible_hybridization_list
]

intervals = get_intervals(reference_lists)


def get_feature_list(atom):
  features = 6 * [0]
  features[0] = safe_index(possible_atom_list, atom.GetSymbol())
  features[1] = safe_index(possible_numH_list, atom.GetTotalNumHs())
  features[2] = safe_index(possible_valence_list, atom.GetImplicitValence())
  features[3] = safe_index(possible_formal_charge_list, atom.GetFormalCharge())
  features[4] = safe_index(possible_number_radical_e_list,
                           atom.GetNumRadicalElectrons())
  features[5] = safe_index(possible_hybridization_list, atom.GetHybridization())

  return features


def features_to_id(features, intervals):
  """Convert list of features into index using spacings provided in intervals"""
  id = 0
  for k in range(len(intervals)):
    id += features[k] * intervals[k]

  # Allow 0 index to correspond to null molecule 1
  id = id + 1
  return id


def id_to_features(id, intervals):
  features = 6 * [0]

  # Correct for null
  id -= 1

  for k in range(0, 6 - 1):
    #print(6-k-1, id)
    features[6 - k - 1] = id // intervals[6 - k - 1]
    id -= features[6 - k - 1] * intervals[6 - k - 1]
  # Correct for last one
  features[0] = id
  return features


def atom_to_id(atom):
  """Return a unique id corresponding to the atom type"""
  features = get_feature_list(atom)
  return features_to_id(features, intervals)


def atom_features(atom, bool_id_feat=False, explicit_H=False):
  if bool_id_feat:
    return np.array([atom_to_id(atom)])
  else:
    results = one_of_k_encoding_unk(
        atom.GetSymbol(),
        [
            'C',
            'N',
            'O',
            'S',
            'F',
            'Si',
            'P',
            'Cl',
            'Br',
            'Mg',
            'Na',
            'Ca',
            'Fe',
            'As',
            'Al',
            'I',
            'B',
            'V',
            'K',
            'Tl',
            'Yb',
            'Sb',
            'Sn',
            'Ag',
            'Pd',
            'Co',
            'Se',
            'Ti',
            'Zn',
            'H',  # H?
            'Li',
            'Ge',
            'Cu',
            'Au',
            'Ni',
            'Cd',
            'In',
            'Mn',
            'Zr',
            'Cr',
            'Pt',
            'Hg',
            'Pb',
            'Unknown'
        ]) + one_of_k_encoding(atom.GetDegree(),
                               [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + \
        one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) + \
        [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
        one_of_k_encoding_unk(atom.GetHybridization(), [
            Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
            SP3D, Chem.rdchem.HybridizationType.SP3D2
        ]) + [atom.GetIsAromatic()]
    # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
    if not explicit_H:
      results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                [0, 1, 2, 3, 4])

    return np.array(results)


def bond_features(bond):
  bt = bond.GetBondType()
  return np.array([
      bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
      bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
      bond.GetIsConjugated(),
      bond.IsInRing()
  ])


def pair_features(mol, edge_list, canon_adj_list, bt_len=6,
                  graph_distance=True):
  if graph_distance:
    max_distance = 7
  else:
    max_distance = 1
  N = mol.GetNumAtoms()
  features = np.zeros((N, N, bt_len + max_distance + 1))
  num_atoms = mol.GetNumAtoms()
  rings = mol.GetRingInfo().AtomRings()
  for a1 in range(num_atoms):
    for a2 in canon_adj_list[a1]:
      # first `bt_len` features are bond features(if applicable)
      features[a1, a2, :bt_len] = np.asarray(
          edge_list[tuple(sorted((a1, a2)))], dtype=float)
    for ring in rings:
      if a1 in ring:
        # `bt_len`-th feature is if the pair of atoms are in the same ring
        features[a1, ring, bt_len] = 1
        features[a1, a1, bt_len] = 0.
    # graph distance between two atoms
    if graph_distance:
      distance = find_distance(
          a1, num_atoms, canon_adj_list, max_distance=max_distance)
      features[a1, :, bt_len + 1:] = distance
  # Euclidean distance between atoms
  if not graph_distance:
    coords = np.zeros((N, 3))
    for atom in range(N):
      pos = mol.GetConformer(0).GetAtomPosition(atom)
      coords[atom, :] = [pos.x, pos.y, pos.z]
    features[:,:,-1] = np.sqrt(np.sum(np.square(
        np.stack([coords] * N, axis=1) - \
        np.stack([coords] * N, axis=0)), axis=2))

  return features


def find_distance(a1, num_atoms, canon_adj_list, max_distance=7):
  distance = np.zeros((num_atoms, max_distance))
  radial = 0
  # atoms `radial` bonds away from `a1`
  adj_list = set(canon_adj_list[a1])
  # atoms less than `radial` bonds away
  all_list = set([a1])
  while radial < max_distance:
    distance[list(adj_list), radial] = 1
    all_list.update(adj_list)
    # find atoms `radial`+1 bonds away
    next_adj = set()
    for adj in adj_list:
      next_adj.update(canon_adj_list[adj])
    adj_list = next_adj - all_list
    radial = radial + 1
  return distance


class ConvMolFeaturizer(Featurizer):

  name = ['conv_mol']

  def __init__(self):
    # Since ConvMol is an object and not a numpy array, need to set dtype to
    # object.
    self.dtype = object

  def _featurize(self, mol):
    """Encodes mol as a ConvMol object."""
    # Get the node features
    idx_nodes = [(a.GetIdx(), atom_features(a)) for a in mol.GetAtoms()]
    idx_nodes.sort()  # Sort by ind to ensure same order as rd_kit
    idx, nodes = list(zip(*idx_nodes))

    # Stack nodes into an array
    nodes = np.vstack(nodes)

    # Get bond lists with reverse edges included
    edge_list = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx())
                 for b in mol.GetBonds()]

    # Get canonical adjacency list
    canon_adj_list = [[] for mol_id in range(len(nodes))]
    for edge in edge_list:
      canon_adj_list[edge[0]].append(edge[1])
      canon_adj_list[edge[1]].append(edge[0])

    return ConvMol(nodes, canon_adj_list)


class WeaveFeaturizer(Featurizer):

  name = ['weave_mol']

  def __init__(self, graph_distance=True, explicit_H=False):
    # Distance is either graph distance(True) or Euclidean distance(False,
    # only support datasets providing Cartesian coordinates)
    self.graph_distance = graph_distance
    # Set dtype
    self.dtype = object
    # If includes explicit hydrogens
    self.explicit_H = explicit_H

  def _featurize(self, mol):
    """Encodes mol as a WeaveMol object."""
    # Atom features
    idx_nodes = [(a.GetIdx(), atom_features(a, explicit_H=self.explicit_H))
                 for a in mol.GetAtoms()]
    idx_nodes.sort()  # Sort by ind to ensure same order as rd_kit
    idx, nodes = list(zip(*idx_nodes))

    # Stack nodes into an array
    nodes = np.vstack(nodes)

    # Get bond lists
    edge_list = {}
    for b in mol.GetBonds():
      edge_list[tuple(sorted([b.GetBeginAtomIdx(),
                              b.GetEndAtomIdx()]))] = bond_features(b)

    # Get canonical adjacency list
    canon_adj_list = [[] for mol_id in range(len(nodes))]
    for edge in edge_list.keys():
      canon_adj_list[edge[0]].append(edge[1])
      canon_adj_list[edge[1]].append(edge[0])

    # Calculate pair features
    pairs = pair_features(
        mol,
        edge_list,
        canon_adj_list,
        bt_len=6,
        graph_distance=self.graph_distance)

    return WeaveMol(nodes, pairs)
