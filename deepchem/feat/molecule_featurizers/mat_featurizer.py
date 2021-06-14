from deepchem.feat.base_classes import MolecularFeaturizer
from deepchem.utils.molecule_feature_utils import one_hot_encode
try:
  import logging
  import numpy as np
  import pandas as pd
  import torch
  from rdkit import Chem
  from rdkit.Chem import AllChem
  from rdkit.Chem import MolFromSmiles
  from sklearn.metrics import pairwise_distances
  from torch.utils import datasets
except:
  raise ImportError("Required Modules not found.")


class MATFeaturizer(MolecularFeaturizer):

  def __init__(
      self,
      add_dummy_node=True,
      one_hot_formal_charge=True,
  ):
    self.mol = mol
    self.add_dummy_node = add_dummy_node
    self.one_hot_formal_charge = one_hot_formal_charge

  def atom_features(self, atom, one_hot_formal_charge=True):
    attrib = []
    attrib += one_hot_encode(atom.GetAtomicNumber(),
                             [5, 6, 7, 8, 9, 15, 16, 17, 35, 53, 999])
    attrib += one_hot_encode(len(atom.GetNeighbors()), [0, 1, 2, 3, 4, 5])
    attrib += one_hot_encode(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])

    if one_hot_formal_charge:
      attrib += one_hot_encode(atom.GetFormalCharge(), [-1, 0, 1])
    else:
      attrib.append(atom.GetFormalCharge())

    attrib.append(atom.IsInRing())
    attrib.append(atom.GetIsAromatic())

    return np.array(attrib, dtype=np.float32)

  def _featurize(self, mol):
    node_features = np.array([
        atom_features(atom, self.one_hot_formal_charge)
        for atom in mol.getAtoms()
    ])

    adjacency_matrix = Chem.rdmolops.getAdjacencyMatrix(mol)
    distance_matrix = Chem.rdmolops.GetDistanceMatrix(mol)

    if self.add_dummy_node:
      m = np.zeros((node_features.shape[0] + 1, node_features.shape[1] + 1))
      m[1:, 1:] = node_features
      m[0, 0] = 1.0
      node_features = m

      m = np.zeros((adjacency_matrix.shape[0] + 1,
                    adjacency_matrix.shape[1] + 1))
      m[1:, 1:] = adjacency_matrix
      adjacency_matrix = m

      m = np.full((distance_matrix.shape[0] + 1, distance_matrix.shape[1] + 1),
                  1e6)
      m[1:, 1:] = distance_matrix
      distance_matrix = m

    return node_features, adjacency_matrix, distance_matrix