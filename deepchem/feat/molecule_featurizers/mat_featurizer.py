from deepchem.feat.base_classes import MolecularFeaturizer
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

use_cuda = torch.cuda.is_available()

if use_cuda:
    FloatTensor = torch.cuda.FloatTensor
    LongTensor = torch.cuda.LongTensor
    IntTensor = torch.cuda.IntTensor
    DoubleTensor = torch.cuda.DoubleTensor
else:
    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor
    IntTensor = torch.cuda.IntTensor
    DoubleTensor = torch.cuda.DoubleTensor

class MATFeaturizer(MolecularFeaturizer):
    def __init__(
      self,
      self.mol = mol,
      self.add_dummy_node = add_dummy_node,
      self.one_hot_formal_charge = one_hot_formal_charge,
      ):
    
    def OHVector(value, array):
        if value not in array:
            value = array[-1]
        return map(lambda x : x == value, array)

    def atom_features(self, atom, one_hot_formal_charge = True):
        attrib = []
        attrib += OHVector(atom.getAtomicNum(), [5, 6, 7, 8, 9, 15, 16, 17, 35, 53, 999])
        attrib += OHVector(len(atom.GetNeighbors()), [0, 1, 2, 3, 4, 5])
        attrib += OHVector(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])

        if one_hot_formal_charge:
            attrib += OHVector(atom.GetFormalCharge(), [-1, 0, 1])
        else:
            attrib.append(atom.GetFormalCharge())
        
        attrib.append(atom.IsInRing())
        attrib.append(atom.GetIsAromatic())

        return np.array(attrib, dtype = np.float32)
    
    def _featurize(self, mol, add_dummy_node, one_hot_formal_charge):
        node_features = np.array([atom_features(atom, one_hot_formal_charge) for atom in mol.getAtoms()])
        adjacency_matrix = np.eye(mol.GetNumAtoms())
        for bond in mol.GetBonds():
            starting_atom = bond.GetBeginAtom().GetIdx()
            ending_atom = bond.getEndAtom().GetIdx()
            adjacency_matrix[starting_atom, ending_atom] = adjacency_matrix[ending_atom, starting_atom] = 1
        
        conformer = mol.GetConformer()
        positional_matrix = np.array([[conformer.GetAtomPosition(k).x, conformer.GetAtomPosition(k).y, conformer.GetAtomPosition(k).z] for k in range(mol.GetNumAtoms())])
        distance_matrix = pairwise_distances(positional_matrix)

        if add_dummy_node:
            m = np.zeros((node_features.shape[0] + 1, node_features.shape[1] + 1))
            m[1:, 1:] = node_features
            m[0, 0] = 1.0
            node_features = m

            m = np.zeros((adjacency_matrix.shape[0] + 1, adjacency_matrix.shape[1] + 1))
            m[1:, 1:] = adjacency_matrix
            adjacency_matrix = m

            m = np.full((distance_matrix.shape[0] + 1, distance_matrix.shape[1] + 1), 1e6)
            m[1:, 1:] = distance_matrix
            distance_matrix = m
        return node_features, adjacency_matrix, distance_matrix