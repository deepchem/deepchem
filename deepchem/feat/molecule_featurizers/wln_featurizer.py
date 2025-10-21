import numpy as np
from typing import List, Optional, Tuple, Any
from deepchem.feat.base_classes import MolecularFeaturizer
from rdkit import Chem
from rdkit.Chem.rdchem import Atom, Bond


class WLNGraph:
    """
    Class to store data for WLN neural networks.

    Parameters
    ----------
    atom_features: np.ndarray
        Atom feature matrix with shape [num_atoms, num_atom_features]
    adjacency_matrix: np.ndarray
        Adjacency matrix with shape [num_atoms, num_atoms]
    bond_features: np.ndarray
        Bond feature matrix with shape [num_atoms, num_atoms, num_bond_features]
    atom_mask: np.ndarray
        Atom mask with shape [num_atoms], indicating valid atoms

    Returns
    -------
    graph: WLNGraph
        A molecule graph with WLN features.
    """

    def __init__(self, atom_features: np.ndarray, adjacency_matrix: np.ndarray,
                 bond_features: np.ndarray, atom_mask: np.ndarray):
        self.atom_features = atom_features
        self.adjacency_matrix = adjacency_matrix
        self.bond_features = bond_features
        self.atom_mask = atom_mask
        
        
class WLNFeaturizer(MolecularFeaturizer):
    
    def __init__(self) -> None:
        super(WLNFeaturizer, self).__init__()
        self.max_atoms = 50
        self.elem_list = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'W', 'Ru', 'Nb', 'Re', 'Te', 'Rh', 'Tc', 'Ba', 'Bi', 'Hf', 'Mo', 'U', 'Sm', 'Os', 'Ir', 'Ce','Gd','Ga','Cs', 'unknown']
        self.atom_degree_list = [0, 1, 2, 3, 4, 5]
        self.atom_explicit_valence_list = [1, 2, 3, 4, 5, 6]
        self.atom_implicit_valence_list = [0, 1, 2, 3, 4, 5]
        self.bond_fdim = 6

    def _onek_encoding_unk(self, x: Any, allowable_set: List[Any]) -> np.ndarray:
        """Helper function for one-hot encoding with unknown handling."""
        if x not in allowable_set:
            x = allowable_set[-1]
        return np.array([x == s for s in allowable_set], dtype=np.float32)

    def _get_atom_features(self, atom: Atom) -> np.ndarray:
        """
        Calculates features for a single atom and returns them as a numpy array.
        """
        return np.concatenate([
            self._onek_encoding_unk(atom.GetSymbol(), self.elem_list),
            self._onek_encoding_unk(atom.GetDegree(), self.atom_degree_list),
            self._onek_encoding_unk(atom.GetExplicitValence(), self.atom_explicit_valence_list),
            self._onek_encoding_unk(atom.GetImplicitValence(), self.atom_implicit_valence_list),
            np.array([float(atom.GetIsAromatic())])
        ])

    def _get_bond_features(self, bond: Bond) -> np.ndarray:
        """
        Calculates features for a single bond and returns them as a numpy array.
        """
        bt = bond.GetBondType()
        features = np.array([
            float(bt == Chem.rdchem.BondType.SINGLE),
            float(bt == Chem.rdchem.BondType.DOUBLE),
            float(bt == Chem.rdchem.BondType.TRIPLE),
            float(bt == Chem.rdchem.BondType.AROMATIC),
            float(bond.GetIsConjugated()),
            float(bond.IsInRing())
        ], dtype=np.float32)
        return features

    def _smiles_to_graph(self, smiles_string: str) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Converts a single SMILES string to its graph representation.
        This function handles SMILES with '.' to represent multiple disconnected molecules.
        """
        mol = Chem.MolFromSmiles(smiles_string)
        if mol is None:
            return None, None, None

        num_atoms = mol.GetNumAtoms()
        atom_features = np.stack([self._get_atom_features(atom) for atom in mol.GetAtoms()])
        
        adj_matrix = np.zeros((num_atoms, num_atoms), dtype=np.float32)
        bond_feature_matrix = np.zeros((num_atoms, num_atoms, self.bond_fdim), dtype=np.float32)

        # RDKit handles iterating over all bonds in all disconnected fragments
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            adj_matrix[i, j] = adj_matrix[j, i] = 1.0
            bond_feats = self._get_bond_features(bond)
            bond_feature_matrix[i, j] = bond_feature_matrix[j, i] = bond_feats

        return atom_features, adj_matrix, bond_feature_matrix

    def _featurize(self, smiles_list: List[str], **kwargs) -> WLNGraph:
        """
        Featurizes a list of SMILES strings into padded batch representations.
        Returns a dictionary with padded atom features, adjacency matrix, bond features, and atom mask.
        """
        graphs = [self._smiles_to_graph(s) for s in smiles_list]
        valid_graphs = [g for g in graphs if g[0] is not None]

        if not valid_graphs:
            # Handle all invalid case; return empty or raise error as per DeepChem convention
            n_mols = len(smiles_list)
            atom_fdim = len(self._get_atom_features(Chem.MolFromSmiles('C').GetAtomWithIdx(0)))
            padded_atom_features = np.zeros((n_mols, self.max_atoms, atom_fdim), dtype=np.float32)
            padded_adj = np.zeros((n_mols, self.max_atoms, self.max_atoms), dtype=np.float32)
            padded_bond_features = np.zeros((n_mols, self.max_atoms, self.max_atoms, self.bond_fdim), dtype=np.float32)
            atom_mask = np.zeros((n_mols, self.max_atoms), dtype=np.float32)
            return WLNGraph(padded_atom_features, padded_adj, padded_bond_features, atom_mask)
            
        atom_fdim = valid_graphs[0][0].shape[1]
        bond_fdim = valid_graphs[0][2].shape[2]
        
        max_atoms = max(g[0].shape[0] for g in valid_graphs)
        batch_size = len(valid_graphs)

        padded_atom_features = np.zeros((batch_size, max_atoms, atom_fdim), dtype=np.float32)
        padded_adj = np.zeros((batch_size, max_atoms, max_atoms), dtype=np.float32)
        padded_bond_features = np.zeros((batch_size, max_atoms, max_atoms, bond_fdim), dtype=np.float32)
        atom_mask = np.zeros((batch_size, max_atoms), dtype=np.float32)

        for i, (atom_feats, adj, bond_feats) in enumerate(valid_graphs):
            num_atoms = atom_feats.shape[0]
            padded_atom_features[i, :num_atoms] = atom_feats
            padded_adj[i, :num_atoms, :num_atoms] = adj
            padded_bond_features[i, :num_atoms, :num_atoms] = bond_feats
            atom_mask[i, :num_atoms] = 1.0

        # Handle invalid SMILES by padding with zeros and mask=0
        if len(valid_graphs) < len(smiles_list):
            extra_zeros = np.zeros((len(smiles_list) - batch_size, max_atoms, atom_fdim), dtype=np.float32)
            padded_atom_features = np.vstack([padded_atom_features, extra_zeros])
            extra_adj = np.zeros((len(smiles_list) - batch_size, max_atoms, max_atoms), dtype=np.float32)
            padded_adj = np.vstack([padded_adj, extra_adj])
            extra_bond = np.zeros((len(smiles_list) - batch_size, max_atoms, max_atoms, bond_fdim), dtype=np.float32)
            padded_bond_features = np.vstack([padded_bond_features, extra_bond])
            extra_mask = np.zeros((len(smiles_list) - batch_size, max_atoms), dtype=np.float32)
            atom_mask = np.vstack([atom_mask, extra_mask])

        return WLNGraph(padded_atom_features, padded_adj, padded_bond_features, atom_mask)