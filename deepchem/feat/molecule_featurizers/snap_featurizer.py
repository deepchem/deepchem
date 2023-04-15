from deepchem.feat import MolecularFeaturizer
from rdkit import Chem
import numpy as np
from deepchem.feat.graph_data import GraphData

allowable_features = {
    'possible_atomic_num_list':
        list(range(0, 119)),  # 0 represents a masked atom
    'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'possible_chirality_list': [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],  # noqa: E122
    'possible_hybridization_list': [
        Chem.rdchem.HybridizationType.S, Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D, Chem.rdchem.HybridizationType.SP3D2,
        Chem.rdchem.HybridizationType.UNSPECIFIED
    ],  # noqa: E122
    'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'possible_implicit_valence_list': [0, 1, 2, 3, 4, 5, 6],
    'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    # 0 represents a masked bond
    'possible_bonds': [
        0, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC
    ],  # noqa: E122
    'possible_bond_dirs': [  # only for double bond stereo information
        Chem.rdchem.BondDir.NONE, Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT
    ]  # noqa: E122
}


class SNAPFeaturizer(MolecularFeaturizer):
    """
    This featurizer is based on the SNAP featurizer used in the paper [1].

    Example
    -------
    >>> smiles = ["CC(=O)C"]
    >>> featurizer = SNAPFeaturizer()
    >>> print(featurizer.featurize(smiles))
    [GraphData(node_features=[4, 2], edge_index=[2, 6], edge_features=[6, 2])]

    References
    ----------

    .. [1] Hu, W. et al. Strategies for Pre-training Graph Neural Networks. Preprint at https://doi.org/10.48550/arXiv.1905.12265 (2020).

    """

    def _featurize(self, mol, **kwargs):
        """
        Converts rdkit mol object to the deepchem Graph Data object. Uses
        simplified atom and bond features, represented as indices.

        Parameters
        ----------
        mol: RDKitMol
            RDKit molecule object

        Returns
        -------
        data: GraphData
            Graph data object with the attributes: x, edge_index, edge_features

        """
        # atoms
        atom_features_list = []
        for atom in mol.GetAtoms():
            atom_feature = [
                allowable_features['possible_atomic_num_list'].index(
                    atom.GetAtomicNum())
            ] + [
                allowable_features['possible_chirality_list'].index(
                    atom.GetChiralTag())
            ]
            atom_features_list.append(atom_feature)
        x = np.array(atom_features_list)

        # bonds
        num_bond_features = 2  # bond type, bond direction
        if len(mol.GetBonds()) > 0:  # mol has bonds
            edges_list = []
            edge_features_list = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                edge_feature = [
                    allowable_features['possible_bonds'].index(
                        bond.GetBondType())
                ] + [
                    allowable_features['possible_bond_dirs'].index(
                        bond.GetBondDir())
                ]
                edges_list.append((i, j))
                edge_features_list.append(edge_feature)
                edges_list.append((j, i))
                edge_features_list.append(edge_feature)

            # Graph connectivity in COO format with shape [2, num_edges]
            edge_index = np.array(edges_list).T

            # Edge feature matrix with shape [num_edges, num_edge_features]
            edge_feats = np.array(edge_features_list)
        else:  # mol has no bonds
            edge_index = np.empty((2, 0), dtype=np.int8)
            edge_feats = np.empty((0, num_bond_features), dtype=np.int8)

        data = GraphData(node_features=x,
                         edge_index=edge_index,
                         edge_features=edge_feats)

        return data
