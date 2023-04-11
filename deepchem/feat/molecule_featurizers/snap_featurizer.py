from deepchem.feat import MolecularFeaturizer, Featurizer
from rdkit import Chem
import numpy as np
from deepchem.feat.graph_data import GraphData

allowable_features = {
    'possible_atomic_num_list':
        list(range(1, 119)),
    'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'possible_chirality_list': [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],
    'possible_hybridization_list': [
        Chem.rdchem.HybridizationType.S, Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D, Chem.rdchem.HybridizationType.SP3D2,
        Chem.rdchem.HybridizationType.UNSPECIFIED
    ],
    'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'possible_implicit_valence_list': [0, 1, 2, 3, 4, 5, 6],
    'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'possible_bonds': [
        Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC
    ],
    'possible_bond_dirs': [  # only for double bond stereo information
        Chem.rdchem.BondDir.NONE, Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT
    ]
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


class EgoGraphFeaturizer(Featurizer):
    """
    This featurizer converts an ego graph to a GraphData object.

    References
    ----------

    .. [1] Hu, W. et al. Strategies for Pre-training Graph Neural Networks. Preprint at https://doi.org/10.48550/arXiv.1905.12265 (2020).

    """

    def __init__(self,
                 allowable_features_downstream=None,
                 allowable_features_pretrain=None,
                 node_id_to_go_labels=None):
        self.allowable_features_downstream = allowable_features_downstream
        self.allowable_features_pretrain = allowable_features_pretrain
        self.node_id_to_go_labels = node_id_to_go_labels
        super().__init__()

    def _featurize(self, g, **kwargs):
        """
        Converts an ego nx graph to a GraphData object.

        Parameters
        ----------
        g : nx.Graph
            NetworkX graph object of the ego graph.
        center_id : str
            Node ID of the center node in the ego graph.
        allowable_features_downstream : list, optional
            List of possible GO function node features for the downstream task.
            The resulting go_target_downstream node feature vector will be in this order.
        allowable_features_pretrain : list, optional
            List of possible GO function node features for the pretraining task.
            The resulting go_target_pretrain node feature vector will be in this order.
        node_id_to_go_labels : dict, optional
            Dictionary that maps node ID to a list of its corresponding GO labels.

        Returns
        -------
        data: GraphData
            GraphData object with the following attributes:
            - edge_attr
            - edge_index
            - x
            - species_id
            - center_node_idx
            - go_target_downstream (only if node_id_to_go_labels is not None)
            - go_target_pretrain (only if node_id_to_go_labels is not None)

        Example
        -------
        >>> import networkx as nx
        >>> g = nx.Graph()
        >>> g.add_node("1")
        >>> g.add_node("2")
        >>> g.add_edge("1", "2", w1=1, w2=2, w3=3, w4=4, w5=5, w6=6, w7=7)
        >>> featurizer = EgoGraphFeaturizer()
        >>> print(featurizer.featurize(g, "1"))
        [Data(edge_attr=[2, 9], edge_index=[2, 2], x=[2, 1], species_id=[1], center_node_idx=[1])]
        """
        g, center_id = g
        n_nodes = g.number_of_nodes()
        # n_edges = g.number_of_edges() unused?

        # nodes
        nx_node_ids = [n_i for n_i in g.nodes()]  # contains list of nx node ids
        # in a particular ordering. Will be used as a mapping to convert
        # between nx node ids and data obj node indices

        x = np.ones(n_nodes).reshape(-1, 1)
        # we don't have any node labels, so set to dummy 1. dim n_nodes x 1

        center_node_idx = nx_node_ids.index(center_id)
        center_node_idx = np.array([center_node_idx])

        # edges
        edges_list = []
        edge_features_list = []
        for node_1, node_2, attr_dict in g.edges(data=True):
            edge_feature = [
                attr_dict['w1'], attr_dict['w2'], attr_dict['w3'],
                attr_dict['w4'], attr_dict['w5'], attr_dict['w6'],
                attr_dict['w7'], 0, 0
            ]  # last 2 indicate self-loop
            # and masking
            edge_feature = np.array(edge_feature, dtype=int)
            # convert nx node ids to data obj node index
            i = nx_node_ids.index(node_1)
            j = nx_node_ids.index(node_2)
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        if len(edges_list) == 0:
            raise ValueError('Ego graph has no edges. Skipping.')
        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = np.array(edges_list, dtype=int).T

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = np.array(edge_features_list)

        try:
            species_id = int(
                nx_node_ids[0].split('.')[0])  # nx node id is of the form:
            # species_id.protein_id
            species_id = np.array([species_id])
        except:  # occurs when nx node id has no species id info. For the extract
            # substructure context pair transform, where we convert a data obj to
            # a nx graph obj (which does not have original node id info)
            species_id = np.array([0])  # dummy species
            # id is 0

        # construct data obj
        data = GraphData(node_features=x, edge_index=edge_index, edge_features=edge_attr)
        data.species_id = species_id
        data.center_node_idx = center_node_idx

        if self.node_id_to_go_labels:  # supervised case with go node labels
            # Construct a dim n_pretrain_go_classes tensor and a
            # n_downstream_go_classes tensor for the center node. 0 is no data
            # or negative, 1 is positive.
            downstream_go_node_feature = [0] * len(
                self.allowable_features_downstream)
            pretrain_go_node_feature = [0] * len(
                self.allowable_features_pretrain)
            if center_id in self.node_id_to_go_labels:
                go_labels = self.node_id_to_go_labels[center_id]
                # get indices of allowable_features_downstream that match with elements
                # in go_labels
                _, node_feature_indices, _ = np.intersect1d(
                    self.allowable_features_downstream,
                    go_labels,
                    return_indices=True)
                for idx in node_feature_indices:
                    downstream_go_node_feature[idx] = 1
                # get indices of allowable_features_pretrain that match with
                # elements in go_labels
                _, node_feature_indices, _ = np.intersect1d(
                    self.allowable_features_pretrain,
                    go_labels,
                    return_indices=True)
                for idx in node_feature_indices:
                    pretrain_go_node_feature[idx] = 1
            data.go_target_downstream = np.array(downstream_go_node_feature)
            data.go_target_pretrain = np.array(pretrain_go_node_feature)

        return data
