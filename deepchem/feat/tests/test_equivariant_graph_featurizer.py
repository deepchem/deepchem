import unittest
from rdkit import Chem
from deepchem.feat import EquivariantGraphFeaturizer
from deepchem.feat.graph_data import GraphData
import numpy as np


class TestEquivariantGraphFeaturizer(unittest.TestCase):
    """
    Tests for EquivariantGraphFeaturizer.
    """

    def setUp(self):
        """
        Set up tests with a simple molecule.
        """
        smiles = 'CCO'
        self.mol = Chem.MolFromSmiles(smiles)

    def test_bonded_graph_featurization(self):
        """
        Test featurization with bonded edges.
        """
        featurizer = EquivariantGraphFeaturizer(fully_connected=False,
                                                embeded=True)
        graphs = featurizer.featurize([self.mol])

        assert len(graphs) == 1
        graph = graphs[0]

        assert isinstance(graph, GraphData)

        assert graph.node_features.shape[0] == self.mol.GetNumAtoms()
        assert graph.positions.shape[0] == self.mol.GetNumAtoms()
        assert graph.edge_index.shape[1] > 0

    def test_fully_connected_graph_featurization(self):
        """
        Test featurization with fully connected edges.
        """
        featurizer = EquivariantGraphFeaturizer(fully_connected=True,
                                                embeded=True)
        graphs = featurizer.featurize([self.mol])
        assert len(graphs) == 1

        graph = graphs[0]
        assert isinstance(graph, GraphData)

        num_atoms = self.mol.GetNumAtoms()

        expected_edges = num_atoms * (num_atoms - 1)  # Fully connected graph
        assert graph.edge_index.shape[1] == expected_edges

    def test_embedded_coordinates(self):
        """
        Test featurization with embedded 3D coordinates.
        """
        featurizer = EquivariantGraphFeaturizer(embeded=True)
        graphs = featurizer.featurize([self.mol])
        assert len(graphs) == 1

        graph = graphs[0]
        assert isinstance(graph, GraphData)
        # 3D positions
        assert graph.positions.shape[1] == 3

    def test_edge_weight_discretization(self):
        """
        Test discretization of edge weights.
        """
        featurizer = EquivariantGraphFeaturizer(weight_bins=[1.0, 2.0, 3.0],
                                                embeded=True)

        graphs = featurizer.featurize([self.mol])
        assert len(graphs) == 1
        graph = graphs[0]
        assert isinstance(graph, GraphData)

        one_hot_weights = graph.edge_weights
        assert one_hot_weights.shape[1] == len(
            featurizer.weight_bins) + 1  # Bin count + 1
        assert np.all(np.sum(one_hot_weights, axis=1) == 1)

    def test_multiple_molecules(self):
        """
        Test featurization of multiple molecules.
        """
        smiles_list = ['CCO', 'CCC']
        mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
        featurizer = EquivariantGraphFeaturizer(fully_connected=False,
                                                embeded=True)
        graphs = featurizer.featurize(mols)
        assert len(graphs) == len(smiles_list)

        for graph, mol in zip(graphs, mols):
            assert graph.node_features.shape[0] == mol.GetNumAtoms()
