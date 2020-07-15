import unittest
import pytest
import numpy as np
from deepchem.feat.molecule_graph import MoleculeGraphData, BatchMoleculeGraphData


class TestMoleculeGraph(unittest.TestCase):

  def test_molecule_graph_data(self):
    num_nodes, num_node_features = 10, 32
    num_edges, num_edge_features = 13, 32
    graph_features = None
    node_features = np.ones((num_nodes, num_node_features))
    edge_index = np.ones((2, num_edges))
    edge_features = np.ones((num_edges, num_edge_features))
    targets = np.ones(5)

    mol_graph = MoleculeGraphData(
        node_features=node_features,
        edge_index=edge_index,
        targets=targets,
        edge_features=edge_features,
        graph_features=graph_features)

    assert mol_graph.num_nodes == num_nodes
    assert mol_graph.num_node_features == num_node_features
    assert mol_graph.num_edges == num_edges
    assert mol_graph.num_edge_features == num_edge_features
    assert mol_graph.targets.shape == (5,)

  def test_invalid_molecule_graph_data(self):
    with pytest.raises(ValueError):
      invalid_node_features = [[0, 1, 2, 3, 4], [5, 6, 7, 8]]
      edge_index = np.ones((2, 5))
      targets = np.ones(5)
      mol_graph = MoleculeGraphData(
          node_features=invalid_node_features,
          edge_index=edge_index,
          targets=targets,
      )

    with pytest.raises(ValueError):
      node_features = np.ones((5, 5))
      invalid_edge_index_shape = np.ones((3, 10))
      targets = np.ones(5)
      mol_graph = MoleculeGraphData(
          node_features=node_features,
          edge_index=invalid_edge_index_shape,
          targets=targets,
      )

    with pytest.raises(TypeError):
      node_features = np.ones((5, 5))
      mol_graph = MoleculeGraphData(node_features=node_features,)

  def test_batch_molecule_graph_data(self):

    num_nodes_list, num_edge_list = [5, 7, 10], [6, 10, 20]
    num_node_features, num_edge_features = 32, 32
    targets = np.ones(5)
    molecule_graph_list = [
        MoleculeGraphData(
            node_features=np.ones((num_nodes, num_node_features)),
            edge_index=np.ones((2, num_edges)),
            targets=targets,
            edge_features=np.ones((num_edges, num_edge_features)),
            graph_features=None)
        for num_nodes, num_edges in zip(num_nodes_list, num_edge_list)
    ]

    batch = BatchMoleculeGraphData(molecule_graph_list)
    assert batch.num_nodes == sum(num_nodes_list)
    assert batch.num_node_features == num_node_features
    assert batch.num_edges == sum(num_edge_list)
    assert batch.num_edge_features == num_edge_features
    assert batch.targets.shape == (3, 5)
    assert batch.graph_idx.shape == (sum(num_nodes_list),)
