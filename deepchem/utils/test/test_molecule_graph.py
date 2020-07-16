import unittest
import pytest
import numpy as np
from deepchem.utils.molecule_graph import MoleculeGraphData, BatchMoleculeGraphData


class TestMoleculeGraph(unittest.TestCase):

  def test_molecule_graph_data(self):
    num_nodes, num_node_features = 4, 32
    num_edges, num_edge_features = 6, 32
    node_features = np.random.random_sample((num_nodes, num_node_features))
    edge_features = np.random.random_sample((num_edges, num_edge_features))
    targets = np.random.random_sample(5)
    edge_index = np.array([
        [0, 1, 2, 2, 3, 4],
        [1, 2, 0, 3, 4, 0],
    ])
    graph_features = None

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
      invalid_node_features_type = list(np.random.random_sample((5, 5)))
      edge_index = np.array([
          [0, 1, 2, 2, 3, 4],
          [1, 2, 0, 3, 4, 0],
      ])
      targets = np.random.random_sample(5)
      mol_graph = MoleculeGraphData(
          node_features=invalid_node_features_type,
          edge_index=edge_index,
          targets=targets,
      )

    with pytest.raises(ValueError):
      node_features = np.random.random_sample((5, 5))
      invalid_edge_index_shape = np.array([
          [0, 1, 2, 2, 3, 4],
          [1, 2, 0, 3, 4, 0],
          [2, 2, 1, 4, 0, 3],
      ])
      targets = np.random.random_sample(5)
      mol_graph = MoleculeGraphData(
          node_features=node_features,
          edge_index=invalid_edge_index_shape,
          targets=targets,
      )

    with pytest.raises(TypeError):
      node_features = np.random.random_sample((5, 5))
      mol_graph = MoleculeGraphData(node_features=node_features)

  def test_batch_molecule_graph_data(self):
    num_nodes_list, num_edge_list = [3, 4, 5], [2, 4, 5]
    num_node_features, num_edge_features = 32, 32
    edge_index_list = [
        np.array([[0, 1], [1, 2]]),
        np.array([[0, 1, 2, 3], [1, 2, 0, 2]]),
        np.array([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]])
    ]
    targets = np.random.random_sample(5)

    molecule_graphs = [
        MoleculeGraphData(
            node_features=np.random.random_sample((num_nodes_list[i],
                                                   num_node_features)),
            edge_index=edge_index_list[i],
            targets=targets,
            edge_features=np.random.random_sample((num_edge_list[i],
                                                   num_edge_features)),
            graph_features=None) for i in range(len(num_edge_list))
    ]
    batch = BatchMoleculeGraphData(molecule_graphs)

    assert batch.num_nodes == sum(num_nodes_list)
    assert batch.num_node_features == num_node_features
    assert batch.num_edges == sum(num_edge_list)
    assert batch.num_edge_features == num_edge_features
    assert batch.targets.shape == (3, 5)
    assert batch.graph_index.shape == (sum(num_nodes_list),)
