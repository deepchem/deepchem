"""
Test for fake_data_generator.py
"""
import numpy as np
from deepchem.utils.fake_data_generator import FakeGraphGenerator, generate_edge_index, remove_self_loops


def test_fake_graph_dataset():
  n_graphs = 10
  n_node_features = 5
  n_edge_features = 3
  n_classes = 2
  z_shape = 5

  # graph-level labels
  fgg = FakeGraphGenerator(
      min_nodes=3,
      max_nodes=10,
      n_node_features=n_node_features,
      avg_degree=4,
      n_edge_features=n_edge_features,
      n_classes=n_classes,
      task='graph',
      z=z_shape)
  graphs = fgg.sample(n_graphs=n_graphs)

  assert len(graphs) == n_graphs
  assert np.unique(graphs.y).shape == (n_classes,)

  graph = graphs.X[0]
  assert graph.node_features.shape[1] == n_node_features
  assert graph.edge_features.shape[1] == n_edge_features
  assert graph.z.shape == (1, z_shape)

  # node-level labels
  fgg = FakeGraphGenerator(
      min_nodes=3,
      max_nodes=10,
      n_node_features=n_node_features,
      avg_degree=4,
      n_edge_features=n_edge_features,
      n_classes=n_classes,
      task='node',
      z=z_shape)
  graphs = fgg.sample(n_graphs=n_graphs)

  assert len(graphs) == n_graphs

  graph = graphs.X[0]
  # graph.y contains node-labels and graph.node_features.shape[0]
  # holds number of nodes in that graph
  assert graph.y.shape[0] == graph.node_features.shape[0]
  assert graph.node_features.shape[1] == n_node_features
  assert graph.edge_features.shape[1] == n_edge_features
  assert graph.z.shape == (1, z_shape)


def test_generate_edge_index():
  n_nodes, avg_degree = 5, 3
  edge_indices = generate_edge_index(n_nodes, avg_degree, remove_loops=False)
  assert edge_indices.shape[0] == 2
  assert edge_indices.shape[1] == n_nodes * avg_degree


def test_remove_self_loops():
  edge_indices = np.array([[1, 2, 3], [1, 2, 4]])
  edge_indices = remove_self_loops(edge_indices)
  assert edge_indices.shape[0] == 2
  assert edge_indices.shape[1] == 1

  edge_indices = np.ones((2, 3))
  edge_indices = remove_self_loops(edge_indices)
  assert edge_indices.shape[0] == 2
  assert edge_indices.shape[1] == 0
