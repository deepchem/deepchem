import unittest

import numpy as np
import pytest

from deepchem.feat.graph_data import BatchGraphData, GraphData, shortest_path_length, WeightedDirectedGraphData


class TestGraph(unittest.TestCase):

    @pytest.mark.torch
    def test_graph_data(self):
        num_nodes, num_node_features = 5, 32
        num_edges, num_edge_features = 6, 32
        node_features = np.random.random_sample((num_nodes, num_node_features))
        edge_features = np.random.random_sample((num_edges, num_edge_features))
        edge_index = np.array([
            [0, 1, 2, 2, 3, 4],
            [1, 2, 0, 3, 4, 0],
        ])
        node_pos_features = None
        # z is kwargs
        z = np.random.random(5)

        graph = GraphData(node_features=node_features,
                          edge_index=edge_index,
                          edge_features=edge_features,
                          node_pos_features=node_pos_features,
                          z=z)

        assert graph.num_nodes == num_nodes
        assert graph.num_node_features == num_node_features
        assert graph.num_edges == num_edges
        assert graph.num_edge_features == num_edge_features
        assert graph.z.shape == z.shape
        assert str(
            graph
        ) == 'GraphData(node_features=[5, 32], edge_index=[2, 6], edge_features=[6, 32], z=[5])'

        # check convert function
        pyg_graph = graph.to_pyg_graph()
        from torch_geometric.data import Data
        assert isinstance(pyg_graph, Data)
        assert tuple(pyg_graph.z.shape) == z.shape

        dgl_graph = graph.to_dgl_graph()
        from dgl import DGLGraph
        assert isinstance(dgl_graph, DGLGraph)

    @pytest.mark.torch
    def test_invalid_graph_data(self):
        with self.assertRaises(ValueError):
            invalid_node_features_type = list(np.random.random_sample((5, 32)))
            edge_index = np.array([
                [0, 1, 2, 2, 3, 4],
                [1, 2, 0, 3, 4, 0],
            ])
            _ = GraphData(
                node_features=invalid_node_features_type,
                edge_index=edge_index,
            )

        with self.assertRaises(ValueError):
            node_features = np.random.random_sample((5, 32))
            invalid_node_index_in_edge_index = np.array([
                [0, 1, 2, 2, 3, 4],
                [1, 2, 0, 3, 4, 5],
            ])
            _ = GraphData(
                node_features=node_features,
                edge_index=invalid_node_index_in_edge_index,
            )

        with self.assertRaises(ValueError):
            node_features = np.random.random_sample((5, 5))
            invalid_edge_index_shape = np.array([
                [0, 1, 2, 2, 3, 4],
                [1, 2, 0, 3, 4, 0],
                [2, 2, 1, 4, 0, 3],
            ],)
            _ = GraphData(
                node_features=node_features,
                edge_index=invalid_edge_index_shape,
            )

        with self.assertRaises(TypeError):
            node_features = np.random.random_sample((5, 32))
            _ = GraphData(node_features=node_features)

    @pytest.mark.torch
    def test_batch_graph_data(self):
        num_nodes_list, num_edge_list = [3, 4, 5], [2, 4, 5]
        num_node_features, num_edge_features = 32, 32
        edge_index_list = [
            np.array([[0, 1], [1, 2]]),
            np.array([[0, 1, 2, 3], [1, 2, 0, 2]]),
            np.array([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]]),
        ]

        graph_list = [
            GraphData(node_features=np.random.random_sample(
                (num_nodes_list[i], num_node_features)),
                      edge_index=edge_index_list[i],
                      edge_features=np.random.random_sample(
                          (num_edge_list[i], num_edge_features)),
                      node_pos_features=None) for i in range(len(num_edge_list))
        ]
        batch = BatchGraphData(graph_list)

        assert batch.num_nodes == sum(num_nodes_list)
        assert batch.num_node_features == num_node_features
        assert batch.num_edges == sum(num_edge_list)
        assert batch.num_edge_features == num_edge_features
        assert batch.graph_index.shape == (sum(num_nodes_list),)
        assert batch.edge_index.max() == sum(num_edge_list)
        assert batch.edge_index.shape == (2, sum(num_edge_list))

    @pytest.mark.torch
    def test_graph_data_single_atom_mol(self):
        """
        Test for graph data when no edges in the graph (example: single atom mol)
        """
        num_nodes, num_node_features = 1, 32
        num_edges = 0
        node_features = np.random.random_sample((num_nodes, num_node_features))
        edge_index = np.empty((2, 0), dtype=int)

        graph = GraphData(node_features=node_features, edge_index=edge_index)

        assert graph.num_nodes == num_nodes
        assert graph.num_node_features == num_node_features
        assert graph.num_edges == num_edges
        assert str(
            graph
        ) == 'GraphData(node_features=[1, 32], edge_index=[2, 0], edge_features=None)'

    @pytest.mark.torch
    def test_graphdata_numpy_to_torch(self):
        """
        Test for converting GraphData numpy arrays to torch tensors
        """
        import torch
        num_nodes, num_node_features = 5, 32
        num_edges, num_edge_features = 6, 32
        node_features = np.random.random_sample((num_nodes, num_node_features))
        edge_features = np.random.random_sample((num_edges, num_edge_features))
        edge_index = np.array([
            [0, 1, 2, 2, 3, 4],
            [1, 2, 0, 3, 4, 0],
        ])
        node_pos_features = None
        # z is kwargs
        z = np.random.random(5)

        graph_np = GraphData(node_features=node_features,
                             edge_index=edge_index,
                             edge_features=edge_features,
                             node_pos_features=node_pos_features,
                             z=z)
        graph = graph_np.numpy_to_torch()

        assert isinstance(graph.node_features, torch.Tensor)
        assert isinstance(graph.edge_index, torch.Tensor)
        assert isinstance(graph.edge_features, torch.Tensor)
        assert graph.node_pos_features is None
        assert isinstance(graph.z, torch.Tensor)

    @pytest.mark.torch
    def test_batchgraphdata_numpy_to_torch(self):
        import torch
        num_nodes_list, num_edge_list = [3, 4, 5], [2, 4, 5]
        num_node_features, num_edge_features = 32, 32
        edge_index_list = [
            np.array([[0, 1], [1, 2]]),
            np.array([[0, 1, 2, 3], [1, 2, 0, 2]]),
            np.array([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]]),
        ]

        graph_list = [
            GraphData(node_features=np.random.random_sample(
                (num_nodes_list[i], num_node_features)),
                      edge_index=edge_index_list[i],
                      edge_features=np.random.random_sample(
                          (num_edge_list[i], num_edge_features)),
                      node_pos_features=None) for i in range(len(num_edge_list))
        ]
        batched_graph = BatchGraphData(graph_list)

        batched_graph = batched_graph.numpy_to_torch()

        assert isinstance(batched_graph, BatchGraphData)
        assert isinstance(batched_graph.node_features, torch.Tensor)
        assert isinstance(batched_graph.edge_index, torch.Tensor)
        assert isinstance(batched_graph.edge_features, torch.Tensor)
        assert batched_graph.node_pos_features is None

    def test_batch_graph_data_with_user_defined_attributes(self):
        edge_index = np.array([[0, 1], [1, 0]])
        node_features_shape = 5
        n_nodes = 2
        g1 = GraphData(node_features=np.random.randn(n_nodes,
                                                     node_features_shape),
                       edge_index=edge_index,
                       user_defined_attribute1=[0, 1])

        g2 = GraphData(node_features=np.random.randn(n_nodes,
                                                     node_features_shape),
                       edge_index=edge_index,
                       user_defined_attribute1=[2, 3])

        g3 = GraphData(node_features=np.random.randn(n_nodes,
                                                     node_features_shape),
                       edge_index=edge_index,
                       user_defined_attribute1=[4, 5])
        g = BatchGraphData([g1, g2, g3])

        assert hasattr(g, 'user_defined_attribute1')
        assert (g.user_defined_attribute1 == np.array([[0, 1], [2, 3],
                                                       [4, 5]])).all()

    def test_shortest_path_length(self):
        node_features = np.random.rand(5, 10)
        edge_index = np.array([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]],
                              dtype=np.int64)
        graph_data = GraphData(node_features, edge_index)

        lengths = shortest_path_length(graph_data, 0)
        assert lengths == {0: 0, 1: 1, 2: 2, 3: 2, 4: 1}

        lengths_cutoff = shortest_path_length(graph_data, 0, cutoff=1)
        assert lengths_cutoff == {0: 0, 1: 1, 4: 1}

    def test_subgraph(self):
        node_features = np.random.rand(5, 10)
        edge_index = np.array([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]],
                              dtype=np.int64)
        edge_features = np.random.rand(5, 3)
        graph_data = GraphData(node_features, edge_index, edge_features)

        nodes = [0, 1, 2, 4]
        subgraph, node_mapping = graph_data.subgraph(nodes)

        assert subgraph.num_nodes == len(nodes)
        assert subgraph.num_edges == 3

        expected_node_features = node_features[nodes]
        np.testing.assert_array_equal(subgraph.node_features,
                                      expected_node_features)

        expected_edge_index = np.array([[0, 1, 3], [1, 2, 0]], dtype=np.int64)
        np.testing.assert_array_equal(subgraph.edge_index, expected_edge_index)

        expected_edge_features = edge_features[[0, 1, 4]]
        np.testing.assert_array_equal(subgraph.edge_features,
                                      expected_edge_features)

        expected_node_mapping = {0: 0, 1: 1, 2: 2, 4: 3}
        assert node_mapping == expected_node_mapping


class TestWeightedDirectedGraph(unittest.TestCase):

    def test_wdgraph_data(self):
        num_nodes, num_node_features = 5, 32
        num_edges, num_edge_features = 7, 32
        node_features = np.random.random_sample((num_nodes, num_node_features))
        edge_features = np.random.random_sample((num_edges, num_edge_features))
        node_to_edge_mapping = [[min(0, x - 1), x] for x in range(num_nodes)]
        node_weights = np.random.rand(num_nodes)
        edge_weights = np.random.rand(num_edges)
        edge_to_node_mapping = np.array([x for x in range(num_edges)])
        edge_to_reverse_edge_mapping = np.array(
            [x for x in range(num_edges)[::-1]])
        # z is kwargs
        z = np.random.random(5)

        wd_graph = WeightedDirectedGraphData(
            node_features=node_features,
            edge_features=edge_features,
            node_to_edge_mapping=node_to_edge_mapping,
            node_weights=node_weights,
            edge_weights=edge_weights,
            edge_to_node_mapping=edge_to_node_mapping,
            edge_to_reverse_edge_mapping=edge_to_reverse_edge_mapping,
            z=z)

        assert wd_graph.num_nodes == num_nodes
        assert wd_graph.num_node_features == num_node_features
        assert wd_graph.num_edges == num_edges
        assert wd_graph.num_edge_features == num_edge_features
        assert wd_graph.z.shape == z.shape
        assert str(
            wd_graph
        ) == """WeightedDirectedGraphData(node_features=[5, 32], edge_features=[7, 32],
                node_to_edge_mapping=5, node_weights=[5],
                edge_weights=[7], edge_to_node_mapping=7,
                edge_to_reverse_edge_mapping=7, z=[5])"""

    def test_invalid_nodee_to_edge_wdgraph_data(self):
        with self.assertRaises(ValueError):
            node_features = np.random.rand(5, 10)
            edge_features = np.random.rand(7, 10)
            invalid_node_to_edge_mapping = [[x, x] for x in range(10)]
            node_weights = np.random.rand(5)
            edge_weights = np.random.rand(7)
            _ = WeightedDirectedGraphData(
                node_features=node_features,
                edge_features=edge_features,
                node_to_edge_mapping=invalid_node_to_edge_mapping,
                node_weights=node_weights,
                edge_weights=edge_weights)

        with self.assertRaises(ValueError):
            node_features = np.random.rand(5, 10)
            edge_features = np.random.rand(7, 10)
            invalid_shape_node_to_edge_mapping = [x for x in range(10)]
            node_weights = np.random.rand(5)
            edge_weights = np.random.rand(7)
            _ = WeightedDirectedGraphData(
                node_features=node_features,
                edge_features=edge_features,
                node_to_edge_mapping=invalid_shape_node_to_edge_mapping,
                node_weights=node_weights,
                edge_weights=edge_weights)

        with self.assertRaises(ValueError):
            node_features = np.random.rand(5, 10)
            edge_features = np.random.rand(7, 10)
            invaild_value_node_to_edge_mapping = [[x, x] for x in range(4)
                                                 ].append([1, 69])
            node_weights = np.random.rand(5)
            edge_weights = np.random.rand(7)
            _ = WeightedDirectedGraphData(
                node_features=node_features,
                edge_features=edge_features,
                node_to_edge_mapping=invaild_value_node_to_edge_mapping,
                node_weights=node_weights,
                edge_weights=edge_weights)

    def test_invalid_weight_wdgraph_data(self):
        with self.assertRaises(ValueError):
            node_features = np.random.rand(5, 10)
            edge_features = np.random.rand(7, 10)
            node_to_edge_mapping = [[x, x] for x in range(5)]
            invalid_shape_node_weights = np.random.rand(4)
            edge_weights = np.random.rand(7)
            _ = WeightedDirectedGraphData(
                node_features=node_features,
                edge_features=edge_features,
                node_to_edge_mapping=node_to_edge_mapping,
                node_weights=invalid_shape_node_weights,
                edge_weights=edge_weights)

        with self.assertRaises(ValueError):
            node_features = np.random.rand(5, 10)
            edge_features = np.random.rand(7, 10)
            node_to_edge_mapping = [[x, x] for x in range(5)]
            node_weights = np.random.rand(5)
            invalid_shape_edge_weights = np.random.rand()
            _ = WeightedDirectedGraphData(
                node_features=node_features,
                edge_features=edge_features,
                node_to_edge_mapping=node_to_edge_mapping,
                node_weights=node_weights,
                edge_weights=invalid_shape_edge_weights)

    def test_invalid_edge_to_node_wdgraph_data(self):
        with self.assertRaises(ValueError):
            node_features = np.random.rand(5, 10)
            edge_features = np.random.rand(7, 10)
            node_to_edge_mapping = [[x, x] for x in range(5)]
            node_weights = np.random.rand(5)
            edge_weights = np.random.rand(7)
            invalid_value_edge_to_node_mapping = np.array([x for x in range(6)
                                                          ].append(69))
            edge_to_reverse_edge_mapping = np.array([x for x in range(7)[::-1]])
            _ = WeightedDirectedGraphData(
                node_features=node_features,
                edge_features=edge_features,
                node_to_edge_mapping=node_to_edge_mapping,
                node_weights=node_weights,
                edge_weights=edge_weights,
                edge_to_node_mapping=invalid_value_edge_to_node_mapping,
                edge_to_reverse_edge_mapping=edge_to_reverse_edge_mapping)

    def test_invalid_edge_to_reverse_edge_wdgraph_data(self):
        with self.assertRaises(ValueError):
            node_features = np.random.rand(5, 10)
            edge_features = np.random.rand(7, 10)
            node_to_edge_mapping = [[x, x] for x in range(5)]
            node_weights = np.random.rand(5)
            edge_weights = np.random.rand(7)
            edge_to_node_mapping = np.array([x for x in range(7)])
            invalid_edge_to_reverse_edge_mapping = np.array(
                [x for x in range(6)[::-1]].append(69))
            _ = WeightedDirectedGraphData(
                node_features=node_features,
                edge_features=edge_features,
                node_to_edge_mapping=node_to_edge_mapping,
                node_weights=node_weights,
                edge_weights=edge_weights,
                edge_to_node_mapping=edge_to_node_mapping,
                edge_to_reverse_edge_mapping=invalid_edge_to_reverse_edge_mapping
            )
