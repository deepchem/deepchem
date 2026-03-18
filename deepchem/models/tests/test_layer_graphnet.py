import pytest
from deepchem.utils.fake_data_generator import FakeGraphGenerator
try:
    import torch
    from deepchem.models.torch_models.layers import GraphNetwork
    has_torch = True
except ModuleNotFoundError:
    has_torch = False


@pytest.mark.torch
def test_graphnet_layer():
    # Testing graphnet for a single graph
    node_features = torch.randn(5, 10)
    edge_features = torch.randn(5, 3)
    global_features = torch.randn(1, 4)
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]]).long()

    graphnet = GraphNetwork(n_node_features=node_features.size(1),
                            n_edge_features=edge_features.size(1),
                            n_global_features=global_features.size(1))

    assert repr(
        graphnet
    ) == 'GraphNetwork(n_node_features=10, n_edge_features=3, n_global_features=4, is_undirected=True, residual_connection=True)'

    new_node_features, new_edge_features, new_global_features = graphnet(
        node_features, edge_index, edge_features, global_features)

    assert node_features.size() == new_node_features.size()
    assert edge_features.size() == new_edge_features.size()
    assert global_features.size() == new_global_features.size()

    # Testing for consistency
    node_features = torch.tensor([[0.7, 0.7], [0.7, 0.7]]).float()
    edge_features = torch.tensor([[0.7, 0.7]]).float()
    global_features = torch.tensor([[1]]).float()
    edge_index = torch.tensor([[0], [1]]).long()

    torch.manual_seed(12345)
    graphnet1 = GraphNetwork(n_node_features=2,
                             n_edge_features=2,
                             n_global_features=1)
    out_node1, out_edge1, out_global1 = graphnet1(node_features, edge_index,
                                                  edge_features,
                                                  global_features)

    torch.manual_seed(12345)
    graphnet2 = GraphNetwork(n_node_features=2,
                             n_edge_features=2,
                             n_global_features=1)
    out_node2, out_edge2, out_global2 = graphnet2(node_features, edge_index,
                                                  edge_features,
                                                  global_features)

    rtol = 1e-5
    atol = 1e-6
    assert torch.allclose(out_node1, out_node2, rtol=rtol, atol=atol)
    assert torch.allclose(out_edge1, out_edge2, rtol=rtol, atol=atol)
    assert torch.allclose(out_global1, out_global2, rtol=rtol, atol=atol)


@pytest.mark.torch
def test_graphnet_for_graphs_in_batch():
    # Testing with a batch of Graphs
    try:
        from torch_geometric.data import Batch
    except ModuleNotFoundError:
        raise ImportError("Tests require pytorch geometric to be installed")

    n_node_features, n_edge_features, n_global_features = 3, 4, 5
    fgg = FakeGraphGenerator(min_nodes=8,
                             max_nodes=12,
                             n_node_features=n_node_features,
                             avg_degree=10,
                             n_edge_features=n_edge_features,
                             n_classes=2,
                             task='graph',
                             z=n_global_features)
    graphs = fgg.sample(n_graphs=10)

    graphnet = GraphNetwork(n_node_features, n_edge_features, n_global_features)

    graph_batch = Batch()
    graph_batch = graph_batch.from_data_list(
        [graph.to_pyg_graph() for graph in graphs.X])

    new_node_features, new_edge_features, new_global_features = graphnet(
        graph_batch.x, graph_batch.edge_index, graph_batch.edge_attr,
        graph_batch.z, graph_batch.batch)
    assert graph_batch.x.size() == new_node_features.size()
    assert graph_batch.edge_attr.size() == new_edge_features.size()
    assert graph_batch.z.size() == new_global_features.size()
