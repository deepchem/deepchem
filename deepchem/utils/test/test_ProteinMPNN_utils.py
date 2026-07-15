import pytest

try:
    import torch
    has_torch = True
except ModuleNotFoundError:
    has_torch = False

from deepchem.utils.ProteinMPNN_utils import gather_edges, gather_nodes, cat_neighbors_nodes


@pytest.mark.torch
def test_gather_edges_output_shape():
    """Test that _gather_edges returns a tensor with the correct shape.

    Given a pairwise edge feature tensor of shape
    ``(batch, num_nodes, num_nodes, edge_features)`` and a neighbor index
    tensor of shape ``(batch, num_nodes, k)``, the output should have shape
    ``(batch, num_nodes, k, edge_features)``.
    """

    batch, num_nodes, k, edge_features = 2, 5, 3, 16
    edges = torch.rand(batch, num_nodes, num_nodes, edge_features)
    neighbor_idx = torch.randint(0, num_nodes, (batch, num_nodes, k))

    out = gather_edges(edges, neighbor_idx)

    assert out.shape == torch.Size([batch, num_nodes, k, edge_features])


@pytest.mark.torch
def test_gather_edges_values():
    """Test that _gather_edges gathers the correct edge feature vectors.

    Constructs a small, deterministic edge tensor and manually verifies
    that each gathered slice matches the edge indexed by the corresponding
    neighbor index.
    """

    # edges[0, i, j] = [i*10 + j, i*10 + j + 1] (distinct per (i,j) pair)
    edges = torch.tensor([[[[0., 1.], [10., 11.], [20., 21.]],
                           [[100., 101.], [110., 111.], [120., 121.]],
                           [[200., 201.], [210., 211.],
                            [220., 221.]]]])  # (1, 3, 3, 2)

    # For node 0: neighbors are [1, 2]; for node 1: [0, 2]; for node 2: [0, 1]
    neighbor_idx = torch.tensor([[[1, 2], [0, 2], [0, 1]]])  # (1, 3, 2)

    out = gather_edges(edges, neighbor_idx)  # (1, 3, 2, 2)

    # Node 0, neighbor 1 → edges[0, 0, 1] = [10, 11]
    assert torch.allclose(out[0, 0, 0], torch.tensor([10., 11.]))
    # Node 0, neighbor 2 → edges[0, 0, 2] = [20, 21]
    assert torch.allclose(out[0, 0, 1], torch.tensor([20., 21.]))
    # Node 1, neighbor 0 → edges[0, 1, 0] = [100, 101]
    assert torch.allclose(out[0, 1, 0], torch.tensor([100., 101.]))
    # Node 2, neighbor 1 → edges[0, 2, 1] = [210, 211]
    assert torch.allclose(out[0, 2, 1], torch.tensor([210., 211.]))


@pytest.mark.torch
def test_gather_edges_single_neighbor():
    """Test gather_edges with k=1 (each node has exactly one neighbor)."""

    batch, num_nodes, edge_features = 1, 4, 8
    edges = torch.rand(batch, num_nodes, num_nodes, edge_features)
    neighbor_idx = torch.randint(0, num_nodes, (batch, num_nodes, 1))

    out = gather_edges(edges, neighbor_idx)

    assert out.shape == torch.Size([batch, num_nodes, 1, edge_features])
    # Verify each gathered vector matches the indexed row of edges
    for n in range(num_nodes):
        idx = neighbor_idx[0, n, 0].item()
        assert torch.allclose(out[0, n, 0], edges[0, n, idx])


@pytest.mark.torch
def test_gather_nodes_output_shape():
    """Test that gather_nodes returns a tensor with the correct shape.

    Given a node feature tensor of shape ``(batch, num_nodes, node_features)``
    and a neighbor index tensor of shape ``(batch, num_nodes, k)``, the output
    should have shape ``(batch, num_nodes, k, node_features)``.
    """

    batch, num_nodes, k, node_features = 2, 5, 3, 32
    nodes = torch.rand(batch, num_nodes, node_features)
    neighbor_idx = torch.randint(0, num_nodes, (batch, num_nodes, k))

    out = gather_nodes(nodes, neighbor_idx)

    assert out.shape == torch.Size([batch, num_nodes, k, node_features])


@pytest.mark.torch
def test_gather_nodes_values():
    """Test that _gather_nodes gathers the correct node feature vectors.

    Constructs a small, deterministic node tensor and verifies that each
    gathered entry corresponds to the node indexed by the neighbor index.
    """

    # batch=1, 3 nodes, node_features=2; each node has a distinct feature
    nodes = torch.tensor([[[1., 0.], [0., 1.], [1., 1.]]])  # (1, 3, 2)

    # For each node gather its two nearest neighbors
    neighbor_idx = torch.tensor([[[1, 2], [0, 2], [0, 1]]])  # (1, 3, 2)

    out = gather_nodes(nodes, neighbor_idx)  # (1, 3, 2, 2)

    # Node 0, neighbor at index 1 → nodes[0, 1] = [0, 1]
    assert torch.allclose(out[0, 0, 0], torch.tensor([0., 1.]))
    # Node 0, neighbor at index 2 → nodes[0, 2] = [1, 1]
    assert torch.allclose(out[0, 0, 1], torch.tensor([1., 1.]))
    # Node 1, neighbor at index 0 → nodes[0, 0] = [1, 0]
    assert torch.allclose(out[0, 1, 0], torch.tensor([1., 0.]))
    # Node 2, neighbor at index 1 → nodes[0, 1] = [0, 1]
    assert torch.allclose(out[0, 2, 1], torch.tensor([0., 1.]))


@pytest.mark.torch
def test_gather_nodes_batched():
    """Test that _gather_nodes handles batches independently.

    Verifies that node gathering is performed per-batch-element and does not
    bleed information across batch items.
    """

    batch, num_nodes, k, node_features = 3, 6, 2, 8
    nodes = torch.rand(batch, num_nodes, node_features)
    neighbor_idx = torch.randint(0, num_nodes, (batch, num_nodes, k))

    out = gather_nodes(nodes, neighbor_idx)

    assert out.shape == torch.Size([batch, num_nodes, k, node_features])
    # Spot-check: verify a specific batch element manually
    for b in range(batch):
        for n in range(num_nodes):
            for ki in range(k):
                idx = neighbor_idx[b, n, ki].item()
                assert torch.allclose(out[b, n, ki], nodes[b, idx])


@pytest.mark.torch
def test_cat_neighbors_nodes_output_shape():
    """Test that _cat_neighbors_nodes returns a tensor with the correct shape.

    The last dimension of the output should equal
    ``edge_features + node_features``, since the function concatenates
    gathered node features onto the existing neighbor edge features.
    """

    batch, num_nodes, k = 2, 5, 3
    node_features, edge_features = 32, 16

    h_nodes = torch.rand(batch, num_nodes, node_features)
    h_neighbors = torch.rand(batch, num_nodes, k, edge_features)
    E_idx = torch.randint(0, num_nodes, (batch, num_nodes, k))

    out = cat_neighbors_nodes(h_nodes, h_neighbors, E_idx)

    assert out.shape == torch.Size(
        [batch, num_nodes, k, edge_features + node_features])


@pytest.mark.torch
def test_cat_neighbors_nodes_values():
    """Test that _cat_neighbors_nodes produces the correct concatenated values.

    Constructs known node and edge feature tensors and verifies that the
    output is ``[h_neighbors, gathered_node_features]`` along the last dim.
    """

    # batch=1, 3 nodes, k=2, node_features=2, edge_features=3
    h_nodes = torch.tensor([[[1., 0.], [0., 1.], [1., 1.]]])  # (1, 3, 2)
    h_neighbors = torch.tensor([[[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
                                 [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]],
                                 [[1.3, 1.4, 1.5], [1.6, 1.7,
                                                    1.8]]]])  # (1, 3, 2, 3)
    E_idx = torch.tensor([[[1, 2], [0, 2], [0, 1]]])  # (1, 3, 2)

    out = cat_neighbors_nodes(h_nodes, h_neighbors, E_idx)  # (1, 3, 2, 5)

    assert out.shape == torch.Size([1, 3, 2, 5])

    # Node 0, neighbor slot 0: edge=[0.1,0.2,0.3], node[1]=[0,1]
    expected_0_0 = torch.tensor([0.1, 0.2, 0.3, 0., 1.])
    assert torch.allclose(out[0, 0, 0], expected_0_0)

    # Node 0, neighbor slot 1: edge=[0.4,0.5,0.6], node[2]=[1,1]
    expected_0_1 = torch.tensor([0.4, 0.5, 0.6, 1., 1.])
    assert torch.allclose(out[0, 0, 1], expected_0_1)

    # Node 1, neighbor slot 0: edge=[0.7,0.8,0.9], node[0]=[1,0]
    expected_1_0 = torch.tensor([0.7, 0.8, 0.9, 1., 0.])
    assert torch.allclose(out[0, 1, 0], expected_1_0)


@pytest.mark.torch
def test_cat_neighbors_nodes_self_loop():
    """Test _cat_neighbors_nodes when a node indexes itself as a neighbor.

    Ensures self-loop indices (where a node is its own neighbor) do not
    cause incorrect behavior.
    """

    batch, num_nodes, k = 1, 3, 1
    node_features, edge_features = 4, 4

    h_nodes = torch.rand(batch, num_nodes, node_features)
    h_neighbors = torch.rand(batch, num_nodes, k, edge_features)
    # Each node points to itself
    E_idx = torch.arange(num_nodes).view(1, num_nodes, 1).expand(batch, -1, k)

    out = cat_neighbors_nodes(h_nodes, h_neighbors, E_idx)

    assert out.shape == torch.Size(
        [batch, num_nodes, k, edge_features + node_features])
    for n in range(num_nodes):
        # Gathered node features should be the node's own features
        assert torch.allclose(out[0, n, 0, edge_features:], h_nodes[0, n])
