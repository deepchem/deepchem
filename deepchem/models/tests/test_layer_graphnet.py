import pytest
try:
  import torch
  from deepchem.models.torch_models.layers import GraphNetwork
  has_torch = True
except ModuleNotFoundError:
  has_torch = False


@pytest.mark.torch
def test_graphnet_layer():
  node_features = torch.randn(5, 10)
  edge_features = torch.randn(5, 3)
  global_features = torch.randn(4)
  edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]]).long()

  graphnet = GraphNetwork(
      n_node_features=node_features.size(1),
      n_edge_features=edge_features.size(1),
      n_global_features=global_features.size(0))

  assert repr(
      graphnet
  ) == 'GraphNetwork(n_node_features=10, n_edge_features=3, n_global_features=4, residual_connection=True)'

  new_node_features, new_edge_features, new_global_features = graphnet(
      node_features, edge_index, edge_features, global_features)

  assert node_features.size() == new_node_features.size()
  assert edge_features.size() == new_edge_features.size()
  assert global_features.size() == new_global_features.size()

  node_features = torch.tensor([[0.7, 0.7], [0.7, 0.7]]).float()
  edge_features = torch.tensor([[0.7, 0.7]]).float()
  global_features = torch.tensor([1]).float()
  edge_index = torch.tensor([[0], [1]]).long()

  torch.manual_seed(12345)
  graphnet1 = GraphNetwork(
      n_node_features=2, n_edge_features=2, n_global_features=1)
  out_node1, out_edge1, out_global1 = graphnet1(node_features, edge_index,
                                                edge_features, global_features)

  torch.manual_seed(12345)
  graphnet2 = GraphNetwork(
      n_node_features=2, n_edge_features=2, n_global_features=1)
  out_node2, out_edge2, out_global2 = graphnet2(node_features, edge_index,
                                                edge_features, global_features)

  rtol = 1e-5
  atol = 1e-6
  assert torch.allclose(out_node1, out_node2, rtol=rtol, atol=atol)
  assert torch.allclose(out_edge1, out_edge2, rtol=rtol, atol=atol)
  assert torch.allclose(out_global1, out_global2, rtol=rtol, atol=atol)
