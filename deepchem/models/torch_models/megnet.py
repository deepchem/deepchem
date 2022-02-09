"""
Implementation of MEGNet class
"""
from typing import Tuple
import torch
import torch.nn as nn
from torch import Tensor

from deepchem.models.torch_models.layers import GraphNetwork as GN


class MEGNet(nn.Module):
  """MatErials Graph Network

  A model for predicting crystal and molecular properties using GraphNetworks.

  Example
  -------
  >>> import torch
  >>> from torch_geometric.data import Data as GraphData
  >>> from deepchem.models.torch_models import MEGNet
  >>> num_nodes, num_node_features = 5, 10
  >>> num_edges, num_edge_attrs = 5, 2
  >>> num_global_features = 4
  >>> node_features = torch.randn(num_nodes, num_node_features)
  >>> edge_attrs = torch.randn(num_edges, num_edge_attrs)
  >>> edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]]).long()
  >>> global_features = torch.randn(num_global_features)
  >>> graph = GraphData(node_features, edge_index, edge_attrs, global_features=global_features)
  >>> model = MEGNet(n_node_features=num_node_features, n_edge_features=num_edge_attrs, n_global_features=num_global_features)
  >>> pred = model(graph.x, graph.edge_index, graph.edge_attr, graph.global_features)

  Note
  ----
  This class requires torch-geometric to be installed.
  """

  def __init__(self,
               n_node_features=32,
               n_edge_features=32,
               n_global_features=32,
               n_blocks=1,
               mode='regression',
               n_classes=2,
               n_tasks=1):
    """

    Parameters
    ----------
    n_tasks: int, default 1
      The number of output size
    mode: str, default 'regression'
      The model type - classification or regression
    n_classes: int, default 2
      The number of classes to predict when used in classification mode.
    """
    super(MEGNet, self).__init__()
    try:
      from torch_geometric.nn import Set2Set
    except ModuleNotFoundError:
      raise ImportError("MEGNet model requires torch_geometric to be installed")

    if mode not in ['classification', 'regression']:
      raise ValueError("mode must be either 'classification' or 'regression'")
    self.n_node_features = n_node_features
    self.n_edge_features = n_edge_features
    self.n_global_features = n_global_features
    self.megnet_blocks = nn.ModuleList()
    self.n_blocks = n_blocks
    for i in range(n_blocks):
      self.megnet_blocks.append(
          GN(n_node_features=n_node_features,
             n_edge_features=n_edge_features,
             n_global_features=n_global_features))
    self.n_tasks = n_tasks
    self.mode = mode
    self.n_classes = n_classes

    self.set2set_nodes = Set2Set(
        in_channels=n_node_features, processing_steps=3, num_layers=1)
    self.set2set_edges = Set2Set(
        in_channels=n_edge_features, processing_steps=3, num_layers=1)

    self.dense = nn.Sequential(
        nn.Linear(
            in_features=2 * n_node_features + 2 * n_edge_features +
            n_global_features,
            out_features=32), nn.Linear(in_features=32, out_features=16))

    if self.mode == 'regression':
      self.out = nn.Linear(in_features=16, out_features=n_tasks)
    elif self.mode == 'classification':
      self.out = nn.Linear(in_features=16, out_features=n_tasks * n_classes)

  def forward(self, node_features: Tensor, edge_index: Tensor,
              edge_features: Tensor,
              global_features: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Parameters
    ----------
    node_features: torch.Tensor
      Input node features of shape :math:`(|\mathcal{V}|, F_n)`
    edge_index: torch.Tensor
      Edge indexes of shape :math:`(2, |\mathcal{E}|)`
    edge_features: torch.Tensor
      Edge features of the graph, shape: :math:`(|\mathcal{E}|, F_e)`
    global_features: torch.Tensor
      Global features of the graph, shape: :math:`(F_g, 1)`
    where :math:`|\mathcal{V}|` and :math:`|\mathcal{E}|` denotes the number of nodes and edges in the graph,
      :math:F_n, :math:F_e, :math:F_g denotes the number of node features, edge features and global state features respectively.

    Returns
    -------
    torch.Tensor: Predictions for the graph
    """
    for i in range(self.n_blocks):
      node_features, edge_features, global_features = self.megnet_blocks[i](
          node_features, edge_index, edge_features, global_features)
    node_features = self.set2set_nodes(
        node_features,
        batch=node_features.new_zeros(node_features.size(0), dtype=torch.int64))
    edge_features = self.set2set_edges(
        edge_features,
        batch=edge_features.new_zeros(edge_features.size(0), dtype=torch.int64))
    out = torch.cat(
        [node_features.squeeze(0),
         edge_features.squeeze(0), global_features],
        axis=0)
    out = self.dense(out)
    return self.out(out)
