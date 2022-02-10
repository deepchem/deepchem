"""
Implementation of MEGNet class
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from deepchem.models.losses import Loss, L2Loss, SparseSoftmaxCrossEntropy
from deepchem.models.torch_models.layers import GraphNetwork as GN
from deepchem.models.torch_models import TorchModel


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


  def forward(self, pyg_batch: PyGBatch):
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
    node_batch_map: torch.LongTensor (optional)
      A vector that maps each node to its respective graph identifier.
    edge_batch_map: torch.LongTensor (optional)
      A vector that maps each edge to its respective graph identifier.

    Returns
    -------
    torch.Tensor: Predictions for the graph
    """
    if node_batch_map is None:
      node_batch_map = node_features.new_zeros(node_features.size(0), dtype=torch.int64)
    if edge_batch_map is None:
      edge_batch_map = edge_features.new_zeros(edge_features.size(0), dtype=torch.int64)

    for i in range(self.n_blocks):
      node_features, edge_features, global_features = self.megnet_blocks[i](
          node_features, edge_index, edge_features, global_features, node_batch_map, edge_batch_map)

    node_features = self.set2set_nodes(node_features, node_batch_map) 
    edge_features = self.set2set_edges(edge_features, edge_batch_map) 
    out = torch.cat([node_features, edge_features, global_features], axis=1)
    out = self.out(self.dense(out))

    if self.mode == 'classification':
      if self.n_tasks == 1:
        logits = out.view(-1, self.n_classes)
        softmax_dim = 1
      else:
        logits = out.view(-1, self.n_tasks, self.n_classes)
        softmax_dim = 2
      proba = F.softmax(logits, dim=softmax_dim)
      return proba, logits
    elif self.mode == 'regression':
      return out


class MEGNetModel(TorchModel):
  """MEGNet Model

  """
  def __init__(self,
               n_node_features=32,
               n_edge_features=32,
               n_global_features=32,
               n_blocks=1,
               mode='regression',
               n_classes=2,
               n_tasks=1):

    model = MEGNet(
        n_node_features=n_node_features,
        n_edge_features=n_edge_features,
        n_global_features=n_global_features,
        n_blocks=n_blocks,
        mode=mode,
        n_classes=n_classes,
        n_tasks=n_tasks)
    if mode == 'regression':
      loss: Loss = L2Loss()
      output_types = ['prediction']
    elif mode == 'classification':
      loss = SparseSoftmaxCrossEntropy()
      output_types = ['prediction', 'loss']
    super(MEGNetModel, self).__init__(
        model, loss=loss, output_types=output_types, **kwargs)

  def _prepare_batch(self, batch):
    """Creates batch data for MEGNet model

    Note
    ----
    Ideally, we should only override default_generator method. But the problem
    here is that we _prepare_batch of TorchModel only supports non-graph
    data types. Hence, we are overriding it here. This should be fixed
    some time in the future.
    """
    try:
      from torch_geometric.data import Batch
    except ModuleNotFoundError:
      raise ImportError("This module requires PyTorch Geometric")

    # We convert deepchem.feat.GraphData to a PyG graph and then
    # batch it.
    graphs, labels, weights = batch
    # The default_generator method returns an array of dc.feat.GraphData objects
    # nested inside a list. To access the nested array of graphs, we are
    # indexing by 0 here.
    graph_list = [graph.to_pyg_graph() for graph in graphs[0]]
    pyg_batch = Batch()
    pyg_batch = pyg_batch.from_data_list(graph_list)

    _, labels, weights = super(MEGNetModel, self)._prepare_batch(([], labels, weights))

    return pyg_batch, labels, weights
