import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F


class CGCNNLayer(nn.Module):
  """The convolutional layer of CGCNN.

  This class was implemented using DGLGraph methods.

  See: https://docs.dgl.ai/en/0.4.x/tutorials/models/1_gnn/9_gat.html
  """

  def __init__(self,
               hidden_node_dim: int,
               edge_dim: int,
               batch_norm: bool = True):
    """
    Parameters
    ----------
    hidden_node_dim: int
        The length of the hidden node feature vectors.
    edge_dim: int
        The length of the edge feature vectors.
    batch_norm: bool, default True
        Whether to apply batch normalization or not.
    """
    super(CGCNNLayer, self).__init__()
    z_dim = 2 * hidden_node_dim + edge_dim
    self.linear_with_sigmoid = nn.Linear(z_dim, hidden_node_dim)
    self.linear_with_softplus = nn.Linear(z_dim, hidden_node_dim)
    self.batch_norm = nn.BatchNorm1d(hidden_node_dim) if batch_norm else None

  def message_func(self, edges):
    z = torch.cat(
        [edges.src['x'], edges.dst['x'], edges.data['edge_attr']], dim=1)
    gated_z = F.sigmoid(self.linear_with_sigmoid(z))
    message_z = F.softplus(self.linear_with_softplus(z))
    return {'gated_z': gated_z, 'message_z': message_z}

  def reduce_func(self, nodes):
    new_h = nodes.data['x'] + torch.sum(
        nodes.mailbox['gated_z'] * nodes.mailbox['message_z'], dim=1)
    if self.batch_norm is not None:
      new_h = self.batch_norm(new_h)
    return {'x': new_h}

  def forward(self, dgl_graph):
    """Update node representaions.

    Parameters
    ----------
    dgl_graph : DGLGraph
        DGLGraph for a batch of graphs. The graph expects that the node features
        are stored in `ndata['x']`, and the edge features are stored in `edata['edge_attr']`.

    Returns
    -------
    dgl_graph : DGLGraph
        DGLGraph for a batch of updated graphs.
    """
    dgl_graph.update_all(self.message_func, self.reduce_func)
    return dgl_graph


class CGCNN(nn.Module):
  """Crystal Graph Convolutional Neural Network.

  This class implements Crystal Graph Convolutional Neural Network.
  Please confirm the detail algorithms from [1]_.

  References
  ----------
  .. [1] Xie, Tian, and Jeffrey C. Grossman. "Crystal graph convolutional neural networks
     for an accurate and interpretable prediction of material properties." Physical review letters
     120.14 (2018): 145301.
  """

  def __init__(
      self,
      in_node_dim: int = 92,
      hidden_node_dim: int = 64,
      in_edge_dim: int = 41,
      num_conv: int = 3,
      predicator_hidden_feats: int = 128,
      n_out: int = 1,
  ):
    """
    Parameters
    ----------
    in_node_dim : int, default 92
        The length of the initial node feature vectors.
    hidden_node_dim : int, default 64
        The length of the hidden node feature vectors.
    in_edge_dim : int, default 41
        The length of the initial edge feature vectors.
    num_conv: int, default 3
        The number of convolutional layers.
    predicator_hidden_feats: int, default 128
        Size of hidden graph representations in the predicator, default to 128.
    n_out: int
        Number of the output size, default to 1.
    """
    self.embedding = nn.Linear(in_node_dim, hidden_node_dim)
    self.convs = [
        CGCNNLayer(
            hidden_node_dim=hidden_node_dim,
            edge_dim=in_edge_dim,
            batch_norm=True) for _ in range(num_conv)
    ]
    self.fc = nn.Linear(hidden_node_dim, predicator_hidden_feats)
    self.out = nn.Linear(predicator_hidden_feats, n_out)

    def forward(self, dgl_graph):
      """Predict labels

      Parameters
      ----------
      dgl_graph : DGLGraph
          DGLGraph for a batch of graphs. The graph expects that the node features
          are stored in `ndata['x']`, and the edge features are stored in `edata['edge_attr']`.

      Returns
      -------
      out : torch.Tensor
          The output value
      """
      graph = dgl_graph
      for conv in self.convs:
        graph = conv(graph)

      # pooling
      graph_feat = dgl.sum_nodes(graph, 'x')
      graph_feat = self.fc(graph_feat)
      out = self.out(graph_feat)
      return out
