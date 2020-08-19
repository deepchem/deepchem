import torch
import torch.nn as nn
import torch.nn.functional as F

from deepchem.models.torch_models.torch_model import TorchModel


class CGCNNLayer(nn.Module):
  """The convolutional layer of CGCNN.

  This class was implemented using DGLGraph methods.
  Please confirm how to use DGLGraph methods from below link.
  See: https://docs.dgl.ai/en/0.4.x/tutorials/models/1_gnn/9_gat.html

  Examples
  --------
  >>> import deepchem as dc
  >>> import pymatgen as mg
  >>> lattice = mg.Lattice.cubic(4.2)
  >>> structure = mg.Structure(lattice, ["Cs", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
  >>> featurizer = dc.feat.CGCNNFeaturizer()
  >>> cgcnn_graph = featurizer.featurize([structure])[0]
  >>> cgcnn_graph.num_node_features
  92
  >>> cgcnn_graph.num_edge_features
  41
  >>> cgcnn_dgl_graph = cgcnn_graph.to_dgl_graph()
  >>> print(type(cgcnn_dgl_graph))
  <class 'dgl.graph.DGLGraph'>
  >>> layer = CGCNNLayer(hidden_node_dim=92, edge_dim=41)
  >>> update_graph = layer(cgcnn_dgl_graph)
  >>> print(type(update_graph))
  <class 'dgl.graph.DGLGraph'>

  Notes
  -----
  This class requires DGL and PyTorch to be installed.
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
    return {'x': new_h}

  def forward(self, dgl_graph):
    """Update node representaions.

    Parameters
    ----------
    dgl_graph: DGLGraph
      DGLGraph for a batch of graphs. The graph expects that the node features
      are stored in `ndata['x']`, and the edge features are stored in `edata['edge_attr']`.

    Returns
    -------
    dgl_graph: DGLGraph
      DGLGraph for a batch of updated graphs.
    """
    dgl_graph.update_all(self.message_func, self.reduce_func)
    if self.batch_norm is not None:
      dgl_graph.ndata['x'] = self.batch_norm(dgl_graph.ndata['x'])
    return dgl_graph


class CGCNN(nn.Module):
  """Crystal Graph Convolutional Neural Network (CGCNN).

  This model takes arbitary crystal structures as an input, and predict material properties
  using the element information and connection of atoms in the crystal. If you want to get
  some material properties which has a high computational cost like band gap in the case
  of DFT, this model may be useful. This model is one of variants of Graph Convolutional
  Networks. The main differences between other GCN models are how to construct graphs and
  how to update node representations. This model defines the crystal graph from structures
  using distances between atoms. The crystal graph is an undirected multigraph which is defined
  by nodes representing atom properties and edges representing connections between atoms
  in a crystal. And, this model updates the node representations using both neighbor node
  and edge representations. Please confirm the detail algorithms from [1]_.

  Examples
  --------
  >>> import deepchem as dc
  >>> import pymatgen as mg
  >>> lattice = mg.Lattice.cubic(4.2)
  >>> structure = mg.Structure(lattice, ["Cs", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
  >>> featurizer = dc.feat.CGCNNFeaturizer()
  >>> cgcnn_feat = featurizer.featurize([structure])[0]
  >>> print(type(cgcnn_feat))
  <class 'deepchem.feat.graph_data.GraphData'>
  >>> cgcnn_dgl_feat = cgcnn_feat.to_dgl_graph()
  >>> print(type(cgcnn_dgl_feat))
  <class 'dgl.graph.DGLGraph'>
  >>> model = dc.models.CGCNN(n_tasks=2)
  >>> out = model(cgcnn_dgl_feat)
  >>> print(type(out))
  <class 'torch.Tensor'>
  >>> out.shape == (1, 2)
  True

  References
  ----------
  .. [1] Xie, Tian, and Jeffrey C. Grossman. "Crystal graph convolutional neural networks
     for an accurate and interpretable prediction of material properties." Physical review letters
     120.14 (2018): 145301.

  Notes
  -----
  This class requires DGL and PyTorch to be installed.
  """

  def __init__(
      self,
      in_node_dim: int = 92,
      hidden_node_dim: int = 64,
      in_edge_dim: int = 41,
      num_conv: int = 3,
      predicator_hidden_feats: int = 128,
      n_tasks: int = 1,
  ):
    """
    Parameters
    ----------
    in_node_dim: int, default 92
      The length of the initial node feature vectors. The 92 is
      based on length of vectors in the atom_init.json.
    hidden_node_dim: int, default 64
      The length of the hidden node feature vectors.
    in_edge_dim: int, default 41
      The length of the initial edge feature vectors. The 41 is
      based on default setting of CGCNNFeaturizer.
    num_conv: int, default 3
      The number of convolutional layers.
    predicator_hidden_feats: int, default 128
      Size for hidden representations in the output MLP predictor, default to 128.
    n_tasks: int, default 1
      Number of the output size, default to 1.
    """
    super(CGCNN, self).__init__()
    self.embedding = nn.Linear(in_node_dim, hidden_node_dim)
    self.conv_layers = nn.ModuleList([
        CGCNNLayer(
            hidden_node_dim=hidden_node_dim,
            edge_dim=in_edge_dim,
            batch_norm=True) for _ in range(num_conv)
    ])
    self.fc = nn.Linear(hidden_node_dim, predicator_hidden_feats)
    self.out = nn.Linear(predicator_hidden_feats, n_tasks)

  def forward(self, dgl_graph):
    """Predict labels

    Parameters
    ----------
    dgl_graph: DGLGraph
      DGLGraph for a batch of graphs. The graph expects that the node features
      are stored in `ndata['x']`, and the edge features are stored in `edata['edge_attr']`.

    Returns
    -------
    out: torch.Tensor
      The output value, the shape is `(batch_size, n_tasks)`.
    """
    try:
      import dgl
    except:
      raise ValueError("This class requires DGL to be installed.")

    graph = dgl_graph
    # embedding node features
    graph.ndata['x'] = self.embedding(graph.ndata['x'])

    # convolutional layer
    for conv in self.conv_layers:
      graph = conv(graph)

    # pooling
    graph_feat = dgl.mean_nodes(graph, 'x')
    graph_feat = self.fc(graph_feat)
    out = self.out(graph_feat)
    return out


class CGCNNModel(TorchModel):
  """Crystal Graph Convolutional Neural Network (CGCNN).

  Here is a simple example of code that uses the CGCNNModel with
  materials dataset.

  >> import deepchem as dc
  >> dataset_config = {"reload": False, "featurizer": dc.feat.CGCNNFeaturizer, "transformers": []}
  >> tasks, datasets, transformers = dc.molnet.load_perovskite(**dataset_config)
  >> train, valid, test = datasets
  >> model = dc.models.CGCNNModel(loss=dc.models.losses.L2Loss(), batch_size=32, learning_rate=0.001)
  >> model.fit(train, nb_epoch=50)

  This model takes arbitary crystal structures as an input, and predict material properties
  using the element information and connection of atoms in the crystal. If you want to get
  some material properties which has a high computational cost like band gap in the case
  of DFT, this model may be useful. This model is one of variants of Graph Convolutional
  Networks. The main differences between other GCN models are how to construct graphs and
  how to update node representations. This model defines the crystal graph from structures
  using distances between atoms. The crystal graph is an undirected multigraph which is defined
  by nodes representing atom properties and edges representing connections between atoms
  in a crystal. And, this model updates the node representations using both neighbor node
  and edge representations. Please confirm the detail algorithms from [1]_.

  References
  ----------
  .. [1] Xie, Tian, and Jeffrey C. Grossman. "Crystal graph convolutional neural networks
     for an accurate and interpretable prediction of material properties." Physical review letters
     120.14 (2018): 145301.

  Notes
  -----
  This class requires DGL and PyTorch to be installed.
  """

  def __init__(self,
               in_node_dim: int = 92,
               hidden_node_dim: int = 64,
               in_edge_dim: int = 41,
               num_conv: int = 3,
               predicator_hidden_feats: int = 128,
               n_tasks: int = 1,
               **kwargs):
    """
    This class accepts all the keyword arguments from TorchModel.

    Parameters
    ----------
    in_node_dim: int, default 92
      The length of the initial node feature vectors. The 92 is
      based on length of vectors in the atom_init.json.
    hidden_node_dim: int, default 64
      The length of the hidden node feature vectors.
    in_edge_dim: int, default 41
      The length of the initial edge feature vectors. The 41 is
      based on default setting of CGCNNFeaturizer.
    num_conv: int, default 3
      The number of convolutional layers.
    predicator_hidden_feats: int, default 128
      Size for hidden representations in the output MLP predictor, default to 128.
    n_tasks: int, default 1
      Number of the output size, default to 1.
    kwargs: Dict
      This class accepts all the keyword arguments from TorchModel.
    """
    model = CGCNN(in_node_dim, hidden_node_dim, in_edge_dim, num_conv,
                  predicator_hidden_feats, n_tasks)
    super(CGCNNModel, self).__init__(model, **kwargs)

  def _prepare_batch(self, batch):
    """Create batch data for CGCNN.

    Parameters
    ----------
    batch: Tuple
      The tuple are `(inputs, labels, weights)`.

    Returns
    -------
    inputs: DGLGraph
      DGLGraph for a batch of graphs.
    labels: List[torch.Tensor] or None
      The labels converted to torch.Tensor
    weights: List[torch.Tensor] or None
      The weights for each sample or sample/task pair converted to torch.Tensor

    Notes
    -----
    This class requires DGL and PyTorch to be installed.
    """
    try:
      import dgl
    except:
      raise ValueError("This class requires DGL to be installed.")

    inputs, labels, weights = batch
    dgl_graphs = [graph.to_dgl_graph() for graph in inputs[0]]
    inputs = dgl.batch(dgl_graphs).to(self.device)
    _, labels, weights = super(CGCNNModel, self)._prepare_batch(([], labels,
                                                                 weights))
    return inputs, labels, weights
