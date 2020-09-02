"""
This is a sample implementation for working PyTorch Geometric with DeepChem!
"""
import torch.nn as nn

from deepchem.models.torch_models.torch_model import TorchModel


class GAT(nn.Module):
  """Graph Attention Networks.

  This model takes arbitary graphs as an input, and predict graph properties. This model is
  one of variants of Graph Convolutional Networks. The main difference between basic GCN models
  is how to update node representations. The GAT uses multi head attention mechanisms which
  outbroke in NLP like Transformer when updating node representations. The most important advantage
  of this approach is that we can get the interpretability like how the model predict the value
  or which part of the graph structure is important from attention-weight. Please confirm
  the detail algorithms from [1]_.

  Examples
  --------
  >>> import deepchem as dc
  >>> from torch_geometric.data import Batch
  >>> smiles = ["C1CCC1", "C1=CC=CN=C1"]
  >>> featurizer = dc.feat.MolGraphConvFeaturizer()
  >>> graphs = featurizer.featurize(smiles)
  >>> print(type(graphs[0]))
  <class 'deepchem.feat.graph_data.GraphData'>
  >>> pyg_graphs = [graph.to_pyg_graph() for graph in graphs]
  >>> print(type(pyg_graphs[0]))
  <class 'torch_geometric.data.data.Data'>
  >>> model = dc.models.GAT(n_tasks=2)
  >>> out = model(Batch.from_data_list(pyg_graphs))
  >>> print(type(out))
  <class 'torch.Tensor'>
  >>> out.shape == (2, 2)
  True

  References
  ----------
  .. [1] Veličković, Petar, et al. "Graph attention networks." arXiv preprint
     arXiv:1710.10903 (2017).

  Notes
  -----
  This class requires PyTorch Geometric to be installed.
  """

  def __init__(
      self,
      in_node_dim: int = 39,
      hidden_node_dim: int = 64,
      heads: int = 4,
      dropout: float = 0.0,
      num_conv: int = 3,
      predictor_hidden_feats: int = 32,
      n_tasks: int = 1,
  ):
    """
    Parameters
    ----------
    in_node_dim: int, default 39
      The length of the initial node feature vectors. The 39 is
      based on `MolGraphConvFeaturizer`.
    hidden_node_dim: int, default 64
      The length of the hidden node feature vectors.
    heads: int, default 4
      The number of multi-head-attentions.
    dropout: float, default 0.0
      The dropout probability for each convolutional layer.
    num_conv: int, default 3
      The number of convolutional layers.
    predictor_hidden_feats: int, default 32
      The size for hidden representations in the output MLP predictor, default to 32.
    n_tasks: int, default 1
      The number of the output size, default to 1.
    """
    try:
      from torch_geometric.nn import GATConv, global_mean_pool
    except:
      raise ValueError("This class requires PyTorch Geometric to be installed.")
    super(GAT, self).__init__()
    self.embedding = nn.Linear(in_node_dim, hidden_node_dim)
    self.conv_layers = nn.ModuleList([
        GATConv(
            in_channels=hidden_node_dim,
            out_channels=hidden_node_dim,
            heads=heads,
            concat=False,
            dropout=dropout) for _ in range(num_conv)
    ])
    self.pooling = global_mean_pool
    self.fc = nn.Linear(hidden_node_dim, predictor_hidden_feats)
    self.out = nn.Linear(predictor_hidden_feats, n_tasks)

  def forward(self, data):
    """Predict labels

    Parameters
    ----------
    data: torch_geometric.data.Batch
      A mini-batch graph data for PyTorch Geometric models.

    Returns
    -------
    out: torch.Tensor
      The output value, the shape is `(batch_size, n_out)`.
    """
    node_feat, edge_index = data.x, data.edge_index
    node_feat = self.embedding(node_feat)

    # convolutional layer
    for conv in self.conv_layers:
      node_feat = conv(node_feat, edge_index)

    # pooling
    graph_feat = self.pooling(node_feat, data.batch)
    graph_feat = self.fc(graph_feat)
    out = self.out(graph_feat)
    return out


class GATModel(TorchModel):
  """Graph Attention Networks (GAT).

  Here is a simple example of code that uses the GATModel with
  molecules dataset.

  >> import deepchem as dc
  >> featurizer = dc.feat.MolGraphConvFeaturizer()
  >> tasks, datasets, transformers = dc.molnet.load_tox21(reload=False, featurizer=featurizer, transformers=[])
  >> train, valid, test = datasets
  >> model = dc.models.GATModel(loss=dc.models.losses.SoftmaxCrossEntropy(), batch_size=32, learning_rate=0.001)
  >> model.fit(train, nb_epoch=50)

  This model takes arbitary graphs as an input, and predict graph properties. This model is
  one of variants of Graph Convolutional Networks. The main difference between basic GCN models
  is how to update node representations. The GAT uses multi head attention mechanisms which
  outbroke in NLP like Transformer when updating node representations. The most important advantage
  of this approach is that we can get the interpretability like how the model predict the value
  or which part of the graph structure is important from attention-weight. Please confirm
  the detail algorithms from [1]_.

  References
  ----------
  .. [1] Veličković, Petar, et al. "Graph attention networks." arXiv preprint
     arXiv:1710.10903 (2017).

  Notes
  -----
  This class requires PyTorch Geometric to be installed.
  """

  def __init__(self,
               in_node_dim: int = 39,
               hidden_node_dim: int = 64,
               heads: int = 4,
               dropout: float = 0.0,
               num_conv: int = 3,
               predictor_hidden_feats: int = 32,
               n_tasks: int = 1,
               **kwargs):
    """
    This class accepts all the keyword arguments from TorchModel.

    Parameters
    ----------
    in_node_dim: int, default 39
      The length of the initial node feature vectors. The 39 is
      based on `MolGraphConvFeaturizer`.
    hidden_node_dim: int, default 64
      The length of the hidden node feature vectors.
    heads: int, default 4
      The number of multi-head-attentions.
    dropout: float, default 0.0
      The dropout probability for each convolutional layer.
    num_conv: int, default 3
      The number of convolutional layers.
    predictor_hidden_feats: int, default 32
      The size for hidden representations in the output MLP predictor, default to 32.
    n_tasks: int, default 1
      The number of the output size, default to 1.
    kwargs: Dict
      This class accepts all the keyword arguments from TorchModel.
    """
    model = GAT(
        in_node_dim,
        hidden_node_dim,
        heads,
        dropout,
        num_conv,
        predictor_hidden_feats,
        n_tasks,
    )
    super(GATModel, self).__init__(model, **kwargs)

  def _prepare_batch(self, batch):
    """Create batch data for GAT.

    Parameters
    ----------
    batch: Tuple
      The tuple are `(inputs, labels, weights)`.

    Returns
    -------
    inputs: torch_geometric.data.Batch
      A mini-batch graph data for PyTorch Geometric models.
    labels: List[torch.Tensor] or None
      The labels converted to torch.Tensor.
    weights: List[torch.Tensor] or None
      The weights for each sample or sample/task pair converted to torch.Tensor.
    """
    try:
      from torch_geometric.data import Batch
    except:
      raise ValueError("This class requires PyTorch Geometric to be installed.")

    inputs, labels, weights = batch
    pyg_graphs = [graph.to_pyg_graph() for graph in inputs[0]]
    inputs = Batch.from_data_list(pyg_graphs)
    _, labels, weights = super(GATModel, self)._prepare_batch(([], labels,
                                                               weights))
    return inputs, labels, weights
