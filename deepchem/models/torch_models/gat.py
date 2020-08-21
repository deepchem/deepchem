"""
This is a sample implementation for working PyTorch Geometric with DeepChem!
"""
import torch.nn as nn

from deepchem.models.torch_models.torch_model import TorchModel


class GAT(nn.Module):
  """Graph Attention Networks.

  TODO: add more docstring

  Examples
  --------
  >>> import deepchem as dc
  >>> smiles = ["C1CCC1", "C1=CC=CN=C1"]
  >>> featurizer = dc.feat.MolGraphConvFeaturizer()
  >>> graphs = featurizer.featurize(smiles)
  >>> print(type(graphs[0]))
  <class 'deepchem.feat.graph_data.GraphData'>
  >>> pyg_graphs = [graph.to_pyg_graph() for graph in graphs]
  >>> print(type(pyg_graphs[0]))
  >>> model = dc.models.GAT(n_out=1)
  >>> out = model(pyg_graphs)
  >>> print(type(out))
  <class 'torch.Tensor'>
  >>> out.shape == (1, 1)
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
      in_node_dim: int = 25,
      hidden_node_dim: int = 64,
      heads: int = 4,
      dropout_rate: float = 0.0,
      num_conv: int = 3,
      predicator_hidden_feats: int = 32,
      n_tasks: int = 1,
  ):
    """
    TODO: add docstring
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
            dropout=dropout_rate) for _ in range(num_conv)
    ])
    self.pooling = global_mean_pool
    self.fc = nn.Linear(hidden_node_dim, predicator_hidden_feats)
    self.out = nn.Linear(predicator_hidden_feats, n_tasks)

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
  """Graph Attention Networks.

  TODO: add more docstring

  Here is a simple example of code that uses the GATModel with
  molecules dataset.

  >> import deepchem as dc
  >> dataset_config = {"reload": False, "featurizer": dc.feat.MolGraphConvFeaturizer, "transformers": []}
  >> tasks, datasets, transformers = dc.molnet.load_tox21(**dataset_config)
  >> train, valid, test = datasets
  >> model = dc.models.GATModel(loss=dc.models.losses.(), batch_size=32, learning_rate=0.001)
  >> model.fit(train, nb_epoch=50)

  References
  ----------
  .. [1] Veličković, Petar, et al. "Graph attention networks." arXiv preprint
     arXiv:1710.10903 (2017).

  Notes
  -----
  This class requires PyTorch Geometric to be installed.
  """

  def __init__(self,
               in_node_dim: int = 25,
               hidden_node_dim: int = 64,
               heads: int = 4,
               dropout_rate: float = 0.0,
               num_conv: int = 3,
               predicator_hidden_feats: int = 32,
               n_tasks: int = 1,
               **kwargs):
    """
    TODO: add docstring
    """
    model = GAT(
        in_node_dim,
        hidden_node_dim,
        heads,
        dropout_rate,
        num_conv,
        predicator_hidden_feats,
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

    Notes
    -----
    This class requires PyTorch Geometric to be installed.
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
