"""
DGL-based PAGTN for graph property prediction.
"""
import torch.nn as nn
import torch.nn.functional as F

from deepchem.models.losses import Loss, L2Loss, SparseSoftmaxCrossEntropy
from deepchem.models.torch_models.torch_model import TorchModel


class Pagtn(nn.Module):
  """Model for Graph Property Prediction

    Examples
    --------

    References
    ----------

    Notes
    -----
    """

  def __init__(self,
               n_tasks: int,
               number_atom_features: int = 94,
               number_bond_features: int = 42,
               mode: str = 'regression',
               n_classes: int = 2,
               ouput_node_features: int = 256,
               hidden_features: int = 32,
               num_layers: int = 5,
               num_heads: int = 1,
               dropout: float = 0.1,
               nfeat_name: str = 'x',
               efeat_name: str = 'edge_attr',
               pool_mode: str = 'sum'):
    """
        Parameters
        ----------
        """
    try:
      import dgl
    except:
      raise ImportError('This class requires dgl.')
    try:
      import dgllife
    except:
      raise ImportError('This class requires dgllife.')

    if mode not in ['classification', 'regression']:
      raise ValueError("mode must be either 'classification' or 'regression'")

    super(Pagtn, self).__init__()

    self.n_tasks = n_tasks
    self.mode = mode
    self.n_classes = n_classes
    self.nfeat_name = nfeat_name
    self.efeat_name = efeat_name
    if mode == 'classification':
      out_size = n_tasks * n_classes
    else:
      out_size = n_tasks

    from dgllife.model import PAGTNPredictor as DGLPAGTNPredictor

    self.model = DGLPAGTNPredictor(
        node_in_feats=number_atom_features,
        node_out_feats=ouput_node_features,
        node_hid_feats=hidden_features,
        edge_feats=number_bond_features,
        depth=num_layers,
        nheads=num_heads,
        dropout=dropout,
        n_tasks=out_size,
        mode=pool_mode)

  def forward(self, g):
    node_feats = g.ndata[self.nfeat_name]
    edge_feats = g.edata[self.efeat_name]
    out = self.model(g, node_feats, edge_feats)

    if self.mode == 'classification':
      if self.n_tasks == 1:
        logits = out.view(-1, self.n_classes)
        softmax_dim = 1
      else:
        logits = out.view(-1, self.n_tasks, self.n_classes)
        softmax_dim = 2
      proba = F.softmax(logits, dim=softmax_dim)
      return proba, logits
    else:
      return out


class PagtnModel(TorchModel):
  """Model for Graph Property Prediction.

    """

  def __init__(self,
               n_tasks: int,
               number_atom_features: int = 94,
               number_bond_features: int = 42,
               mode: str = 'regression',
               n_classes: int = 2,
               num_layers: int = 5,
               num_heads: int = 1,
               dropout: float = 0.1,
               pool_mode: str = 'sum',
               **kwargs):
    """
        """
    model = Pagtn(
        n_tasks=n_tasks,
        number_atom_features=number_atom_features,
        number_bond_features=number_bond_features,
        mode=mode,
        n_classes=n_classes,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        pool_mode=pool_mode)
    if mode == 'regression':
      loss: Loss = L2Loss()
      output_types = ['prediction']
    else:
      loss = SparseSoftmaxCrossEntropy()
      output_types = ['prediction', 'loss']
    super(PagtnModel, self).__init__(
        model, loss=loss, output_types=output_types, **kwargs)

  def _prepare_batch(self, batch):
    try:
      import dgl
    except:
      raise ImportError('This class requires dgl.')

    inputs, labels, weights = batch
    dgl_graphs = [
        graph.to_dgl_graph(self_loop=self._self_loop) for graph in inputs[0]
    ]
    inputs = dgl.batch(dgl_graphs).to(self.device)
    _, labels, weights = super(PagtnModel, self)._prepare_batch(([], labels,
                                                                 weights))
    return inputs, labels, weights
