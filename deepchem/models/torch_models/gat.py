"""
DGL-based GAT for graph property prediction.
"""
import torch.nn as nn
import torch.nn.functional as F

from deepchem.models.losses import Loss, L2Loss, SparseSoftmaxCrossEntropy
from deepchem.models.torch_models.torch_model import TorchModel


class GAT(nn.Module):
  """Model for Graph Property Prediction Based on Graph Attention Networks (GAT).

  This model proceeds as follows:

  * Update node representations in graphs with a variant of GAT
  * For each graph, compute its representation by 1) a weighted sum of the node
    representations in the graph, where the weights are computed by applying a
    gating function to the node representations 2) a max pooling of the node
    representations 3) concatenating the output of 1) and 2)
  * Perform the final prediction using an MLP

  Examples
  --------

  >>> import deepchem as dc
  >>> import dgl
  >>> from deepchem.models import GAT
  >>> smiles = ["C1CCC1", "C1=CC=CN=C1"]
  >>> featurizer = dc.feat.MolGraphConvFeaturizer()
  >>> graphs = featurizer.featurize(smiles)
  >>> print(type(graphs[0]))
  <class 'deepchem.feat.graph_data.GraphData'>
  >>> dgl_graphs = [graphs[i].to_dgl_graph(self_loop=True) for i in range(len(graphs))]
  >>> # Batch two graphs into a graph of two connected components
  >>> batch_dgl_graph = dgl.batch(dgl_graphs)
  >>> model = GAT(n_tasks=1, mode='regression')
  >>> preds = model(batch_dgl_graph)
  >>> print(type(preds))
  <class 'torch.Tensor'>
  >>> preds.shape == (2, 1)
  True

  References
  ----------
  .. [1] Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò,
         and Yoshua Bengio. "Graph Attention Networks." ICLR 2018.

  Notes
  -----
  This class requires DGL (https://github.com/dmlc/dgl) and DGL-LifeSci
  (https://github.com/awslabs/dgl-lifesci) to be installed.
  """

  def __init__(self,
               n_tasks: int,
               graph_attention_layers: list = None,
               n_attention_heads: int = 8,
               agg_modes: list = None,
               activation=F.elu,
               residual: bool = True,
               dropout: float = 0.,
               alpha: float = 0.2,
               predictor_hidden_feats: int = 128,
               predictor_dropout: float = 0.,
               mode: str = 'regression',
               number_atom_features: int = 30,
               n_classes: int = 2,
               nfeat_name: str = 'x'):
    """
    Parameters
    ----------
    n_tasks: int
      Number of tasks.
    graph_attention_layers: list of int
      Width of channels per attention head for GAT layers. graph_attention_layers[i]
      gives the width of channel for each attention head for the i-th GAT layer. If
      both ``graph_attention_layers`` and ``agg_modes`` are specified, they should have
      equal length. If not specified, the default value will be [8, 8].
    n_attention_heads: int
      Number of attention heads in each GAT layer.
    agg_modes: list of str
      The way to aggregate multi-head attention results for each GAT layer, which can be
      either 'flatten' for concatenating all-head results or 'mean' for averaging all-head
      results. ``agg_modes[i]`` gives the way to aggregate multi-head attention results for
      the i-th GAT layer. If both ``graph_attention_layers`` and ``agg_modes`` are
      specified, they should have equal length. If not specified, the model will flatten
      multi-head results for intermediate GAT layers and compute mean of multi-head results
      for the last GAT layer.
    activation: activation function or None
      The activation function to apply to the aggregated multi-head results for each GAT
      layer. If not specified, the default value will be ELU.
    residual: bool
      Whether to add a residual connection within each GAT layer. Default to True.
    dropout: float
      The dropout probability within each GAT layer. Default to 0.
    alpha: float
      A hyperparameter in LeakyReLU, which is the slope for negative values. Default to 0.2.
    predictor_hidden_feats: int
      The size for hidden representations in the output MLP predictor. Default to 128.
    predictor_dropout: float
      The dropout probability in the output MLP predictor. Default to 0.
    mode: str
      The model type, 'classification' or 'regression'. Default to 'regression'.
    number_atom_features: int
      The length of the initial atom feature vectors. Default to 30.
    n_classes: int
      The number of classes to predict per task
      (only used when ``mode`` is 'classification'). Default to 2.
    nfeat_name: str
      For an input graph ``g``, the model assumes that it stores node features in
      ``g.ndata[nfeat_name]`` and will retrieve input node features from that.
      Default to 'x'.
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

    super(GAT, self).__init__()

    self.n_tasks = n_tasks
    self.mode = mode
    self.n_classes = n_classes
    self.nfeat_name = nfeat_name
    if mode == 'classification':
      out_size = n_tasks * n_classes
    else:
      out_size = n_tasks

    from dgllife.model import GATPredictor as DGLGATPredictor

    if isinstance(graph_attention_layers, list) and isinstance(agg_modes, list):
      assert len(graph_attention_layers) == len(agg_modes), \
        'Expect graph_attention_layers and agg_modes to have equal length, ' \
        'got {:d} and {:d}'.format(len(graph_attention_layers), len(agg_modes))

    # Decide first number of GAT layers
    if graph_attention_layers is not None:
      num_gnn_layers = len(graph_attention_layers)
    elif agg_modes is not None:
      num_gnn_layers = len(agg_modes)
    else:
      num_gnn_layers = 2

    if graph_attention_layers is None:
      graph_attention_layers = [8] * num_gnn_layers
    if agg_modes is None:
      agg_modes = ['flatten' for _ in range(num_gnn_layers - 1)]
      agg_modes.append('mean')

    if activation is not None:
      activation = [activation] * num_gnn_layers

    self.model = DGLGATPredictor(
        in_feats=number_atom_features,
        hidden_feats=graph_attention_layers,
        num_heads=[n_attention_heads] * num_gnn_layers,
        feat_drops=[dropout] * num_gnn_layers,
        attn_drops=[dropout] * num_gnn_layers,
        alphas=[alpha] * num_gnn_layers,
        residuals=[residual] * num_gnn_layers,
        agg_modes=agg_modes,
        activations=activation,
        n_tasks=out_size,
        predictor_hidden_feats=predictor_hidden_feats,
        predictor_dropout=predictor_dropout)

  def forward(self, g):
    """Predict graph labels

    Parameters
    ----------
    g: DGLGraph
      A DGLGraph for a batch of graphs. It stores the node features in
      ``dgl_graph.ndata[self.nfeat_name]``.

    Returns
    -------
    torch.Tensor
      The model output.

      * When self.mode = 'regression',
        its shape will be ``(dgl_graph.batch_size, self.n_tasks)``.
      * When self.mode = 'classification', the output consists of probabilities
        for classes. Its shape will be
        ``(dgl_graph.batch_size, self.n_tasks, self.n_classes)`` if self.n_tasks > 1;
        its shape will be ``(dgl_graph.batch_size, self.n_classes)`` if self.n_tasks is 1.
    torch.Tensor, optional
      This is only returned when self.mode = 'classification', the output consists of the
      logits for classes before softmax.
    """
    node_feats = g.ndata[self.nfeat_name]
    out = self.model(g, node_feats)

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


class GATModel(TorchModel):
  """Model for Graph Property Prediction Based on Graph Attention Networks (GAT).

  This model proceeds as follows:

  * Update node representations in graphs with a variant of GAT
  * For each graph, compute its representation by 1) a weighted sum of the node
    representations in the graph, where the weights are computed by applying a
    gating function to the node representations 2) a max pooling of the node
    representations 3) concatenating the output of 1) and 2)
  * Perform the final prediction using an MLP

  Examples
  --------

  >>>
  >> import deepchem as dc
  >> from deepchem.models import GATModel
  >> featurizer = dc.feat.MolGraphConvFeaturizer()
  >> tasks, datasets, transformers = dc.molnet.load_tox21(
  ..     reload=False, featurizer=featurizer, transformers=[])
  >> train, valid, test = datasets
  >> model = GATModel(mode='classification', n_tasks=len(tasks),
  ..                  batch_size=32, learning_rate=0.001)
  >> model.fit(train, nb_epoch=50)

  References
  ----------
  .. [1] Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò,
         and Yoshua Bengio. "Graph Attention Networks." ICLR 2018.

  Notes
  -----
  This class requires DGL (https://github.com/dmlc/dgl) and DGL-LifeSci
  (https://github.com/awslabs/dgl-lifesci) to be installed.
  """

  def __init__(self,
               n_tasks: int,
               graph_attention_layers: list = None,
               n_attention_heads: int = 8,
               agg_modes: list = None,
               activation=F.elu,
               residual: bool = True,
               dropout: float = 0.,
               alpha: float = 0.2,
               predictor_hidden_feats: int = 128,
               predictor_dropout: float = 0.,
               mode: str = 'regression',
               number_atom_features: int = 30,
               n_classes: int = 2,
               self_loop: bool = True,
               **kwargs):
    """
    Parameters
    ----------
    n_tasks: int
      Number of tasks.
    graph_attention_layers: list of int
      Width of channels per attention head for GAT layers. graph_attention_layers[i]
      gives the width of channel for each attention head for the i-th GAT layer. If
      both ``graph_attention_layers`` and ``agg_modes`` are specified, they should have
      equal length. If not specified, the default value will be [8, 8].
    n_attention_heads: int
      Number of attention heads in each GAT layer.
    agg_modes: list of str
      The way to aggregate multi-head attention results for each GAT layer, which can be
      either 'flatten' for concatenating all-head results or 'mean' for averaging all-head
      results. ``agg_modes[i]`` gives the way to aggregate multi-head attention results for
      the i-th GAT layer. If both ``graph_attention_layers`` and ``agg_modes`` are
      specified, they should have equal length. If not specified, the model will flatten
      multi-head results for intermediate GAT layers and compute mean of multi-head results
      for the last GAT layer.
    activation: activation function or None
      The activation function to apply to the aggregated multi-head results for each GAT
      layer. If not specified, the default value will be ELU.
    residual: bool
      Whether to add a residual connection within each GAT layer. Default to True.
    dropout: float
      The dropout probability within each GAT layer. Default to 0.
    alpha: float
      A hyperparameter in LeakyReLU, which is the slope for negative values. Default to 0.2.
    predictor_hidden_feats: int
      The size for hidden representations in the output MLP predictor. Default to 128.
    predictor_dropout: float
      The dropout probability in the output MLP predictor. Default to 0.
    mode: str
      The model type, 'classification' or 'regression'. Default to 'regression'.
    number_atom_features: int
      The length of the initial atom feature vectors. Default to 30.
    n_classes: int
      The number of classes to predict per task
      (only used when ``mode`` is 'classification'). Default to 2.
    self_loop: bool
      Whether to add self loops for the nodes, i.e. edges from nodes to themselves.
      When input graphs have isolated nodes, self loops allow preserving the original feature
      of them in message passing. Default to True.
    kwargs
      This can include any keyword argument of TorchModel.
    """
    model = GAT(
        n_tasks=n_tasks,
        graph_attention_layers=graph_attention_layers,
        n_attention_heads=n_attention_heads,
        agg_modes=agg_modes,
        activation=activation,
        residual=residual,
        dropout=dropout,
        alpha=alpha,
        predictor_hidden_feats=predictor_hidden_feats,
        predictor_dropout=predictor_dropout,
        mode=mode,
        number_atom_features=number_atom_features,
        n_classes=n_classes)
    if mode == 'regression':
      loss: Loss = L2Loss()
      output_types = ['prediction']
    else:
      loss = SparseSoftmaxCrossEntropy()
      output_types = ['prediction', 'loss']
    super(GATModel, self).__init__(
        model, loss=loss, output_types=output_types, **kwargs)

    self._self_loop = self_loop

  def _prepare_batch(self, batch):
    """Create batch data for GAT.

    Parameters
    ----------
    batch: tuple
      The tuple is ``(inputs, labels, weights)``.

    Returns
    -------
    inputs: DGLGraph
      DGLGraph for a batch of graphs.
    labels: list of torch.Tensor or None
      The graph labels.
    weights: list of torch.Tensor or None
      The weights for each sample or sample/task pair converted to torch.Tensor.
    """
    try:
      import dgl
    except:
      raise ImportError('This class requires dgl.')

    inputs, labels, weights = batch
    dgl_graphs = [
        graph.to_dgl_graph(self_loop=self._self_loop) for graph in inputs[0]
    ]
    inputs = dgl.batch(dgl_graphs).to(self.device)
    _, labels, weights = super(GATModel, self)._prepare_batch(([], labels,
                                                               weights))
    return inputs, labels, weights
