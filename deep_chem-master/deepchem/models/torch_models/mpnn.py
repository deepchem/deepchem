"""
DGL-based MPNN for graph property prediction.
"""
import torch.nn as nn
import torch.nn.functional as F

from deepchem.models.losses import Loss, L2Loss, SparseSoftmaxCrossEntropy
from deepchem.models.torch_models.torch_model import TorchModel


class MPNN(nn.Module):
  """Model for Graph Property Prediction.

  This model proceeds as follows:

  * Combine latest node representations and edge features in updating node representations,
    which involves multiple rounds of message passing
  * For each graph, compute its representation by combining the representations
    of all nodes in it, which involves a Set2Set layer.
  * Perform the final prediction using an MLP

  Examples
  --------

  >>> import deepchem as dc
  >>> import dgl
  >>> from deepchem.models.torch_models import MPNN
  >>> smiles = ["C1CCC1", "C1=CC=CN=C1"]
  >>> featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
  >>> graphs = featurizer.featurize(smiles)
  >>> print(type(graphs[0]))
  <class 'deepchem.feat.graph_data.GraphData'>
  >>> dgl_graphs = [graphs[i].to_dgl_graph(self_loop=True) for i in range(len(graphs))]
  >>> # Batch two graphs into a graph of two connected components
  >>> batch_dgl_graph = dgl.batch(dgl_graphs)
  >>> model = MPNN(n_tasks=1, mode='regression')
  >>> preds = model(batch_dgl_graph)
  >>> print(type(preds))
  <class 'torch.Tensor'>
  >>> preds.shape == (2, 1)
  True

  References
  ----------
  .. [1] Justin Gilmer, Samuel S. Schoenholz, Patrick F. Riley, Oriol Vinyals, George E. Dahl.
         "Neural Message Passing for Quantum Chemistry." ICML 2017.

  Notes
  -----
  This class requires DGL (https://github.com/dmlc/dgl) and DGL-LifeSci
  (https://github.com/awslabs/dgl-lifesci) to be installed.
  """

  def __init__(self,
               n_tasks: int,
               node_out_feats: int = 64,
               edge_hidden_feats: int = 128,
               num_step_message_passing: int = 3,
               num_step_set2set: int = 6,
               num_layer_set2set: int = 3,
               mode: str = 'regression',
               number_atom_features: int = 30,
               number_bond_features: int = 11,
               n_classes: int = 2,
               nfeat_name: str = 'x',
               efeat_name: str = 'edge_attr'):
    """
    Parameters
    ----------
    n_tasks: int
      Number of tasks.
    node_out_feats: int
      The length of the final node representation vectors. Default to 64.
    edge_hidden_feats: int
      The length of the hidden edge representation vectors. Default to 128.
    num_step_message_passing: int
      The number of rounds of message passing. Default to 3.
    num_step_set2set: int
      The number of set2set steps. Default to 6.
    num_layer_set2set: int
      The number of set2set layers. Default to 3.
    mode: str
      The model type, 'classification' or 'regression'. Default to 'regression'.
    number_atom_features: int
      The length of the initial atom feature vectors. Default to 30.
    number_bond_features: int
      The length of the initial bond feature vectors. Default to 11.
    n_classes: int
      The number of classes to predict per task
      (only used when ``mode`` is 'classification'). Default to 2.
    nfeat_name: str
      For an input graph ``g``, the model assumes that it stores node features in
      ``g.ndata[nfeat_name]`` and will retrieve input node features from that.
      Default to 'x'.
    efeat_name: str
      For an input graph ``g``, the model assumes that it stores edge features in
      ``g.edata[efeat_name]`` and will retrieve input edge features from that.
      Default to 'edge_attr'.
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

    super(MPNN, self).__init__()

    self.n_tasks = n_tasks
    self.mode = mode
    self.n_classes = n_classes
    self.nfeat_name = nfeat_name
    self.efeat_name = efeat_name
    if mode == 'classification':
      out_size = n_tasks * n_classes
    else:
      out_size = n_tasks

    from dgllife.model import MPNNPredictor as DGLMPNNPredictor

    self.model = DGLMPNNPredictor(
        node_in_feats=number_atom_features,
        edge_in_feats=number_bond_features,
        node_out_feats=node_out_feats,
        edge_hidden_feats=edge_hidden_feats,
        n_tasks=out_size,
        num_step_message_passing=num_step_message_passing,
        num_step_set2set=num_step_set2set,
        num_layer_set2set=num_layer_set2set)

  def forward(self, g):
    """Predict graph labels

    Parameters
    ----------
    g: DGLGraph
      A DGLGraph for a batch of graphs. It stores the node features in
      ``dgl_graph.ndata[self.nfeat_name]`` and edge features in
      ``dgl_graph.edata[self.efeat_name]``.

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


class MPNNModel(TorchModel):
  """Model for graph property prediction

  This model proceeds as follows:

  * Combine latest node representations and edge features in updating node representations,
    which involves multiple rounds of message passing
  * For each graph, compute its representation by combining the representations
    of all nodes in it, which involves a Set2Set layer.
  * Perform the final prediction using an MLP

  Examples
  --------

  >>>
  >> import deepchem as dc
  >> from deepchem.models.torch_models import MPNNModel
  >> featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
  >> tasks, datasets, transformers = dc.molnet.load_tox21(
  ..     reload=False, featurizer=featurizer, transformers=[])
  >> train, valid, test = datasets
  >> model = MPNNModel(mode='classification', n_tasks=len(tasks),
  ..                   batch_size=32, learning_rate=0.001)
  >> model.fit(train, nb_epoch=50)

  References
  ----------
  .. [1] Justin Gilmer, Samuel S. Schoenholz, Patrick F. Riley, Oriol Vinyals, George E. Dahl.
         "Neural Message Passing for Quantum Chemistry." ICML 2017.

  Notes
  -----
  This class requires DGL (https://github.com/dmlc/dgl) and DGL-LifeSci
  (https://github.com/awslabs/dgl-lifesci) to be installed.
  """

  def __init__(self,
               n_tasks: int,
               node_out_feats: int = 64,
               edge_hidden_feats: int = 128,
               num_step_message_passing: int = 3,
               num_step_set2set: int = 6,
               num_layer_set2set: int = 3,
               mode: str = 'regression',
               number_atom_features: int = 30,
               number_bond_features: int = 11,
               n_classes: int = 2,
               self_loop: bool = False,
               **kwargs):
    """
    Parameters
    ----------
    n_tasks: int
      Number of tasks.
    node_out_feats: int
      The length of the final node representation vectors. Default to 64.
    edge_hidden_feats: int
      The length of the hidden edge representation vectors. Default to 128.
    num_step_message_passing: int
      The number of rounds of message passing. Default to 3.
    num_step_set2set: int
      The number of set2set steps. Default to 6.
    num_layer_set2set: int
      The number of set2set layers. Default to 3.
    mode: str
      The model type, 'classification' or 'regression'. Default to 'regression'.
    number_atom_features: int
      The length of the initial atom feature vectors. Default to 30.
    number_bond_features: int
      The length of the initial bond feature vectors. Default to 11.
    n_classes: int
      The number of classes to predict per task
      (only used when ``mode`` is 'classification'). Default to 2.
    self_loop: bool
      Whether to add self loops for the nodes, i.e. edges from nodes to themselves.
      Generally, an MPNNModel does not require self loops. Default to False.
    kwargs
      This can include any keyword argument of TorchModel.
    """
    model = MPNN(
        n_tasks=n_tasks,
        node_out_feats=node_out_feats,
        edge_hidden_feats=edge_hidden_feats,
        num_step_message_passing=num_step_message_passing,
        num_step_set2set=num_step_set2set,
        num_layer_set2set=num_layer_set2set,
        mode=mode,
        number_atom_features=number_atom_features,
        number_bond_features=number_bond_features,
        n_classes=n_classes)
    if mode == 'regression':
      loss: Loss = L2Loss()
      output_types = ['prediction']
    else:
      loss = SparseSoftmaxCrossEntropy()
      output_types = ['prediction', 'loss']
    super(MPNNModel, self).__init__(
        model, loss=loss, output_types=output_types, **kwargs)

    self._self_loop = self_loop

  def _prepare_batch(self, batch):
    """Create batch data for MPNN.

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
    _, labels, weights = super(MPNNModel, self)._prepare_batch(([], labels,
                                                                weights))
    return inputs, labels, weights
