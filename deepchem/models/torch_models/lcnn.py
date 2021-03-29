import torch
import torch.nn as nn
from deepchem.models.torch_models.torch_model import TorchModel
from deepchem.models.losses import L2Loss


class LCNNBlock(nn.Module):
  """
  The Lattice Convolution layer of LCNN

  The following class implements the lattice convolution function which is
  based on graph convolution networks where,
  [1] Each atom is represented as a node
  [2] Adjacent atom based on distance are considered as neighbors.

  Operations in Lattice Convolution:

  [1] In graph aggregation step- node features of neighbors are concatenated and
  into a linear layer. But since diffrent permutation of order of neighbors could
  be considered because of which , diffrent permutation of the lattice
  structure are considered in diffrent symmetrical angles (0 , 60 ,120 180 , 240 , 300 )

  [2] After the linear layer on each permutations, they are added up for each node and
  each node is transformed into a vector.

  Examples
  --------
  >>> import deepchem as dc
  >>> from deepchem.models.torch_models.lcnn import LCNNBlock
  >>> from deepchem.feat.graph_data import GraphData
  >>> import numpy as np
  >>> nodes = np.array([0, 1, 2])
  >>> x = np.zeros((nodes.size, nodes.max()+1))
  >>> x[np.arange(nodes.size),nodes] = 1
  >>> v = np.array([ 0,0, 0,0, 1,1, 1,1, 2,2, 2,2 ])
  >>> u = np.array([ 1,2, 2,1, 2,0, 0,2, 1,0, 0,1 ])
  >>> graph = GraphData(node_features=x, edge_index=np.array([u, v]))
  >>> model = LCNNBlock(3*2, 3, 2)
  >>> G = graph.to_dgl_graph()
  >>> x = G.ndata.pop('x')
  >>> print(model(G, x).shape)
  torch.Size([3, 3])

  """

  def __init__(self,
               input_feature: int,
               output_feature: int = 19,
               n_permutation_list: int = 6,
               dropout: float = 0.2,
               UseBN: bool = True):
    """
    Lattice Convolution Layer used in the main model

    Parameters
    ----------
    input_feature: int
        Dimenion of the concatenated input vector. Node_feature_size*number of neighbors
    output_feature: int, default 19
        Dimension of feature size of the convolution
    n_permutation_list: int, default 6
        Diffrent permutations taken along diffrent directions
    dropout: float
        p value for dropout between 0.0 to 1.0
    UseBN: bool
        To use batch normalisation
    """
    super(LCNNBlock, self).__init__()

    self.conv_weights = nn.Linear(input_feature, output_feature)
    self.batch_norm = nn.BatchNorm1d(output_feature)
    self.UseBN = UseBN
    self.activation = Shifted_softplus()
    self.dropout = Custom_dropout(dropout, n_permutation_list)
    self.permutation = n_permutation_list

  def reduce_func(self, nodes):
    number_of_sites = nodes.mailbox['m'].shape[0]
    return {
        'X_site': nodes.mailbox['m'].view(number_of_sites, self.permutation, -1)
    }

  def forward(self, G, node_feats):
    """
    Update node representations.

    Parameters
    ----------
    G: DGLGraph
        DGLGraph for a batch of graphs.
    node_feats: torch.Tensor
        The node features. The shape is `(N, Node_feature_size)`.

    Returns
    -------
    node_feats: torch.Tensor
        The updated node features. The shape is `(N, Node_feature_size)`.
    """
    try:
      import dgl.function as fn
    except:
      raise ImportError("This class requires DGL to be installed.")
    G = G.local_var()
    G.ndata['x'] = node_feats
    G.update_all(fn.copy_u('x', 'm'), self.reduce_func)
    X = self.conv_weights(G.ndata['X_site'])
    X = torch.stack([self.batch_norm(X_i) for X_i in X])
    node_feats = torch.stack([self.dropout(X_i).sum(axis=0) for X_i in X])
    return node_feats


class Atom_Wise_Convolution(nn.Module):
  """
  Performs self convolution to each node
  """

  def __init__(self,
               input_feature: int,
               output_feature: int,
               dropout: float = 0.2,
               UseBN: bool = True):
    """
    Parameters
    ----------
    input_feature: int
        Size of input feature size
    output_feature: int
        Size of output feature size
    dropout: float, defult 0.2
        p value for dropout between 0.0 to 1.0
    UseBN: bool
        Setting it to True will perform Batch Normalisation
    """
    super(Atom_Wise_Convolution, self).__init__()
    self.conv_weights = nn.Linear(input_feature, output_feature)
    self.batch_norm = nn.LayerNorm(output_feature)
    self.UseBN = UseBN
    self.activation = Shifted_softplus()
    self.dropout = nn.Dropout(p=dropout)

  def forward(self, node_feats):
    """
    Update node representations.

    Parameters
    ----------
    node_feats: torch.Tensor
        The node features. The shape is `(N, Node_feature_size)`.

    Returns
    -------
    node_feats: torch.Tensor
        The updated node features. The shape is `(N, Node_feature_size)`.

    """

    node_feats = self.conv_weights(node_feats)
    if self.UseBN:
      node_feats = self.batch_norm(node_feats)

    node_feats = self.activation(node_feats)
    node_feats = self.dropout(node_feats)

    return node_feats


class Shifted_softplus(nn.Module):
  """
  Performs a Shifter softplus loss, which modifies with a value of log(2)
  """

  def __init__(self):
    super(Shifted_softplus, self).__init__()
    self.act = nn.Softplus()
    self.shift = nn.Parameter(torch.tensor([0.69310]), False)

  def forward(self, X):
    """
    Applies the Activation function

    Parameters
    ----------
    node_feats: torch.Tensor
        The node features.

    Returns
    -------
    node_feats: torch.Tensor
        The updated node features.

    """
    node_feats = self.act(X) - self.shift
    return node_feats


class Custom_dropout(nn.Module):
  """
  An implementation for few , Given a task perform a rowise sum of 2-d
  matrix , you get a zero out the contribution of few of rows in the matrix
  Given, X a 2-d matrix consisting of row vectors (1-d) x1 , x2 ,..xn.
  Sum = x1 + 0.x2 + .. + 0.xi + .. +xn
  """

  def __init__(self, dp_rate: float, n_permutation: int):
    """
    Parameters
    ----------
    dp_rate: float
        p value of dropout.
    """
    super(Custom_dropout, self).__init__()
    self.dropout = nn.Dropout(p=dp_rate)
    self.ones = nn.Parameter(torch.ones(n_permutation), requires_grad=False)

  def forward(self, layer):
    """
    Returns
    -------
    node_feats: torch.Tensor
        Updated tensor.
    """
    mask = self.dropout(self.ones).view(layer.shape[0], 1).repeat(
        1, layer.shape[1])
    return mask * layer


class LCNN(nn.Module):
  """
  The Lattice Convolution Neural Network (LCNN)

  This model takes lattice representation of Adsorbate Surface to predict
  coverage effects taking into consideration the adjacent elements interaction
  energies.

  The model follows the following steps

  [1] It performs n lattice convolution operations. For more details look at the LCNNBlock class
  [2] Followed by Linear layer transforming into sitewise_n_feature
  [3] Transformation to scalar value for each node.
  [4] Average of properties per each element in a configuration

  Examples
  --------
  >>> import deepchem as dc
  >>> from pymatgen.core import Structure
  >>> import numpy as np
  >>> PRIMITIVE_CELL = {
  ...   "lattice": [[2.818528, 0.0, 0.0],
  ...               [-1.409264, 2.440917, 0.0],
  ...               [0.0, 0.0, 25.508255]],
  ...   "coords": [[0.66667, 0.33333, 0.090221],
  ...              [0.33333, 0.66667, 0.18043936],
  ...              [0.0, 0.0, 0.27065772],
  ...              [0.66667, 0.33333, 0.36087608],
  ...              [0.33333, 0.66667, 0.45109444],
  ...              [0.0, 0.0, 0.49656991]],
  ...   "species": ['H', 'H', 'H', 'H', 'H', 'He'],
  ...   "site_properties": {'SiteTypes': ['S1', 'S1', 'S1', 'S1', 'S1', 'A1']}
  ... }
  >>> PRIMITIVE_CELL_INF0 = {
  ...    "cutoff": np.around(6.00),
  ...    "structure": Structure(**PRIMITIVE_CELL),
  ...    "aos": ['1', '0', '2'],
  ...    "pbc": [True, True, False],
  ...    "ns": 1,
  ...    "na": 1
  ... }
  >>> DATA_POINT = {
  ...   "lattice": [[1.409264, -2.440917, 0.0],
  ...               [4.227792, 2.440917, 0.0],
  ...               [0.0, 0.0, 23.17559]],
  ...   "coords": [[0.0, 0.0, 0.099299],
  ...              [0.0, 0.33333, 0.198598],
  ...              [0.5, 0.16667, 0.297897],
  ...              [0.0, 0.0, 0.397196],
  ...              [0.0, 0.33333, 0.496495],
  ...              [0.5, 0.5, 0.099299],
  ...              [0.5, 0.83333, 0.198598],
  ...              [0.0, 0.66667, 0.297897],
  ...              [0.5, 0.5, 0.397196],
  ...              [0.5, 0.83333, 0.496495],
  ...              [0.0, 0.66667, 0.54654766],
  ...              [0.5, 0.16667, 0.54654766]],
  ...   "species": ['H', 'H', 'H', 'H', 'H', 'H',
  ...               'H', 'H', 'H', 'H', 'He', 'He'],
  ...   "site_properties": {
  ...     "SiteTypes": ['S1', 'S1', 'S1', 'S1', 'S1',
  ...                   'S1', 'S1', 'S1', 'S1', 'S1',
  ...                   'A1', 'A1'],
  ...     "oss": ['-1', '-1', '-1', '-1', '-1', '-1',
  ...             '-1', '-1', '-1', '-1', '0', '2']
  ...                   }
  ... }
  >>> featuriser = dc.feat.LCNNFeaturizer(**PRIMITIVE_CELL_INF0)
  >>> lcnn_feat = featuriser._featurize(Structure(**DATA_POINT)).to_dgl_graph()
  >>> print(type(lcnn_feat))
  <class 'dgl.heterograph.DGLHeteroGraph'>
  >>> model = LCNN()
  >>> out = model(lcnn_feat)
  >>> print(type(out))
  <class 'torch.Tensor'>


  Refrences
  -----------

  [1] Jonathan Lym,Geun Ho Gu, Yousung Jung , and Dionisios G. Vlachos
  "Lattice Convolutional Neural Network Modeling of Adsorbate Coverage
  Effects" The Journal of Physical Chemistry

  [2] https://forum.deepchem.io/t/lattice-convolutional-neural-network-modeling-of-adsorbate-coverage-effects/124

  Notes
  -----
  This class requires DGL and PyTorch to be installed.
  """

  def __init__(self,
               n_occupancy: int = 3,
               n_neighbor_sites: int = 19,
               n_permutation: int = 6,
               n_task: int = 1,
               dropout_rate: float = 0.2,
               n_conv: int = 2,
               n_features: int = 19,
               sitewise_n_feature: int = 25):
    """
    Parameters
    ----------

    n_occupancy: int, default 3
        number of possible occupancy
    n_neighbor_sites_list: int, default 19
        Number of neighbors of each site.
    n_permutation: int, default 6
        Diffrent permutations taken along diffrent directions.
    n_task: int, default 1
        Number of tasks
    dropout_rate: float, default 0.2
        p value for dropout between 0.0 to 1.0
    nconv: int, default 2
        number of convolutions performed
    n_feature: int, default 19
        number of feature for each site
    sitewise_n_feature: int, default 25
        number of features for atoms for site-wise activation

    """
    super(LCNN, self).__init__()

    modules = [LCNNBlock(n_occupancy * n_neighbor_sites, n_features)]
    for i in range(n_conv - 1):
      modules.append(
          LCNNBlock(n_features * n_neighbor_sites, n_features, n_permutation))

    self.LCNN_blocks = nn.Sequential(*modules)
    self.Atom_wise_Conv = Atom_Wise_Convolution(n_features, sitewise_n_feature)
    self.Atom_wise_Lin = nn.Linear(sitewise_n_feature, sitewise_n_feature)
    self.fc = nn.Linear(sitewise_n_feature, n_task)
    self.activation = Shifted_softplus()

  def forward(self, G):
    """
    Parameters
    ----------
    G: DGLGraph
        DGLGraph for a batch of graphs.

    Returns
    -------
    y: torch.Tensor
        A single scalar value

    """
    try:
      import dgl
    except:
      raise ImportError("This class requires DGL to be installed.")
    G = G.local_var()
    node_feats = G.ndata.pop('x')
    for conv in self.LCNN_blocks:
      node_feats = conv(G, node_feats)
    node_feats = self.Atom_wise_Conv(node_feats)
    node_feats = self.Atom_wise_Lin(node_feats)
    G.ndata['new'] = self.activation(node_feats)
    y = dgl.mean_nodes(G, 'new')
    y = self.fc(y)
    return y


class LCNNModel(TorchModel):
  """
  Lattice Convolutional Neural Network (LCNN).
  Here is a simple example of code that uses the LCNNModel with
  Platinum 2d Adsorption dataset.

  This model takes arbitrary configurations of Molecules on an adsorbate and predicts
  their formation energy. These formation energies are found using DFT calculations and
  LCNNModel is to automate that process. This model defines a crystal graph using the
  distance between atoms. The crystal graph is an undirected regular graph (equal neighbours)
  and different permutations of the neighbours are pre-computed using the LCNNFeaturizer.
  On each node for each permutation, the neighbour nodes are concatenated which are further operated.
  This model has only a node representation. Please confirm the detail algorithms from [1]_.

  Examples
  --------
  >>>
  >> import deepchem as dc
  >> from pymatgen.core import Structure
  >> import numpy as np
  >> from deepchem.feat import LCNNFeaturizer
  >> from deepchem.molnet import load_Platinum_Adsorption
  >> PRIMITIVE_CELL = {
  ..   "lattice": [[2.818528, 0.0, 0.0],
  ..               [-1.409264, 2.440917, 0.0],
  ..               [0.0, 0.0, 25.508255]],
  ..   "coords": [[0.66667, 0.33333, 0.090221],
  ..              [0.33333, 0.66667, 0.18043936],
  ..              [0.0, 0.0, 0.27065772],
  ..              [0.66667, 0.33333, 0.36087608],
  ..              [0.33333, 0.66667, 0.45109444],
  ..              [0.0, 0.0, 0.49656991]],
  ..   "species": ['H', 'H', 'H', 'H', 'H', 'He'],
  ..   "site_properties": {'SiteTypes': ['S1', 'S1', 'S1', 'S1', 'S1', 'A1']}
  .. }
  >> PRIMITIVE_CELL_INF0 = {
  ..    "cutoff": np.around(6.00),
  ..    "structure": Structure(**PRIMITIVE_CELL),
  ..    "aos": ['1', '0', '2'],
  ..    "pbc": [True, True, False],
  ..    "ns": 1,
  ..    "na": 1
  .. }
  >> tasks, datasets, transformers = load_Platinum_Adsorption(
  ..    featurizer= LCNNFeaturizer( **PRIMITIVE_CELL_INF0)
  .. )
  >> train, val, test = datasets
  >> model = LCNNModel(mode='regression',
  ..                   batch_size=8,
  ..                   learning_rate=0.001)
  >> model = LCNN()
  >> out = model(lcnn_feat)
  >> model.fit(train, nb_epoch=10)


  References
  ----------
  .. [1] Jonathan Lym and Geun Ho Gu, J. Phys. Chem. C 2019, 123, 18951−18959.

  Notes
  -----
  This class requires DGL and PyTorch to be installed.
  """

  def __init__(self,
               n_occupancy: int = 3,
               n_neighbor_sites_list: int = 19,
               n_permutation_list: int = 6,
               n_task: int = 1,
               dropout_rate: float = 0.4,
               n_conv: int = 2,
               n_features: int = 44,
               sitewise_n_feature: int = 25,
               **kwargs):
    """
    This class accepts all the keyword arguments from TorchModel.

    Parameters
    ----------

    n_occupancy: int, default 3
        number of possible occupancy.
    n_neighbor_sites_list: int, default 19
        Number of neighbors of each site.
    n_permutation: int, default 6
        Diffrent permutations taken along diffrent directions.
    n_task: int, default 1
        Number of tasks.
    dropout_rate: float, default 0.4
        p value for dropout between 0.0 to 1.0
    nconv: int, default 2
        number of convolutions performed.
    n_feature: int, default 44
        number of feature for each site.
    sitewise_n_feature: int, default 25
        number of features for atoms for site-wise activation.
    kwargs: Dict
        This class accepts all the keyword arguments from TorchModel.
    """

    def init_weights(m):
      if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)

    model = LCNN(n_occupancy, n_neighbor_sites_list, n_permutation_list, n_task,
                 dropout_rate, n_conv, n_features, sitewise_n_feature)
    model.apply(init_weights)
    loss = L2Loss()
    output_types = ['prediction']
    super(LCNNModel, self).__init__(
        model, loss=loss, output_types=output_types, **kwargs)

  def _prepare_batch(self, batch):
    """
    Create batch data for LCNN.

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
    """
    try:
      import dgl
    except:
      raise ImportError("This class requires DGL to be installed.")

    inputs, labels, weights = batch
    dgl_graphs = [graph.to_dgl_graph() for graph in inputs[0]]
    inputs = dgl.batch(dgl_graphs).to(self.device)
    _, labels, weights = super(LCNNModel, self)._prepare_batch(([], labels,
                                                                weights))
    return inputs, labels, weights
