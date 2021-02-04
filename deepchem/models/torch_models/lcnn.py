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
    """

  def __init__(self,
               input_feature: int,
               output_feature: int = 5,
               n_permutation_list: int = 6,
               dropout: float = 0.2,
               UseBN: bool = True):
    """
        Lattice Convolution Layer used in the main model

        Parameters
        ----------
        input_feature: int
            Dimenion of the concatenated input vector. Node_feature_size*number of neighbors
        output_feature: int
            Dimension of feature size of the convolution
        n_permutation_list: int
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
    G.ndata['x'] = node_feats
    G.update_all(fn.copy_u('x', 'm'), self.reduce_func)
    X = self.conv_weights(G.ndata['X_site'])
    X = torch.stack([self.batch_norm(X_i) for X_i in X])
    node_feats = torch.stack([self.dropout(X_i).sum(axis=0) for X_i in X])
    return node_feats


class Atom_Wise_Linear(nn.Module):
  """
    Performs Matrix Multiplication
    It is used to transform each node wise feature into a scalar.

    """

  def __init__(self,
               input_feature: int,
               output_feature: int,
               dropout: float = 0.0,
               UseBN: bool = True):
    """
        Parameters
        ----------
        input_feature: int
            Size of input feature size
        output_feature: int
            Size of output feature size
        dropout: float
            p value for dropout between 0.0 to 1.0
        UseBN: bool
            Setting it to True will perform Batch Normalisation

        """
    super(Atom_Wise_Linear, self).__init__()
    self.conv_weights = nn.Linear(input_feature, output_feature)

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
        dropout: float
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

    [1] It performs n lattice convolution operations.
        For more details look at the LCNNBlock class
    [2] Followed by Linear layer transforming into sitewise_n_feature
    [3] Transformation to scalar value for each node.
    [4] Average of properties per each element in a configuration

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
               dropout_rate: float = 0.2,
               n_conv: int = 2,
               n_features: int = 19,
               sitewise_n_feature: int = 25):
    """
        parameters
        ----------

        n_occupancy: int
            number of possible occupancy
        n_neighbor_sites_list: int
            Number of neighbors of each site.
        n_permutation: int
            Diffrent permutations taken along diffrent directions.
        dropout_rate: float
            p value for dropout between 0.0 to 1.0
        nconv: int
            number of convolutions performed
        n_feature: int
            number of feature for each site
        sitewise_n_feature: int
            number of features for atoms for site-wise activation

        """
    super(LCNN, self).__init__()

    modules = [LCNNBlock(n_occupancy * n_neighbor_sites, n_features)]
    for i in range(n_conv - 1):
      modules.append(
          LCNNBlock(n_features * n_neighbor_sites, n_features, n_permutation))

    self.LCNN_blocks = nn.Sequential(*modules)
    self.Atom_wise_Conv = Atom_Wise_Convolution(n_features, sitewise_n_feature)
    self.Atom_wise_Lin = Atom_Wise_Linear(sitewise_n_feature,
                                          sitewise_n_feature)

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
    node_feats = G.ndata.pop('x')

    for conv in self.LCNN_blocks:
      node_feats = conv(G, node_feats)
    node_feats = self.Atom_wise_Conv(node_feats)
    node_feats = self.Atom_wise_Lin(node_feats).sum(axis=1)
    G.ndata['new'] = node_feats

    y = dgl.mean_nodes(G, 'new')
    return y


class LCNNModel(TorchModel):
  """Lattice Convolutional Neural Network (LCNN).
    Here is a simple example of code that uses the LCNNModel with
    Platinum 2d Adsorption dataset.

    - import deepchem as dc
    - tasks, datasets, transformers = dc.molnet.load_Platinum_Adsorption()
    - train, valid, test = datasets
    - model = dc.models.LCNNModel(mode='regression', batch_size=8, learning_rate=0.001)
    - model.fit(train, nb_epoch=10)

    This model takes arbitrary configurations of Molecules on an adsorbate and predicts
    their formation energy. These formation energies are found using DFT calculations and
    LCNNModel is to automate that process. This model defines a crystal graph using the
    distance between atoms. The crystal graph is an undirected regular graph (equal neighbours)
    and different permutations of the neighbours are pre-computed using the LCNNFeaturizer.
    On each node for each permutation, the neighbour nodes are concatenated which are further operated.
    This model has only a node representation. Please confirm the detail algorithms from [1]_.

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
               dropout_rate: float = 0.4,
               n_conv: int = 2,
               n_features: int = 44,
               sitewise_n_feature: int = 25,
               **kwargs):

    def init_weights(m):
      if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)

    model = LCNN(n_occupancy, n_neighbor_sites_list, n_permutation_list,
                 dropout_rate, n_conv, n_features, sitewise_n_feature)
    model.apply(init_weights)
    loss = L2Loss()
    output_types = ['prediction']
    super(LCNNModel, self).__init__(
        model, loss=loss, output_types=output_types, **kwargs)

  def _prepare_batch(self, batch):
    """Create batch data for LCNN.
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
