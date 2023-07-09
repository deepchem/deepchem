from typing import Callable, List, Union

import dgl
import dgl.function as fn
import torch
from torch import nn
from torch.nn import functional as F

from deepchem.models.losses import NTXentMultiplePositives
from deepchem.models.torch_models import ModularTorchModel
from deepchem.models.torch_models.layers import MultilayerPerceptron
from deepchem.models.torch_models.pna_gnn import PNA, AtomEncoder
from deepchem.utils.graph_utils import fourier_encode_dist


class Net3DLayer(nn.Module):
    """
    Net3DLayer is a single layer of a 3D graph neural network based on the 3D Infomax architecture [1].

    This class expects a DGL graph with node features stored under the name 'feat' and edge features stored under the name 'd' (representing 3D distances). The edge features are updated by the message network and the node features are updated by the update network.

    Parameters
    ----------
    edge_dim : int
        The dimension of the edge features.
    hidden_dim : int
        The dimension of the hidden layers.
    reduce_func : str
        The reduce function to use for aggregating messages. Can be either 'sum' or 'mean'.
    batch_norm : bool, optional (default=False)
        Whether to use batch normalization.
    batch_norm_momentum : float, optional (default=0.1)
        The momentum for the batch normalization layers.
    dropout : float, optional (default=0.0)
        The dropout rate for the layers.
    mid_activation : str, optional (default='SiLU')
        The activation function to use in the network.
    message_net_layers : int, optional (default=2)
        The number of message network layers.
    update_net_layers : int, optional (default=2)
        The number of update network layers.

    References
    ----------
    .. [1] Stärk, H. et al. 3D Infomax improves GNNs for Molecular Property Prediction. Preprint at https://doi.org/10.48550/arXiv.2110.04126 (2022).

    Examples
    --------
    >>> net3d_layer = Net3DLayer(edge_dim=3, hidden_dim=3)
    >>> graph = dgl.graph(([0, 1], [1, 2]))
    >>> graph.ndata['feat'] = torch.tensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
    >>> graph.edata['d'] = torch.tensor([[0.5, 0.6, 0.7], [0.8, 0.9, 1.0]])
    >>> output = net3d_layer(graph)
    """

    def __init__(self,
                 edge_dim: int,
                 hidden_dim: int,
                 reduce_func: str = 'sum',
                 batch_norm: bool = False,
                 batch_norm_momentum: float = 0.1,
                 dropout: float = 0.0,
                 message_net_layers: int = 2,
                 update_net_layers: int = 2):
        super(Net3DLayer, self).__init__()

        self.message_network = nn.Sequential(
            MultilayerPerceptron(d_input=hidden_dim * 2 + edge_dim,
                                 d_output=hidden_dim,
                                 d_hidden=(hidden_dim,) *
                                 (message_net_layers - 1),
                                 batch_norm=batch_norm,
                                 batch_norm_momentum=batch_norm_momentum,
                                 dropout=dropout), torch.nn.SiLU())
        if reduce_func == 'sum':
            self.reduce_func = fn.sum
        elif reduce_func == 'mean':
            self.reduce_func = fn.mean
        else:
            raise ValueError('reduce function not supported: ', reduce_func)

        self.update_network = MultilayerPerceptron(
            d_input=hidden_dim,
            d_hidden=(hidden_dim,) * (update_net_layers - 1),
            d_output=hidden_dim,
            batch_norm=True,
            batch_norm_momentum=batch_norm_momentum)

        self.soft_edge_network = nn.Linear(hidden_dim, 1)

    def forward(self, input_graph: dgl.DGLGraph):
        """Perform a forward pass on the given graph.

        Parameters
        ----------
        input_graph : dgl.DGLGraph
            The graph to perform the forward pass on.

        Returns
        -------
        dgl.DGLGraph
            The updated graph after the forward pass.
        """
        # copy the input graph to avoid in-place operations
        graph = input_graph.local_var()
        graph.ndata['feat'] = input_graph.ndata['feat'].clone()
        graph.edata['d'] = input_graph.edata['d'].clone()

        graph.update_all(message_func=self.message_function,
                         reduce_func=self.reduce_func(msg='m', out='m_sum'),
                         apply_node_func=self.update_function)
        return graph

    def message_function(self, edges):
        """Computes the message and edge weight for a given set of edges.

        Parameters
        ----------
        edges : dgl.EdgeBatch
            A dgl.EdgeBatch object containing the edges information (data, batch size, etc.).

        Returns
        -------
        dict
            A dictionary containing the message multiplied by the edge weight.
        """
        message_input = torch.cat(
            [edges.src['feat'], edges.dst['feat'], edges.data['d']], dim=-1)
        message = self.message_network(message_input)
        edges.data['d'] += message
        edge_weight = torch.sigmoid(self.soft_edge_network(message))
        return {'m': message * edge_weight}

    def update_function(self, nodes):
        """
        Update function for updating node features based on the aggregated messages.

        This function is used in the forward method to perform a forward pass on the graph.

        Parameters
        ----------
        nodes : dgl.NodeBatch
            A node batch object containing the nodes information (data, batch size, etc.).

        Returns
        -------
        dict
            A dictionary containing the updated features.
        """
        h = nodes.data['feat']
        input = torch.cat([nodes.data['m_sum'] + nodes.data['feat']], dim=-1)
        h_new = self.update_network(input)
        output = h_new + h
        return {'feat': output}


class Net3D(nn.Module):
    """
    Net3D is a 3D graph neural network that expects a DGL graph input with 3D coordinates stored under the name 'd' and node features stored under the name 'feat'. It is based on the 3D Infomax architecture [1].

    Parameters
    ----------
    hidden_dim : int
        The dimension of the hidden layers.
    target_dim : int
        The dimension of the output layer.
    readout_aggregators : List[str]
        A list of aggregator functions for the readout layer. Options are 'sum', 'max', 'min', 'mean'.
    batch_norm : bool, optional (default=False)
        Whether to use batch normalization.
    node_wise_output_layers : int, optional (default=2)
        The number of output layers for each node.
    readout_batchnorm : bool, optional (default=True)
        Whether to use batch normalization in the readout layer.
    batch_norm_momentum : float, optional (default=0.1)
        The momentum for the batch normalization layers.
    reduce_func : str, optional (default='sum')
        The reduce function to use for aggregating messages.
    dropout : float, optional (default=0.0)
        The dropout rate for the layers.
    propagation_depth : int, optional (default=4)
        The number of propagation layers in the network.
    readout_layers : int, optional (default=2)
        The number of readout layers in the network.
    readout_hidden_dim : int, optional (default=None)
        The dimension of the hidden layers in the readout network.
    fourier_encodings : int, optional (default=0)
        The number of Fourier encodings to use.
    activation : str, optional (default='SiLU')
        The activation function to use in the network.
    update_net_layers : int, optional (default=2)
        The number of update network layers.
    message_net_layers : int, optional (default=2)
        The number of message network layers.
    use_node_features : bool, optional (default=False)
        Whether to use node features as input.

    Examples
    --------
    >>> from deepchem.feat.molecule_featurizers.conformer_featurizer import RDKitConformerFeaturizer
    >>> from deepchem.models.torch_models.gnn3d import Net3D
    >>> smiles = ["C[C@H](F)Cl", "C[C@@H](F)Cl"]
    >>> featurizer = RDKitConformerFeaturizer(num_conformers=2)
    >>> data = featurizer.featurize(smiles)
    >>> dgldata = [[graph.to_dgl_graph() for graph in conf] for conf in data]
    >>> net3d = Net3D(hidden_dim=3, target_dim=2, readout_aggregators=['sum', 'mean'])
    >>> output = [[net3d(graph) for graph in conf] for conf in dgldata]

    References
    ----------
    .. [1] Stärk, H. et al. 3D Infomax improves GNNs for Molecular Property Prediction. Preprint at https://doi.org/10.48550/arXiv.2110.04126 (2022).
    """

    def __init__(self,
                 hidden_dim,
                 target_dim,
                 readout_aggregators: List[str],
                 node_wise_output_layers=2,
                 batch_norm=True,
                 batch_norm_momentum=0.1,
                 reduce_func='sum',
                 dropout=0.0,
                 propagation_depth: int = 4,
                 readout_layers: int = 2,
                 readout_hidden_dim=None,
                 fourier_encodings=4,
                 update_net_layers=2,
                 message_net_layers=2,
                 use_node_features=False):
        super(Net3D, self).__init__()
        self.fourier_encodings = fourier_encodings
        edge_in_dim = 1 if fourier_encodings == 0 else 2 * fourier_encodings + 1

        self.edge_input = nn.Sequential(
            MultilayerPerceptron(d_input=edge_in_dim,
                                 d_output=hidden_dim,
                                 d_hidden=(hidden_dim,),
                                 batch_norm=batch_norm,
                                 batch_norm_momentum=batch_norm_momentum),
            torch.nn.SiLU())

        self.use_node_features = use_node_features
        if self.use_node_features:
            self.atom_encoder = AtomEncoder(hidden_dim)
        else:
            self.node_embedding = nn.Parameter(torch.empty((hidden_dim,)))
            nn.init.normal_(self.node_embedding)

        self.mp_layers = nn.ModuleList()
        for _ in range(propagation_depth):
            self.mp_layers.append(
                Net3DLayer(edge_dim=hidden_dim,
                           hidden_dim=hidden_dim,
                           batch_norm=batch_norm,
                           batch_norm_momentum=batch_norm_momentum,
                           dropout=dropout,
                           reduce_func=reduce_func,
                           message_net_layers=message_net_layers,
                           update_net_layers=update_net_layers))

        self.node_wise_output_layers = node_wise_output_layers
        if self.node_wise_output_layers > 0:
            self.node_wise_output_network = MultilayerPerceptron(
                d_input=hidden_dim,
                d_output=hidden_dim,
                d_hidden=(hidden_dim,),
                batch_norm=batch_norm,
                batch_norm_momentum=batch_norm_momentum)

        if readout_hidden_dim is None:
            readout_hidden_dim = hidden_dim
        self.readout_aggregators = readout_aggregators

        self.output = MultilayerPerceptron(
            d_input=hidden_dim * len(self.readout_aggregators),
            d_output=target_dim,
            d_hidden=(readout_hidden_dim,) *
            (readout_layers -
             1),  # -1 because the input layer is not considered a hidden layer
            batch_norm=False)

    def forward(self, graph: dgl.DGLGraph):
        """
        Forward pass of the Net3D model.

        Parameters
        ----------
        graph : dgl.DGLGraph
            The input graph with node features stored under the key 'x' and edge distances stored under the key 'd'.

        Returns
        -------
        torch.Tensor
            The graph representation tensor of shape (1, target_dim).
        """
        if self.use_node_features:
            graph.ndata['feat'] = self.atom_encoder(graph.ndata['x'])
        else:
            graph.ndata['feat'] = self.node_embedding[None, :].expand(
                graph.number_of_nodes(), -1)

        if self.fourier_encodings > 0:
            graph.edata['d'] = fourier_encode_dist(
                graph.edata['d'], num_encodings=self.fourier_encodings)
        graph.apply_edges(self.input_edge_func)

        for mp_layer in self.mp_layers:
            graph = mp_layer(graph)

        if self.node_wise_output_layers > 0:
            graph.apply_nodes(self.output_node_func)

        readouts_to_cat = [
            dgl.readout_nodes(graph, 'feat', op=aggr)
            for aggr in self.readout_aggregators
        ]
        readout = torch.cat(readouts_to_cat, dim=-1)
        return self.output(readout.squeeze())

    def output_node_func(self, nodes):
        """
        Apply the node-wise output network to the node features.

        Parameters
        ----------
        nodes : dgl.NodeBatch
            A batch of nodes with features stored under the key 'feat'.

        Returns
        -------
        dict
            A dictionary with the updated node features under the key 'feat'.
        """
        return {'feat': self.node_wise_output_network(nodes.data['feat'])}

    def input_edge_func(self, edges):
        """
        Apply the edge input network to the edge features.

        Parameters
        ----------
        edges : dgl.EdgeBatch
            A batch of edges with distances stored under the key 'd'.

        Returns
        -------
        dict
            A dictionary with the updated edge features under the key 'd'.
        """
        return {'d': F.silu(self.edge_input(edges.data['d']))}


class InfoMax3DModular(ModularTorchModel):
    """
    InfoMax3DModular is a modular torch model that uses a 2D PNA model and a 3D Net3D model to maximize the mutual information between their representations. The 2D model can then be used for downstream tasks without the need for 3D coordinates. This is based off the work in [1].

    This class expects data in featurized by the RDKitConformerFeaturizer. This featurizer produces features of the type Array[Array[List[GraphData]]].
    The outermost array is the dataset array, the second array is the molecule, the list contains the conformers for that molecule and the GraphData object is the featurized graph for that conformer with node_pos_features holding the 3D coordinates.
    If you are not using RDKitConformerFeaturizer, your input data features should look like this: Dataset[Molecule[Conformers[GraphData]]].

    Parameters
    ----------
    hidden_dim : int
        The dimension of the hidden layers.
    target_dim : int
        The dimension of the output layer.
    aggregators : List[str]
        A list of aggregator functions for the PNA model. Options are 'mean', 'sum', 'min', 'max', 'std', 'var', 'moment3', 'moment4', 'moment5'.
    readout_aggregators : List[str]
        A list of aggregator functions for the readout layer. Options are 'sum', 'max', 'min', 'mean'.
    scalers : List[str]
        A list of scaler functions for the PNA model. Options are 'identity', 'amplification', 'attenuation'.
    residual : bool, optional (default=True)
        Whether to use residual connections in the PNA model.
    node_wise_output_layers : int, optional (default=2)
        The number of output layers for each node in the Net3D model.
    pairwise_distances : bool, optional (default=False)
        Whether to use pairwise distances in the PNA model.
    activation : Union[Callable, str], optional (default="relu")
        The activation function to use in the PNA model.
    reduce_func : str, optional (default='sum')
        The reduce function to use for aggregating messages in the Net3D model.
    batch_norm : bool, optional (default=True)
        Whether to use batch normalization in the PNA model.
    batch_norm_momentum : float, optional (default=0.1)
        The momentum for the batch normalization layers.
    propagation_depth : int, optional (default=5)
        The number of propagation layers in the PNA and Net3D models.
    dropout : float, optional (default=0.0)
        The dropout rate for the layers in the PNA and Net3D models.
    readout_layers : int, optional (default=2)
        The number of readout layers in the PNA and Net3D models.
    readout_hidden_dim : int, optional (default=None)
        The dimension of the hidden layers in the readout network.
    fourier_encodings : int, optional (default=4)
        The number of Fourier encodings to use in the Net3D model.
    update_net_layers : int, optional (default=2)
        The number of update network layers in the Net3D model.
    message_net_layers : int, optional (default=2)
        The number of message network layers in the Net3D model.
    use_node_features : bool, optional (default=False)
        Whether to use node features as input in the Net3D model.
    posttrans_layers : int, optional (default=1)
        The number of post-transformation layers in the PNA model.
    pretrans_layers : int, optional (default=1)
        The number of pre-transformation layers in the PNA model.
    kwargs : dict
        Additional keyword arguments.

    References
    ----------
    .. [1] Stärk, H. et al. 3D Infomax improves GNNs for Molecular Property Prediction. Preprint at https://doi.org/10.48550/arXiv.2110.04126 (2022).

    Examples
    --------
    >>> from deepchem.feat.graph_data import BatchGraphData
    >>> from deepchem.feat.molecule_featurizers.conformer_featurizer import RDKitConformerFeaturizer
    >>> from deepchem.models.torch_models.gnn3d import InfoMax3DModular
    >>> import numpy as np
    >>> import deepchem as dc
    >>> from deepchem.data.datasets import NumpyDataset
    >>> smiles = ["C[C@H](F)Cl", "C[C@@H](F)Cl"]
    >>> featurizer = RDKitConformerFeaturizer(num_conformers=2)
    >>> data = featurizer.featurize(smiles)
    >>> dataset = NumpyDataset(X=data)
    >>> model = InfoMax3DModular(hidden_dim=64,
    ...                          target_dim=10,
    ...                          aggregators=['max'],
    ...                          readout_aggregators=['mean'],
    ...                          scalers=['identity'])
    >>> loss = model.fit(dataset, nb_epoch=1)
    """

    def __init__(self,
                 hidden_dim,
                 target_dim,
                 aggregators: List[str],
                 readout_aggregators: List[str],
                 scalers: List[str],
                 residual: bool = True,
                 node_wise_output_layers: int = 2,
                 pairwise_distances: bool = False,
                 activation: Union[Callable, str] = "relu",
                 reduce_func: str = 'sum',
                 batch_norm: bool = True,
                 batch_norm_momentum: float = 0.1,
                 propagation_depth: int = 5,
                 dropout: float = 0.0,
                 readout_layers: int = 2,
                 readout_hidden_dim: int = 1,
                 fourier_encodings: int = 4,
                 update_net_layers: int = 2,
                 message_net_layers: int = 2,
                 use_node_features: bool = False,
                 posttrans_layers: int = 1,
                 pretrans_layers: int = 1,
                 **kwargs):
        self.hidden_dim = hidden_dim
        self.target_dim = target_dim
        self.aggregators = aggregators
        self.readout_aggregators = readout_aggregators
        self.scalers = scalers
        self.residual = residual
        self.node_wise_output_layers = node_wise_output_layers
        self.pairwise_distances = pairwise_distances
        self.activation = activation
        self.reduce_func = reduce_func
        self.batch_norm = batch_norm
        self.batch_norm_momentum = batch_norm_momentum
        self.propagation_depth = propagation_depth
        self.dropout = dropout
        self.readout_layers = readout_layers
        self.readout_hidden_dim = readout_hidden_dim
        self.fourier_encodings = fourier_encodings
        self.update_net_layers = update_net_layers
        self.message_net_layers = message_net_layers
        self.use_node_features = use_node_features
        self.posttrans_layers = posttrans_layers
        self.pretrans_layers = pretrans_layers
        self.kwargs = kwargs
        self.criterion = NTXentMultiplePositives()._create_pytorch_loss()
        self.components = self.build_components()
        self.model = self.build_model()
        super().__init__(self.model, self.components, **kwargs)

    def build_components(self):
        """
        Build the components of the InfoMax3DModular model.

        Returns
        -------
        dict
            A dictionary containing the '2d' PNA model and the '3d' Net3D model.
        """
        return {
            '2d':
                PNA(hidden_dim=self.hidden_dim,
                    target_dim=self.target_dim,
                    aggregators=self.aggregators,
                    scalers=self.scalers,
                    readout_aggregators=self.readout_aggregators,
                    readout_hidden_dim=self.readout_hidden_dim,
                    readout_layers=self.readout_layers,
                    residual=self.residual,
                    pairwise_distances=self.pairwise_distances,
                    activation=self.activation,
                    batch_norm=self.batch_norm,
                    batch_norm_momentum=self.batch_norm_momentum,
                    propagation_depth=self.propagation_depth,
                    dropout=self.dropout,
                    posttrans_layers=self.posttrans_layers,
                    pretrans_layers=self.pretrans_layers,
                    **self.kwargs),
            '3d':
                Net3D(hidden_dim=self.hidden_dim,
                      target_dim=self.target_dim,
                      readout_aggregators=self.readout_aggregators,
                      node_wise_output_layers=self.node_wise_output_layers,
                      batch_norm=True,
                      batch_norm_momentum=self.batch_norm_momentum,
                      reduce_func=self.reduce_func,
                      dropout=self.dropout,
                      propagation_depth=self.propagation_depth,
                      readout_layers=self.readout_layers,
                      readout_hidden_dim=self.readout_hidden_dim,
                      fourier_encodings=self.fourier_encodings,
                      update_net_layers=self.update_net_layers,
                      message_net_layers=self.message_net_layers,
                      use_node_features=self.use_node_features),
        }

    def build_model(self):
        """
        Build the InfoMax3DModular model. This is the 2D network which is meant to be used for inference.

        Returns
        -------
        PNA
            The 2D PNA model component.
        """
        return self.components['2d']

    def loss_func(self, inputs, labels, weights):
        """
        Compute the loss function for the InfoMax3DModular model.

        Parameters
        ----------
        inputs : dgl.DGLGraph
            The input graph with node features stored under the key 'x' and edge distances stored under the key 'd'.
        labels : torch.Tensor
            The ground truth labels.
        weights : torch.Tensor
            The weights for each sample.

        Returns
        -------
        torch.Tensor
            The computed loss value.
        """
        encodings2d = []
        encodings3d = []
        for conformers in inputs:
            # 2d model takes only the first conformer
            encodings2d.append(self.components['2d'](conformers[0]))
            # 3d model takes all conformers
            encodings3d.append(
                [self.components['3d'](conf) for conf in conformers])

        # concat the lists such that the 2d encodings is of shape batch_size x target_dim
        # and the 3d encodings is of shape batch_size*num_conformers x target_dim
        encodings2d = torch.cat(encodings2d, dim=0)
        encodings3d = torch.cat(
            [torch.cat(conf, dim=0) for conf in encodings3d], dim=0)
        loss = self.criterion(encodings2d, encodings3d)
        return loss

    def _prepare_batch(self, batch):
        """
        Prepare a batch of data for the InfoMax3DModular model.

        Parameters
        ----------
        batch : tuple
            A tuple containing the inputs, labels, and weights.

        Returns
        -------
        tuple
            A tuple containing the prepared batch graph, labels, and weights.
        """
        inputs, labels, weights = batch
        inputs = inputs[0]

        # convert the GraphData objects to DGL graphs
        graphs = [[
            graph_data.to_dgl_graph().to(self.device) for graph_data in row
        ] for row in inputs]
        return graphs, labels, weights
