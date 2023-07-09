from functools import partial
from math import sqrt
from typing import Callable, Dict, List, Union

import dgl
import torch
from torch import nn

from deepchem.feat.molecule_featurizers.conformer_featurizer import (
    full_atom_feature_dims,
    full_bond_feature_dims,
)
from deepchem.models.torch_models.layers import MultilayerPerceptron
from deepchem.utils.graph_utils import (
    aggregate_max,
    aggregate_mean,
    aggregate_min,
    aggregate_moment,
    aggregate_std,
    aggregate_sum,
    aggregate_var,
    scale_amplification,
    scale_attenuation,
    scale_identity,
)

PNA_AGGREGATORS = {
    "mean": aggregate_mean,
    "sum": aggregate_sum,
    "max": aggregate_max,
    "min": aggregate_min,
    "std": aggregate_std,
    "var": aggregate_var,
    "moment3": partial(aggregate_moment, n=3),
    "moment4": partial(aggregate_moment, n=4),
    "moment5": partial(aggregate_moment, n=5),
}

PNA_SCALERS = {
    "identity": scale_identity,
    "amplification": scale_amplification,
    "attenuation": scale_attenuation,
}


class AtomEncoder(torch.nn.Module):
    """
    Encodes atom features into embeddings based on the Open Graph Benchmark feature set in conformer_featurizer.

    Parameters
    ----------
    emb_dim : int
        The dimension that the returned embedding will have.
    padding : bool, optional (default=False)
        If true then the last index will be used for padding.

    Examples
    --------
    >>> from deepchem.feat.molecule_featurizers.conformer_featurizer import full_atom_feature_dims
    >>> atom_encoder = AtomEncoder(emb_dim=32)
    >>> num_rows = 10
    >>> atom_features = torch.stack([
    ... torch.randint(low=0, high=dim, size=(num_rows,))
    ... for dim in full_atom_feature_dims
    ... ], dim=1)
    >>> atom_embeddings = atom_encoder(atom_features)
    """

    def __init__(self, emb_dim, padding=False):
        super(AtomEncoder, self).__init__()

        self.atom_embedding_list = torch.nn.ModuleList()
        self.padding = padding

        for dim in full_atom_feature_dims:
            if padding:
                emb = torch.nn.Embedding(dim + 1, emb_dim, padding_idx=0)
            else:
                emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def reset_parameters(self):
        """
        Reset the parameters of the atom embeddings.

        This method resets the weights of the atom embeddings by initializing
        them with a uniform distribution between -sqrt(3) and sqrt(3).
        """
        for embedder in self.atom_embedding_list:
            embedder.weight.data.uniform_(-sqrt(3), sqrt(3))

    def forward(self, x):
        """
        Compute the atom embeddings for the given atom features.

        Parameters
        ----------
        x : torch.Tensor, shape (batch_size, num_atoms, num_features)
            The input atom features tensor.

        Returns
        -------
        x_embedding : torch.Tensor, shape (batch_size, num_atoms, emb_dim)
            The computed atom embeddings.
        """
        x_embedding = 0
        for i in range(x.shape[1]):
            if self.padding:
                x_embedding += self.atom_embedding_list[i](x[:, i].long() + 1)
            else:
                x_embedding += self.atom_embedding_list[i](x[:, i].long())

        return x_embedding


class BondEncoder(torch.nn.Module):
    """
    Encodes bond features into embeddings based on the Open Graph Benchmark feature set in conformer_featurizer.

    Parameters
    ----------
    emb_dim : int
        The dimension that the returned embedding will have.
    padding : bool, optional (default=False)
        If true then the last index will be used for padding.

    Examples
    --------
    >>> from deepchem.feat.molecule_featurizers.conformer_featurizer import full_bond_feature_dims
    >>> bond_encoder = BondEncoder(emb_dim=32)
    >>> num_rows = 10
    >>> bond_features = torch.stack([
    ... torch.randint(low=0, high=dim, size=(num_rows,))
    ... for dim in full_bond_feature_dims
    ... ], dim=1)
    >>> bond_embeddings = bond_encoder(bond_features)
    """

    def __init__(self, emb_dim, padding=False):
        super(BondEncoder, self).__init__()

        self.bond_embedding_list = torch.nn.ModuleList()
        self.padding = padding

        for dim in full_bond_feature_dims:
            if padding:
                emb = torch.nn.Embedding(dim + 1, emb_dim, padding_idx=0)
            else:
                emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, edge_attr):
        """
        Compute the bond embeddings for the given bond features.

        Parameters
        ----------
        edge_attr : torch.Tensor, shape (batch_size, num_edges, num_features)
            The input bond features tensor.

        Returns
        -------
        bond_embedding : torch.Tensor, shape (batch_size, num_edges, emb_dim)
            The computed bond embeddings.
        """
        bond_embedding = 0
        for i in range(edge_attr.shape[1]):
            if self.padding:
                bond_embedding += self.bond_embedding_list[i](
                    edge_attr[:, i].long() + 1)
            else:
                bond_embedding += self.bond_embedding_list[i](
                    edge_attr[:, i].long())

        return bond_embedding


class PNALayer(nn.Module):
    """
    Principal Neighbourhood Aggregation Layer (PNA) from [1].

    Parameters
    ----------
    in_dim : int
        Input dimension of the node features.
    out_dim : int
        Output dimension of the node features.
    in_dim_edges : int
        Input dimension of the edge features.
    aggregators : List[str]
        List of aggregator functions to use. Options are "mean", "sum", "max", "min", "std", "var", "moment3", "moment4", "moment5".
    scalers : List[str]
        List of scaler functions to use. Options are "identity", "amplification", "attenuation".
    activation : Union[Callable, str], optional, default="relu"
        Activation function to use.
    last_activation : Union[Callable, str], optional, default="none"
        Last activation function to use.
    dropout : float, optional, default=0.0
        Dropout rate.
    residual : bool, optional, default=True
        Whether to use residual connections.
    pairwise_distances : bool, optional, default=False
        Whether to use pairwise distances.
    batch_norm : bool, optional, default=True
        Whether to use batch normalization.
    batch_norm_momentum : float, optional, default=0.1
        Momentum for the batch normalization layers.
    avg_d : Dict[str, float], optional, default={"log": 1.0}
        Dictionary containing the average degree of the graph.
    posttrans_layers : int, optional, default=2
        Number of post-transformation layers.
    pretrans_layers : int, optional, default=1
        Number of pre-transformation layers.

    References
    ----------
    .. [1] Corso, G., Cavalleri, L., Beaini, D., Liò, P. & Veličković, P. Principal Neighbourhood Aggregation for Graph Nets. Preprint at https://doi.org/10.48550/arXiv.2004.05718 (2020).

    Examples
    --------
    >>> import dgl
    >>> import numpy as np
    >>> import torch
    >>> from deepchem.models.torch_models.pna_gnn import PNALayer
    >>> in_dim = 32
    >>> out_dim = 64
    >>> in_dim_edges = 16
    >>> aggregators = ["mean", "max"]
    >>> scalers = ["identity", "amplification", "attenuation"]
    >>> pna_layer = PNALayer(in_dim=in_dim,
    ...                      out_dim=out_dim,
    ...                      in_dim_edges=in_dim_edges,
    ...                      aggregators=aggregators,
    ...                      scalers=scalers)
    >>> num_nodes = 10
    >>> num_edges = 20
    >>> node_features = torch.randn(num_nodes, in_dim)
    >>> edge_features = torch.randn(num_edges, in_dim_edges)
    >>> g = dgl.graph((np.random.randint(0, num_nodes, num_edges),
    ...                np.random.randint(0, num_nodes, num_edges)))
    >>> g.ndata['feat'] = node_features
    >>> g.edata['feat'] = edge_features
    >>> g.ndata['feat'] = pna_layer(g)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        in_dim_edges: int,
        aggregators: List[str],
        scalers: List[str],
        activation: Union[Callable, str] = "relu",
        dropout: float = 0.0,
        residual: bool = True,
        pairwise_distances: bool = False,
        batch_norm: bool = True,
        batch_norm_momentum=0.1,
        avg_d: Dict[str, float] = {"log": 1.0},
        posttrans_layers: int = 2,
        pretrans_layers: int = 1,
    ):
        super(PNALayer, self).__init__()
        self.aggregators = [PNA_AGGREGATORS[aggr] for aggr in aggregators]
        self.scalers = [PNA_SCALERS[scale] for scale in scalers]
        self.edge_features = in_dim_edges > 0
        self.activation = activation
        self.avg_d = avg_d
        self.pairwise_distances = pairwise_distances
        self.residual = residual
        if in_dim != out_dim:
            self.residual = False

        self.pretrans = MultilayerPerceptron(
            d_input=(2 * in_dim + in_dim_edges +
                     1) if self.pairwise_distances else
            (2 * in_dim + in_dim_edges),
            d_output=in_dim,
            d_hidden=(in_dim,) * (pretrans_layers - 1),
            batch_norm=batch_norm,
            batch_norm_momentum=batch_norm_momentum,
            dropout=dropout)

        self.posttrans = MultilayerPerceptron(
            d_input=(len(self.aggregators) * len(self.scalers) + 1) * in_dim,
            d_hidden=(out_dim,) * (posttrans_layers - 1),
            d_output=out_dim,
            batch_norm=batch_norm,
            batch_norm_momentum=batch_norm_momentum,
            dropout=dropout)

    def forward(self, g):
        """
        Forward pass of the PNA layer.

        Parameters
        ----------
        g : dgl.DGLGraph
            Input graph

        Returns
        -------
        h : torch.Tensor
            Node feature tensor
        """
        h = g.ndata['feat']
        h_in = h
        # pretransformation
        g.apply_edges(self.pretrans_edges)

        # aggregation
        g.update_all(self.message_func, self.reduce_func)
        h = torch.cat([h, g.ndata['feat']], dim=-1)
        # post-transformation
        h = self.posttrans(h)
        if self.residual:
            h = h + h_in

        return h

    def message_func(self, edges) -> Dict[str, torch.Tensor]:
        """
        The message function to generate messages along the edges.

        Parameters
        ----------
        edges : dgl.EdgeBatch
            Batch of edges.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing the edge features.
        """
        return {"e": edges.data["e"]}

    def reduce_func(self, nodes) -> Dict[str, torch.Tensor]:
        """
        The reduce function to aggregate the messages.
        Apply the aggregators and scalers, and concatenate the results.

        Parameters
        ----------
        nodes : dgl.NodeBatch
            Batch of nodes.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing the aggregated node features.
        """
        h_in = nodes.data['feat']
        h = nodes.mailbox["e"]
        D = h.shape[-2]

        h_to_cat = [
            aggr(h=h, h_in=h_in)  # type: ignore
            for aggr in self.aggregators
        ]
        h = torch.cat(h_to_cat, dim=-1)

        if len(self.scalers) > 1:
            h = torch.cat(
                [
                    scale(h, D=D, avg_d=self.avg_d)  # type: ignore
                    for scale in self.scalers
                ],
                dim=-1)

        return {'feat': h}

    def pretrans_edges(self, edges) -> Dict[str, torch.Tensor]:
        """
        Return a mapping to the concatenation of the features from
        the source node, the destination node, and the edge between them (if applicable).

        Parameters
        ----------
        edges : dgl.EdgeBatch
            Batch of edges.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing the concatenated features.
        """

        if self.edge_features and self.pairwise_distances:
            squared_distance = torch.sum((edges.src['x'] - edges.dst['x'])**2,
                                         dim=-1)[:, None]
            z2 = torch.cat([
                edges.src['feat'], edges.dst['feat'], edges.data['feat'],
                squared_distance
            ],
                           dim=-1)
        elif not self.edge_features and self.pairwise_distances:
            squared_distance = torch.sum((edges.src['x'] - edges.dst['x'])**2,
                                         dim=-1)[:, None]
            z2 = torch.cat(
                [edges.src['feat'], edges.dst['feat'], squared_distance],
                dim=-1)
        elif self.edge_features and not self.pairwise_distances:
            z2 = torch.cat(
                [edges.src['feat'], edges.dst['feat'], edges.data['feat']],
                dim=-1)
        else:
            z2 = torch.cat([edges.src['feat'], edges.dst['feat']], dim=-1)
        return {"e": self.pretrans(z2)}


class PNAGNN(nn.Module):
    """
    Principal Neighbourhood Aggregation Graph Neural Network [1]. This defines the message passing layers of the PNA model.

    Parameters
    ----------
    hidden_dim : int
        Dimension of the hidden layers.
    aggregators : List[str]
        List of aggregator functions to use.
    scalers : List[str]
        List of scaler functions to use. Options are "identity", "amplification", "attenuation".
    residual : bool, optional, default=True
        Whether to use residual connections.
    pairwise_distances : bool, optional, default=False
        Whether to use pairwise distances.
    activation : Union[Callable, str], optional, default="relu"
        Activation function to use.
    batch_norm : bool, optional, default=True
        Whether to use batch normalization in the layers before the aggregator.
    batch_norm_momentum : float, optional, default=0.1
        Momentum for the batch normalization layers.
    propagation_depth : int, optional, default=5
        Number of propagation layers.
    dropout : float, optional, default=0.0
        Dropout rate.
    posttrans_layers : int, optional, default=1
        Number of post-transformation layers.
    pretrans_layers : int, optional, default=1
        Number of pre-transformation layers.

    References
    ----------
    .. [1] Corso, G., Cavalleri, L., Beaini, D., Liò, P. & Veličković, P. Principal Neighbourhood Aggregation for Graph Nets. Preprint at https://doi.org/10.48550/arXiv.2004.05718 (2020).

    Examples
    --------
    >>> import numpy as np
    >>> from deepchem.feat.molecule_featurizers.conformer_featurizer import RDKitConformerFeaturizer
    >>> from deepchem.feat.graph_data import BatchGraphData
    >>> from deepchem.models.torch_models.pna_gnn import PNAGNN
    >>> featurizer = RDKitConformerFeaturizer(num_conformers=2)
    >>> smiles = ['C1=CC=NC=C1', 'CC(=O)C', 'C']
    >>> featurizer = RDKitConformerFeaturizer(num_conformers=2, rmsd_cutoff=1)
    >>> data = featurizer.featurize(smiles)
    >>> features = BatchGraphData(np.concatenate(data))
    >>> features = features.to_dgl_graph()
    >>> model = PNAGNN(hidden_dim=16,
    ...                aggregators=['mean', 'sum'],
    ...                scalers=['identity'])
    >>> output = model(features)
    """

    def __init__(self,
                 hidden_dim,
                 aggregators: List[str],
                 scalers: List[str],
                 residual: bool = True,
                 pairwise_distances: bool = False,
                 activation: Union[Callable, str] = "relu",
                 batch_norm: bool = True,
                 batch_norm_momentum=0.1,
                 propagation_depth: int = 5,
                 dropout: float = 0.0,
                 posttrans_layers: int = 1,
                 pretrans_layers: int = 1,
                 **kwargs):
        super(PNAGNN, self).__init__()

        self.mp_layers = nn.ModuleList()

        for _ in range(propagation_depth):
            self.mp_layers.append(
                PNALayer(in_dim=hidden_dim,
                         out_dim=int(hidden_dim),
                         in_dim_edges=hidden_dim,
                         aggregators=aggregators,
                         scalers=scalers,
                         pairwise_distances=pairwise_distances,
                         residual=residual,
                         dropout=dropout,
                         activation=activation,
                         avg_d={"log": 1.0},
                         posttrans_layers=posttrans_layers,
                         pretrans_layers=pretrans_layers,
                         batch_norm=batch_norm,
                         batch_norm_momentum=batch_norm_momentum),)
        self.atom_encoder = AtomEncoder(emb_dim=hidden_dim)
        self.bond_encoder = BondEncoder(emb_dim=hidden_dim)

    def forward(self, input_graph: dgl.DGLGraph) -> dgl.DGLGraph:
        """
        Forward pass of the PNAGNN model.

        Parameters
        ----------
        input_graph : dgl.DGLGraph
            Input graph with node and edge features.

        Returns
        -------
        graph : dgl.DGLGraph
            Output graph with updated node features after applying the message passing layers.
        """
        graph = input_graph.clone()
        graph.ndata['feat'] = self.atom_encoder(graph.ndata['x'])
        graph.edata['feat'] = self.bond_encoder(graph.edata['edge_attr'])

        for mp_layer in self.mp_layers:
            graph.ndata['feat'] = mp_layer(graph)

        return graph


class PNA(nn.Module):
    """
    Message passing neural network for graph representation learning [1]_.

    Parameters
    ----------
    hidden_dim : int
        Hidden dimension size.
    target_dim : int
        Dimensionality of the output, for example for binary classification target_dim = 1.
    aggregators : List[str]
        Type of message passing functions. Options are 'mean','sum','max','min','std','var','moment3','moment4','moment5'.
    scalers : List[str]
        Type of normalization layers in the message passing network. Options are 'identity','amplification','attenuation'.
    readout_aggregators : List[str]
        Type of aggregators in the readout network.
    readout_hidden_dim : int, default None
       The dimension of the hidden layer in the readout network. If not provided, the readout has the same dimensionality of the final layer of the PNA layer, which is the hidden dimension size.
    readout_layers : int, default 1
        The number of linear layers in the readout network.
    residual : bool, default True
        Whether to use residual connections.
    pairwise_distances : bool, default False
        Whether to use pairwise distances.
    activation : Union[Callable, str]
        Activation function to use.
    batch_norm : bool, default True
        Whether to use batch normalization in the layers before the aggregator..
    batch_norm_momentum : float, default 0.1
        Momentum for the batch normalization layers.
    propagation_depth : int, default
        Number of propagation layers.
    dropout : float, default 0.0
        Dropout probability in the message passing layers.
    posttrans_layers : int, default 1
        Number of post-transformation layers.
    pretrans_layers : int, default 1
        Number of pre-transformation layers.

    References
    ----------
    .. [1] Corso, G., Cavalleri, L., Beaini, D., Liò, P. & Veličković, P. Principal Neighbourhood Aggregation for Graph Nets. Preprint at https://doi.org/10.48550/arXiv.2004.05718 (2020).

    Examples
    --------
    >>> import numpy as np
    >>> from deepchem.feat.graph_data import BatchGraphData
    >>> from deepchem.models.torch_models.pna_gnn import PNA
    >>> from deepchem.feat.molecule_featurizers.conformer_featurizer import RDKitConformerFeaturizer
    >>> smiles = ["C1=CC=CN=C1", "C1CCC1"]
    >>> featurizer = RDKitConformerFeaturizer(num_conformers=2)
    >>> data = featurizer.featurize(smiles)
    >>> features = BatchGraphData(np.concatenate(data))
    >>> features = features.to_dgl_graph()
    >>> target_dim = 1
    >>> model = PNA(hidden_dim=16, target_dim=target_dim)
    >>> output = model(features)
    >>> print(output.shape)
    torch.Size([1, 1])
    """

    def __init__(self,
                 hidden_dim: int,
                 target_dim: int,
                 aggregators: List[str] = ['mean'],
                 scalers: List[str] = ['identity'],
                 readout_aggregators: List[str] = ['mean'],
                 readout_hidden_dim: int = 1,
                 readout_layers: int = 2,
                 residual: bool = True,
                 pairwise_distances: bool = False,
                 activation: Union[Callable, str] = "relu",
                 batch_norm: bool = True,
                 batch_norm_momentum: float = 0.1,
                 propagation_depth: int = 5,
                 dropout: float = 0.0,
                 posttrans_layers: int = 1,
                 pretrans_layers: int = 1,
                 **kwargs):
        super(PNA, self).__init__()
        self.node_gnn = PNAGNN(hidden_dim=hidden_dim,
                               aggregators=aggregators,
                               scalers=scalers,
                               residual=residual,
                               pairwise_distances=pairwise_distances,
                               activation=activation,
                               batch_norm=batch_norm,
                               batch_norm_momentum=batch_norm_momentum,
                               propagation_depth=propagation_depth,
                               dropout=dropout,
                               posttrans_layers=posttrans_layers,
                               pretrans_layers=pretrans_layers)
        if readout_hidden_dim == 1:
            readout_hidden_dim = hidden_dim
        self.readout_aggregators = readout_aggregators
        self.output = MultilayerPerceptron(
            d_input=hidden_dim * len(self.readout_aggregators),
            d_hidden=(readout_hidden_dim,) * readout_layers,
            batch_norm=False,
            d_output=target_dim)

    def forward(self, graph: dgl.DGLGraph):
        graph = self.node_gnn(graph)
        readouts_to_cat = [
            dgl.readout_nodes(graph, 'feat', op=aggr)
            for aggr in self.readout_aggregators
        ]
        readout = torch.cat(readouts_to_cat, dim=-1)
        return self.output(readout)
