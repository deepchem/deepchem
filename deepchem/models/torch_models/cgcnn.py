"""
This is a sample implementation for working DGL with DeepChem!
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from deepchem.models.losses import Loss, L2Loss, SparseSoftmaxCrossEntropy
from deepchem.models.torch_models.torch_model import TorchModel


class CGCNNLayer(nn.Module):
    """The convolutional layer of CGCNN.

    This class was implemented using DGLGraph methods.
    Please confirm how to use DGLGraph methods from below link.
    See: https://docs.dgl.ai/en/0.4.x/tutorials/models/1_gnn/9_gat.html

    Examples
    --------
    >>> import deepchem as dc
    >>> from pymatgen.core import Lattice, Structure
    >>> lattice = Lattice.cubic(4.2)
    >>> structure = Structure(lattice, ["Cs", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    >>> featurizer = dc.feat.CGCNNFeaturizer()
    >>> cgcnn_graph = featurizer.featurize([structure])[0]
    >>> cgcnn_graph.num_node_features
    92
    >>> cgcnn_graph.num_edge_features
    41
    >>> cgcnn_dgl_graph = cgcnn_graph.to_dgl_graph()
    >>> print(type(cgcnn_dgl_graph))
    <class 'dgl.heterograph.DGLHeteroGraph'>
    >>> layer = CGCNNLayer(hidden_node_dim=92, edge_dim=41)
    >>> node_feats = cgcnn_dgl_graph.ndata.pop('x')
    >>> edge_feats = cgcnn_dgl_graph.edata.pop('edge_attr')
    >>> new_node_feats, new_edge_feats = layer(cgcnn_dgl_graph, node_feats, edge_feats)

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
        liner_out_dim = 2 * hidden_node_dim
        self.linear = nn.Linear(z_dim, liner_out_dim)
        self.batch_norm = nn.BatchNorm1d(liner_out_dim) if batch_norm else None

    def message_func(self, edges):
        z = torch.cat([edges.src['x'], edges.dst['x'], edges.data['edge_attr']],
                      dim=1)
        z = self.linear(z)
        if self.batch_norm is not None:
            z = self.batch_norm(z)
        gated_z, message_z = z.chunk(2, dim=1)
        gated_z = torch.sigmoid(gated_z)
        message_z = F.softplus(message_z)
        return {'message': gated_z * message_z}

    def reduce_func(self, nodes):
        nbr_sumed = torch.sum(nodes.mailbox['message'], dim=1)
        new_x = F.softplus(nodes.data['x'] + nbr_sumed)
        return {'new_x': new_x}

    def forward(self, dgl_graph, node_feats, edge_feats):
        """Update node representations.

        Parameters
        ----------
        dgl_graph: DGLGraph
            DGLGraph for a batch of graphs.
        node_feats: torch.Tensor
            The node features. The shape is `(N, hidden_node_dim)`.
        edge_feats: torch.Tensor
            The edge features. The shape is `(N, hidden_node_dim)`.

        Returns
        -------
        node_feats: torch.Tensor
            The updated node features. The shape is `(N, hidden_node_dim)`.
        """
        dgl_graph.ndata['x'] = node_feats
        dgl_graph.edata['edge_attr'] = edge_feats
        dgl_graph.update_all(self.message_func, self.reduce_func)
        node_feats = dgl_graph.ndata.pop('new_x')
        return node_feats


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
    >>> from pymatgen.core import Lattice, Structure
    >>> lattice = Lattice.cubic(4.2)
    >>> structure = Structure(lattice, ["Cs", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    >>> featurizer = dc.feat.CGCNNFeaturizer()
    >>> cgcnn_feat = featurizer.featurize([structure])[0]
    >>> print(type(cgcnn_feat))
    <class 'deepchem.feat.graph_data.GraphData'>
    >>> cgcnn_dgl_feat = cgcnn_feat.to_dgl_graph()
    >>> print(type(cgcnn_dgl_feat))
    <class 'dgl.heterograph.DGLHeteroGraph'>
    >>> model = dc.models.CGCNN(mode='regression', n_tasks=2)
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
        predictor_hidden_feats: int = 128,
        n_tasks: int = 1,
        mode: str = 'regression',
        n_classes: int = 2,
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
        predictor_hidden_feats: int, default 128
            The size for hidden representations in the output MLP predictor.
        n_tasks: int, default 1
            The number of the output size.
        mode: str, default 'regression'
            The model type, 'classification' or 'regression'.
        n_classes: int, default 2
            The number of classes to predict (only used in classification mode).
        """
        try:
            import dgl
        except:
            raise ImportError("This class requires DGL to be installed.")
        super(CGCNN, self).__init__()
        if mode not in ['classification', 'regression']:
            raise ValueError(
                "mode must be either 'classification' or 'regression'")

        self.n_tasks = n_tasks
        self.mode = mode
        self.n_classes = n_classes
        self.embedding = nn.Linear(in_node_dim, hidden_node_dim)
        self.conv_layers = nn.ModuleList([
            CGCNNLayer(hidden_node_dim=hidden_node_dim,
                       edge_dim=in_edge_dim,
                       batch_norm=True) for _ in range(num_conv)
        ])
        self.pooling = dgl.mean_nodes
        self.fc = nn.Linear(hidden_node_dim, predictor_hidden_feats)
        if self.mode == 'regression':
            self.out = nn.Linear(predictor_hidden_feats, n_tasks)
        else:
            self.out = nn.Linear(predictor_hidden_feats, n_tasks * n_classes)

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
            The output values of this model.
            If mode == 'regression', the shape is `(batch_size, n_tasks)`.
            If mode == 'classification', the shape is `(batch_size, n_tasks, n_classes)` (n_tasks > 1)
            or `(batch_size, n_classes)` (n_tasks == 1) and the output values are probabilities of each class label.
        """
        graph = dgl_graph
        # embedding node features
        node_feats = graph.ndata.pop('x')
        edge_feats = graph.edata.pop('edge_attr')
        node_feats = self.embedding(node_feats)

        # convolutional layer
        for conv in self.conv_layers:
            node_feats = conv(graph, node_feats, edge_feats)

        # pooling
        graph.ndata['updated_x'] = node_feats
        graph_feat = F.softplus(self.pooling(graph, 'updated_x'))
        graph_feat = F.softplus(self.fc(graph_feat))
        out = self.out(graph_feat)

        if self.mode == 'regression':
            return out
        else:
            logits = out.view(-1, self.n_tasks, self.n_classes)
            # for n_tasks == 1 case
            logits = torch.squeeze(logits)
            proba = F.softmax(logits)
            return proba, logits


class CGCNNModel(TorchModel):
    """Crystal Graph Convolutional Neural Network (CGCNN).

    Here is a simple example of code that uses the CGCNNModel with
    materials dataset.

    Examples
    --------
    >>> import deepchem as dc
    >>> dataset_config = {"reload": False, "featurizer": dc.feat.CGCNNFeaturizer(), "transformers": []}
    >>> tasks, datasets, transformers = dc.molnet.load_perovskite(**dataset_config)
    >>> train, valid, test = datasets
    >>> model = dc.models.CGCNNModel(mode='regression', batch_size=32, learning_rate=0.001)
    >>> avg_loss = model.fit(train, nb_epoch=50)

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
                 predictor_hidden_feats: int = 128,
                 n_tasks: int = 1,
                 mode: str = 'regression',
                 n_classes: int = 2,
                 **kwargs):
        """This class accepts all the keyword arguments from TorchModel.

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
        predictor_hidden_feats: int, default 128
            The size for hidden representations in the output MLP predictor.
        n_tasks: int, default 1
            The number of the output size.
        mode: str, default 'regression'
            The model type, 'classification' or 'regression'.
        n_classes: int, default 2
            The number of classes to predict (only used in classification mode).
        kwargs: Dict
            This class accepts all the keyword arguments from TorchModel.
        """
        model = CGCNN(in_node_dim, hidden_node_dim, in_edge_dim, num_conv,
                      predictor_hidden_feats, n_tasks, mode, n_classes)
        if mode == "regression":
            loss: Loss = L2Loss()
            output_types = ['prediction']
        else:
            loss = SparseSoftmaxCrossEntropy()
            output_types = ['prediction', 'loss']
        super(CGCNNModel, self).__init__(model,
                                         loss=loss,
                                         output_types=output_types,
                                         **kwargs)

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
        """
        try:
            import dgl
        except:
            raise ImportError("This class requires DGL to be installed.")

        inputs, labels, weights = batch
        dgl_graphs = [graph.to_dgl_graph() for graph in inputs[0]]
        inputs = dgl.batch(dgl_graphs).to(self.device)
        _, labels, weights = super(CGCNNModel, self)._prepare_batch(
            ([], labels, weights))
        return inputs, labels, weights
