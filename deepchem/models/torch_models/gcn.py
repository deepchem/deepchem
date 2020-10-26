"""
DGL-based GCN for graph property prediction.
"""
import torch.nn as nn
import torch.nn.functional as F

from deepchem.models.losses import L2Loss, SparseSoftmaxCrossEntropy
from deepchem.models.torch_models.torch_model import TorchModel

class GCN(nn.Module):
    """Model for Graph Property Prediction Based on Graph Convolution Networks (GCN).

    This model proceeds as follows:

    * Update node representations in graphs with a variant of GCN
    * For each graph, compute its representation by 1) a weighted sum of the node
      representations in the graph, where the weights are computed by applying a
      gating function to the node representations 2) a max pooling of the node
      representations 3) concatenating the output of 1) and 2)
    * Perform the final prediction using an MLP

    Examples
    --------

    >>> import deepchem as dc
    >>> import pymatgen as mg
    >>> from deepchem.models import GCN
    >>> lattice = mg.Lattice.cubic(4.2)
    >>> structure = mg.Structure(lattice, ["Cs", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    >>> featurizer = dc.feat.CGCNNFeaturizer()
    >>> cgcnn_graph = featurizer.featurize([structure])[0]
    >>> cgcnn_graph.num_node_features
    92
    >>> cgcnn_dgl_graph = cgcnn_graph.to_dgl_graph()
    >>> model = GCN(in_node_dim=92, hidden_node_dim=92, num_gnn_layers=2)
    >>> model(cgcnn_dgl_graph)

    References
    ----------
    .. [1] Thomas N. Kipf and Max Welling. "Semi-Supervised Classification with Graph
       Convolutional Networks." ICLR 2017.

    Notes
    -----
    This class requires DGL (https://github.com/dmlc/dgl) and DGL-LifeSci
    (https://github.com/awslabs/dgl-lifesci) to be installed.
    """
    def __init__(self,
                 in_node_dim: int,
                 hidden_node_dim: int,
                 num_gnn_layers: int,
                 activation = None,
                 residual: bool = True,
                 batchnorm: bool = False,
                 dropout: float = 0.,
                 predictor_hidden_feats: int = 128,
                 predictor_dropout: float = 0.,
                 n_tasks: int = 1,
                 mode: str = 'regression',
                 n_classes: int = 2,
                 nfeat_name: str = 'x'):
        """
        Parameters
        ----------
        in_node_dim: int
            The length of the initial node feature vectors.
        hidden_node_dim: int
            The length of the hidden node feature vectors.
        num_gnn_layers: int
            The number of GCN layers.
        activation: callable
            The activation function to apply to the output of each GCN layer.
            By default, no activation function will be applied.
        residual: bool
            Whether to add a residual connection within each GCN layer. Default to True.
        batchnorm: bool
            Whether to apply batch normalization to the output of each GCN layer.
            Default to False.
        dropout: float
            The dropout probability for the output of each GCN layer. Default to 0.
        predictor_hidden_feats: int
            The size for hidden representations in the output MLP predictor. Default to 128.
        predictor_dropout: float
            The dropout probability in the output MLP predictor. Default to 0.
        n_tasks: int
            The output size.
        mode: str
            The model type, 'classification' or 'regression'.
        n_classes: int
            The number of classes to predict per task
            (only used when ``mode`` is 'classification').
        nfeat_name: str
            For an input graph ``g``, the model assumes that it stores node features in
            ``g.ndata[nfeat_name]`` and will retrieve input node features from that.
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

        super(GCN, self).__init__()

        self.n_tasks = n_tasks
        self.mode = mode
        self.n_classes = n_classes
        self.nfeat_name = nfeat_name
        if mode == 'classification':
            out_size = n_tasks * n_classes
        else:
            out_size = n_tasks

        from dgllife.model import GCNPredictor as DGLGCNPredictor

        if activation is not None:
            activation = [activation] * num_gnn_layers

        self.model = DGLGCNPredictor(in_feats=in_node_dim,
                                     hidden_feats=[hidden_node_dim] * num_gnn_layers,
                                     activation=activation,
                                     residual=[residual] * num_gnn_layers,
                                     batchnorm=[batchnorm] * num_gnn_layers,
                                     dropout=[dropout] * num_gnn_layers,
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
        node_feats = g.ndata.pop(self.nfeat_name)
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

class GCNModel(TorchModel):
    """Model for Graph Property Prediction Based on Graph Convolution Networks (GCN).

    This model proceeds as follows:

    * Update node representations in graphs with a variant of GCN
    * For each graph, compute its representation by 1) a weighted sum of the node
      representations in the graph, where the weights are computed by applying a
      gating function to the node representations 2) a max pooling of the node
      representations 3) concatenating the output of 1) and 2)
    * Perform the final prediction using an MLP

    Examples
    --------
    # Todo

    References
    ----------
    .. [1] Thomas N. Kipf and Max Welling. "Semi-Supervised Classification with Graph
       Convolutional Networks." ICLR 2017.

    Notes
    -----
    This class requires DGL (https://github.com/dmlc/dgl) and DGL-LifeSci
    (https://github.com/awslabs/dgl-lifesci) to be installed.
    """
    def __init__(self,
                 in_node_dim: int,
                 hidden_node_dim: int,
                 num_gnn_layers: int,
                 activation = None,
                 residual: bool = True,
                 batchnorm: bool = False,
                 dropout: float = 0.,
                 predictor_hidden_feats: int = 128,
                 predictor_dropout: float = 0.,
                 n_tasks: int = 1,
                 mode: str = 'regression',
                 n_classes: int = 2,
                 nfeat_name: str = 'x',
                 **kwargs):
        """
        Parameters
        ----------
        in_node_dim: int
            The length of the initial node feature vectors.
        hidden_node_dim: int
            The length of the hidden node feature vectors.
        num_gnn_layers: int
            The number of GCN layers.
        activation: callable
            The activation function to apply to the output of each GCN layer.
            By default, no activation function will be applied.
        residual: bool
            Whether to add a residual connection within each GCN layer. Default to True.
        batchnorm: bool
            Whether to apply batch normalization to the output of each GCN layer.
            Default to False.
        dropout: float
            The dropout probability for the output of each GCN layer. Default to 0.
        predictor_hidden_feats: int
            The size for hidden representations in the output MLP predictor. Default to 128.
        predictor_dropout: float
            The dropout probability in the output MLP predictor. Default to 0.
        n_tasks: int
            The output size.
        mode: str
            The model type, 'classification' or 'regression'.
        n_classes: int
            The number of classes to predict per task
            (only used when ``mode`` is 'classification').
        nfeat_name: str
            For an input graph ``g``, the model assumes that it stores node features in
            ``g.ndata[nfeat_name]`` and will retrieve input node features from that.
        kwargs
            This can include any keyword argument of TorchModel.
        """
        model = GCN(in_node_dim=in_node_dim,
                    hidden_node_dim=hidden_node_dim,
                    num_gnn_layers=num_gnn_layers,
                    activation=activation,
                    residual=residual,
                    batchnorm=batchnorm,
                    dropout=dropout,
                    predictor_hidden_feats=predictor_hidden_feats,
                    predictor_dropout=predictor_dropout,
                    n_tasks=n_tasks,
                    mode=mode,
                    n_classes=n_classes,
                    nfeat_name=nfeat_name)
        if mode == 'regression':
            loss = L2Loss()
            output_types = ['prediction']
        else:
            loss = SparseSoftmaxCrossEntropy()
            output_types = ['prediction', 'loss']
        super(GCNModel, self).__init__(
            model, loss=loss, output_types=output_types, **kwargs)

    def _prepare_batch(self, batch):
        """Create batch data for GCN.

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
        dgl_graphs = [graph.to_dgl_graph() for graph in inputs[0]]
        inputs = dgl.batch(dgl_graphs).to(self.device)
        _, labels, weights = super(GCNModel, self)._prepare_batch(([], labels, weights))
        return inputs, labels, weights
