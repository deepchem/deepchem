from deepchem.models.torch_models import TorchModel
from deepchem.models.losses import L2Loss
from deepchem.models.torch_models.layers import SE3GraphConv, SE3ResidualAttention, SE3GraphNorm, SE3AvgPooling, Fiber, SE3MaxPooling
from deepchem.utils.equivariance_utils import get_equivariant_basis_and_r
from typing import Optional, Union, List

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:
    raise ImportError('These classes require PyTorch to be installed.')


class SE3Transformer(nn.Module):
    """
    SE(3)-Equivariant Graph Convolutional Network with Attention (SE3Transformer).

    This model is designed for SE(3)-equivariant learning on molecular graphs.
    The core of the model consists of a series of SE(3)-equivariant layers, including **residual attention** layers,
    graph convolution layers, and pooling layers. The architecture allows the model to learn representations
    that are invariant to rotations and translations of the molecular structures.

    The SE3Transformer model consists of:
    1. SE(3)-equivariant graph convolution layers** for learning node-level representations.
    2. Residual attention layers to enable better feature learning.
    3. Pooling layers (either max or average pooling) to aggregate node features into a graph-level representation.
    4. Fully connected layers for final output prediction.

    Parameters
    ----------
    num_layers : int
        Number of SE(3) layers to stack in the model.
    atom_feature_size : int
        Size of input atom features (dimensionality of the initial node features).
    num_channels : int
        Number of channels in the hidden layers.
    num_nlayers : int, optional (default=1)
        Number of layers in the residual attention blocks.
    num_degrees : int, optional (default=4)
        Degree of the SE(3)-equivariant features (i.e., the number of spherical harmonics components to use).
    edge_dim : int, optional (default=4)
        Dimensionality of edge features.
    div : float, optional (default=4)
        Division factor for the attention mechanism.
    pooling : str, optional (default='avg')
        Type of pooling to use after graph convolutions. Choices are 'avg' (average pooling) or 'max' (max pooling).
    n_heads : int, optional (default=1)
        Number of attention heads for the attention mechanism.
    **kwargs :
        Additional arguments for TorchModel.

    Example
    -------
    >>> import torch
    >>> import dgl
    >>> import rdkit
    >>> from rdkit import Chem
    >>> import deepchem as dc
    >>> from deepchem.models.torch_models import SE3Transformer
    >>> from deepchem.utils.equivariance_utils import get_equivariant_basis_and_r

    >>> mol = Chem.MolFromSmiles("CCO")


    >>> featurizer = dc.feat.EquivariantGraphFeaturizer(fully_connected=True, embeded=True)
    >>> features = featurizer.featurize([mol])[0]


    >>> G = dgl.graph((features.edge_index[0], features.edge_index[1]), num_nodes=len(features.node_features))

    >>> # Assign SE(3)-equivariant node & edge features
    >>> G.ndata['x'] = torch.tensor(features.node_features, dtype=torch.float32) # Node features
    >>> G.ndata['pos'] = torch.tensor(features.node_pos_features, dtype=torch.float32)  # 3D coordinates
    >>> G.edata['edge_attr'] = torch.tensor(features.edge_features, dtype=torch.float32)  # Edge distances
    >>> G.edata['w'] = torch.tensor(features.edge_weights, dtype=torch.float32)  # Edge weights
    >>> basis, r = get_equivariant_basis_and_r(G, 3)
    >>> h = {'0': G.ndata['x'].unsqueeze(-1)}
    >>> model = SE3Transformer(num_layers=1, atom_feature_size=6, num_channels=32, edge_dim=5)
    >>> pred = model(G)

    References
    ----------
    .. [1] SE(3)-Transformers: 3D Roto-Translation Equivariant Attention Networks
           Fabian B. Fuchs, Daniel E. Worrall, Volker Fischer, Max Welling
           NeurIPS 2020, https://arxiv.org/abs/2006.10503
    """

    def __init__(self,
                 num_layers: int,
                 atom_feature_size: int,
                 num_channels: int,
                 n_tasks: int = 1,
                 num_nlayers: int = 1,
                 num_degrees: int = 4,
                 edge_dim: int = 4,
                 div: float = 4,
                 pooling: str = 'avg',
                 n_heads: int = 1,
                 **kwargs):
        super().__init__()
        # Build the network
        self.num_layers = num_layers
        self.num_nlayers = num_nlayers
        self.num_channels = num_channels
        self.num_degrees = num_degrees
        self.edge_dim = edge_dim
        self.div = div
        self.pooling = pooling
        self.n_heads = n_heads
        self.n_tasks = n_tasks

        self.fibers = {
            'in': Fiber(1, atom_feature_size),
            'mid': Fiber(num_degrees, self.num_channels),
            'out': Fiber(1, num_degrees * self.num_channels)
        }

        blocks = self._build_gcn(self.fibers, self.n_tasks)
        self.SE3G, self.FC = blocks

    def _build_gcn(self, fibers, out_dim):
        """
        Builds the SE(3) graph convolution layers, attention layers, graph normalization, and pooling.

        Parameters
        ----------
        fibers : dict
            Dictionary of Fiber objects that specify the feature sizes for each layer.
        out_dim : int
            The output dimensionality of the final fully connected layer.

        Returns
        -------
        tuple
            A tuple containing the SE(3) graph layers and fully connected layers.
        """
        # Equivariant layers
        SE3G = []
        fin = fibers['in']
        for _ in range(self.num_layers):
            SE3G.append(
                SE3ResidualAttention(fin,
                                     fibers['mid'],
                                     edge_dim=self.edge_dim,
                                     div=self.div,
                                     n_heads=self.n_heads))
            SE3G.append(SE3GraphNorm(fibers['mid']))
            fin = fibers['mid']

        SE3G.append(
            SE3GraphConv(fibers['mid'],
                         fibers['out'],
                         self_interaction=True,
                         edge_dim=self.edge_dim))

        # Pooling
        if self.pooling == 'avg':
            SE3G.append(SE3AvgPooling())
        elif self.pooling == 'max':
            SE3G.append(SE3MaxPooling())
        else:
            raise ValueError(f"Pooling type {self.pooling} not supported.")

        # Fully connected layers
        FC = []
        FC.append(
            nn.Linear(self.fibers['out'].n_features,
                      self.fibers['out'].n_features))
        FC.append(nn.ReLU(inplace=True))
        FC.append(nn.Linear(self.fibers['out'].n_features, out_dim))

        return nn.ModuleList(SE3G), nn.ModuleList(FC)

    def forward(self, G):
        """
        Forward pass for the SE(3)-equivariant model.

        Parameters
        ----------
        G : dgl.DGLGraph
            DGL graph object containing the molecular structure.

        Returns
        -------
        torch.Tensor
            Model's output after passing through the network layers.
        """
        basis, r = get_equivariant_basis_and_r(G, self.num_degrees - 1)

        h = {'0': G.ndata['x'].unsqueeze(-1)}

        for layer in self.SE3G:
            h = layer(h, G=G, r=r, basis=basis)

        for layer in self.FC:
            h = layer(h)

        return h


class SE3TransformerModel(TorchModel):
    """
    SE3TransformerModel Deepchem Wrapper.

    This class wraps the SE3Transformer model for compatibility with DeepChem's model interface.

    Parameters
    ----------
    num_layers : int
        Number of SE(3) Attention layers.
    atom_feature_size : int
        Dimensionality of atom features.
    num_channels : int
        Number of channels for the hidden layers.
    num_nlayers : int, optional (default=1)
        Number of layers for the residual attention blocks.
    num_degrees : int, optional (default=4)
        Degree of SE(3)-equivariant features.
    edge_dim : int, optional (default=4)
        Dimensionality of edge features.
    pooling : str, optional (default='avg')
        Pooling type: 'avg' for average pooling, 'max' for max pooling.
    n_heads : int, optional (default=1)
        Number of attention heads.
    device : torch.device, optional
        The device (CPU or GPU) on which the model will run.
    **kwargs :
        Additional arguments for TorchModel.

    Example
    -------
    >>> import torch
    >>> from deepchem.models.torch_models import SE3TransformerModel
    >>> import dgl
    >>> import rdkit
    >>> from rdkit import Chem
    >>> import numpy as np
    >>> import deepchem as dc
    >>> smiles = ["CCO", "CC(=O)O", "C1=CC=CC=C1",]
    >>> featurizer = dc.feat.EquivariantGraphFeaturizer(fully_connected=True, embeded=True)
    >>> mols_g = [featurizer.featurize(Chem.MolFromSmiles(mol))[0] for mol in smiles]
    >>> # Extract SE(3)-equivariant features
    >>> labels = np.random.rand(len(mols_g), 12)
    >>> weights = np.ones_like(labels)
    >>> dataset = dc.data.NumpyDataset(X=mols_g, y=labels, w=weights)
    >>> model = SE3TransformerModel(
    ...    task='homo',
    ...    num_layers=7,
    ...    atom_feature_size=6,
    ...    num_workers=4,
    ...    num_channels=32,
    ...    num_nlayers=1,
    ...    num_degrees=4,
    ...    edge_dim=5,
    ...    pooling='max',
    ...    n_heads=8,
    ...    batch_size=12,)
    >>> loss = model.fit(dataset, nb_epoch=1)
    """

    def __init__(self,
                 num_layers: int,
                 atom_feature_size: int,
                 num_workers: int,
                 num_channels: int,
                 task: Union[str, List[str]] = 'homo',
                 num_nlayers: int = 1,
                 num_degrees: int = 4,
                 edge_dim: int = 4,
                 pooling: str = 'avg',
                 n_heads: int = 1,
                 device: Optional[torch.device] = torch.device('cpu'),
                 **kwargs) -> None:
        if device is None:

            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device
        output_types = ['prediction']
        loss = L2Loss()

        if isinstance(task, str):
            self.task = [task]
        else:
            self.task = task
        n_tasks = len(self.task)

        # Initialize the SE3Transformer model
        model = SE3Transformer(
            n_tasks=n_tasks,
            num_layers=num_layers,
            atom_feature_size=atom_feature_size,
            num_workers=num_workers,
            num_channels=num_channels,
            num_nlayers=num_nlayers,
            num_degrees=num_degrees,
            edge_dim=edge_dim,
            pooling=pooling,
            n_heads=n_heads,
        )
        super(SE3TransformerModel, self).__init__(model,
                                                  loss=loss,
                                                  device=self.device,
                                                  output_types=output_types,
                                                  **kwargs)

    def _prepare_batch(self, batch):
        """
        Prepares a batch of data for the SE3Transformer model.

        This function processes DGL graphs from a DeepChem dataset batch,
        converting the graphs and labels into tensors suitable for training.

        Parameters
        ----------
        batch : Tuple
            A batch of data from the dataset in the form (X, y, w, ids).

        Returns
        -------
        Tuple : (input_tensors, labels, weights)
            Tensors prepared for model input, labels, and weights.
        """
        try:
            import dgl
        except ModuleNotFoundError:
            raise ImportError('These classes require DGL to be installed.')

        inputs, labels, weights = batch

        QM9_TASKS = {
            "mu": 0,
            "alpha": 1,
            "homo": 2,
            "lumo": 3,
            "gap": 4,
            "r2": 5,
            "zpve": 6,
            "cv": 7,
            "u0": 8,
            "u298": 9,
            "h298": 10,
            "g298": 11
        }
        if labels is not None and weights is not None:
            indices = []
            for t in self.task:
                indices.append(QM9_TASKS[t])

            labels = [labels[0][:, indices]]
            weights = [weights[0][:, indices]]

        dgl_graphs = []
        for graph in inputs[0]:
            dgl_g = graph.to_dgl_graph()
            dgl_g.edata['w'] = torch.tensor(graph.edge_weights,
                                            dtype=torch.float32)
            dgl_graphs.append(dgl_g)

        inputs = dgl.batch(dgl_graphs).to(self.device)
        _, labels, weights = super(SE3TransformerModel, self)._prepare_batch(
            ([], labels, weights))

        return inputs, labels, weights
