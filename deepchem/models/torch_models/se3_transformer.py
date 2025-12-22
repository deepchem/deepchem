from deepchem.models.torch_models import TorchModel
from deepchem.models.losses import L1Loss
from typing import List, Optional, Union, Dict, Tuple, Any
from deepchem.models.torch_models.layers import SE3GraphConv, SE3ResidualAttention, SE3GraphNorm, SE3AvgPooling, Fiber, SE3MaxPooling
from deepchem.utils.equivariance_utils import get_equivariant_basis_and_r
from deepchem.utils.data_utils import load_from_disk, save_to_disk
import logging
try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:
    raise ImportError('These classes require PyTorch to be installed.')

logger = logging.getLogger(__name__)
LayerType = Union[SE3ResidualAttention, SE3GraphNorm, SE3GraphConv,
                  SE3AvgPooling, SE3MaxPooling, nn.Module]


class SE3Transformer(nn.Module):
    """
    SE(3)-Transformer: A 3D equivariant graph neural network with attention [1].

    This model is designed to be equivariant under 3D rigid body motions (rotations and translations),
    making it particularly suitable for molecular modeling, protein design, and other physics-informed tasks.

    Parameters
    ----------
    num_layers : int
        Number of equivariant attention layers in the GCN.
    atom_feature_size : int
        Dimensionality of the input atomic features.
    num_channels : int
        Number of channels per degree in intermediate SE(3) tensors.
    n_tasks : int, optional
        Number of regression/classification tasks (default is 1).
    num_nlayers : int, optional
        Number of layers in the FC head (default is 1).
    num_degrees : int, optional
        Maximum degree (L) of the SE(3) tensors (default is 4).
    edge_dim : int, optional
        Dimensionality of edge features (default is 4).
    div : float, optional
        Channel division factor in attention layers (default is 2).
    pooling : {'max', 'avg'}, optional
        Global pooling strategy over nodes (default is 'max').
    n_heads : int, optional
        Number of attention heads in SE(3) attention layers (default is 1).
    device : torch.device, optional
        Device to run the model on (default is CPU).

    Example
    -------
    >>> import torch
    >>> import dgl
    >>> from deepchem.models.torch_models import SE3Transformer
    >>> g = dgl.graph(([0, 1, 2], [1, 2, 0]))   # simple 3-node cycle
    >>> g.ndata["x"] = torch.randn(3, 6)        # node features
    >>> g.edata["edge_attr"] = torch.randn(3, 3)
    >>> g.edata["w"] = torch.randn(3, 4)        # matches edge_dim=4
    >>> model = SE3Transformer(
    ...     num_layers=2,
    ...     atom_feature_size=6,
    ...     num_channels=8,
    ...     n_tasks=1,
    ...     num_nlayers=1,
    ...     num_degrees=3,
    ...     edge_dim=4,
    ...     pooling="max",
    ...     n_heads=2,
    ... )
    >>> out = model(g)
    >>> out.shape
    torch.Size([1, 1])

    References
    ----------
    .. [1] SE(3)-Transformers: 3D Roto-Translation Equivariant Attention Networks
           Fabian B. Fuchs, Daniel E. Worrall, Volker Fischer, Max Welling
           NeurIPS 2020, https://arxiv.org/abs/2006.10503
    """
    from dgl import DGLGraph

    def __init__(self,
                 num_layers: int,
                 atom_feature_size: int,
                 num_channels: int,
                 n_tasks: int = 1,
                 num_nlayers: int = 1,
                 num_degrees: int = 4,
                 edge_dim: int = 4,
                 div: float = 2,
                 pooling: str = 'max',
                 n_heads: int = 1,
                 device: Optional[torch.device] = torch.device('cpu'),
                 **kwargs):
        super().__init__()

        # Store hyperparameters
        self.num_layers = num_layers
        self.num_nlayers = num_nlayers
        self.num_channels = num_channels
        self.num_degrees = num_degrees
        self.edge_dim = edge_dim
        self.div = div
        self.pooling = pooling
        self.n_heads = n_heads
        self.n_tasks = n_tasks
        self.device = device

        # Define SE(3) fibers (degree to channel mapping)
        self.fibers = {
            'in': Fiber(1, atom_feature_size),
            'mid': Fiber(num_degrees, self.num_channels),
            'out': Fiber(1, num_degrees * self.num_channels)
        }

        # Build network blocks
        self.Gblock, self.FCblock = self._build_gcn(self.fibers, self.n_tasks)

    def _build_gcn(self, fibers: Dict[str, Fiber],
                   out_dim: int) -> Tuple[nn.ModuleList, nn.ModuleList]:
        """
        Build the SE(3)-equivariant GCN block and fully connected head.

        Parameters
        ----------
        fibers : dict
            Dictionary of input, intermediate, and output fibers.
        out_dim : int
            Output dimension of the final prediction layer.

        Returns
        -------
        Gblock : nn.ModuleList
            List of SE(3)-equivariant layers.
        FCblock : nn.ModuleList
            List of fully connected layers after pooling.
        """

        Gblock: List[LayerType] = []
        fin = fibers['in']
        for _ in range(self.num_layers):
            Gblock.append(
                SE3ResidualAttention(fin,
                                     fibers['mid'],
                                     edge_dim=self.edge_dim,
                                     div=self.div,
                                     n_heads=self.n_heads))
            Gblock.append(SE3GraphNorm(fibers['mid']))
            fin = fibers['mid']

        # Final SE(3) convolution
        Gblock.append(
            SE3GraphConv(fibers['mid'],
                         fibers['out'],
                         self_interaction=True,
                         edge_dim=self.edge_dim))

        # Pooling layer
        if self.pooling == 'avg':
            Gblock.append(SE3AvgPooling())
        elif self.pooling == 'max':
            Gblock.append(SE3MaxPooling())
        else:
            raise ValueError(f"Unsupported pooling type: {self.pooling}")

        # Fully connected layers
        FCblock = [
            nn.Linear(fibers['out'].n_features,
                      fibers['out'].n_features).float(),
            nn.ReLU(inplace=True),
            nn.Linear(fibers['out'].n_features, out_dim).float()
        ]

        return nn.ModuleList(Gblock), nn.ModuleList(FCblock)

    def forward(self, G: DGLGraph) -> Union[Dict[str, Any], torch.Tensor]:
        """
        Forward pass through the SE(3)-Transformer model.

        Parameters
        ----------
        G : DGLGraph
            Graph with node and edge features. Node features should be under `G.ndata['x']`.


        Returns
        -------
        torch.Tensor
            Output tensor of shape [batch_size, n_tasks].
        """
        basis, r = get_equivariant_basis_and_r(G,
                                               self.num_degrees - 1,
                                               compute_gradients=False)
        h: Union[Dict[str, Any], torch.Tensor] = {
            '0': G.ndata['x'].unsqueeze(-1)
        }

        for layer in self.Gblock:
            h = layer(h, G=G, r=r, basis=basis)

        for layer in self.FCblock:
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
    num_workers : int
        Number of DGL / DataLoader worker processes used for batching and featurization.
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
    >>> import shutil
    >>> import os
    >>> smiles = ["CCO", "CC(=O)O", "C1=CC=CC=C1",]
    >>> featurizer = dc.feat.EquivariantGraphFeaturizer(fully_connected=False, embeded=True)
    >>> mols_g = [featurizer.featurize(Chem.MolFromSmiles(mol))[0] for mol in smiles]
    >>> # Extract SE(3)-equivariant features
    >>> labels = np.random.rand(len(mols_g), 1)
    >>> weights = np.ones_like(labels)
    >>> dataset = dc.data.NumpyDataset(X=mols_g, y=labels, w=weights)
    >>> model = SE3TransformerModel(
    ...    num_layers=7,
    ...    atom_feature_size=6,
    ...    num_workers=4,
    ...    num_channels=32,
    ...    num_nlayers=1,
    ...    num_degrees=4,
    ...    edge_dim=4,
    ...    pooling='max',
    ...    n_heads=8,
    ...    batch_size=12,)
    >>> loss = model.fit(dataset, nb_epoch=1)
    >>> dir_path = "cache"
    >>> if os.path.exists(dir_path) and os.path.isdir(dir_path):
    ...     shutil.rmtree(dir_path)
    """

    def __init__(self,
                 num_layers: int,
                 atom_feature_size: int,
                 num_workers: int,
                 num_channels: int,
                 num_nlayers: int = 1,
                 num_degrees: int = 4,
                 edge_dim: int = 4,
                 pooling: str = 'avg',
                 n_heads: int = 1,
                 device: Optional[torch.device] = torch.device('cpu'),
                 **kwargs) -> None:
        if device is None:

            if torch.cuda.is_available():
                self.device = torch.device('cuda:0')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device
        output_types = ['prediction']
        loss = L1Loss()

        model = SE3Transformer(
            num_layers=num_layers,
            atom_feature_size=atom_feature_size,
            num_workers=num_workers,
            num_channels=num_channels,
            num_nlayers=num_nlayers,
            num_degrees=num_degrees,
            edge_dim=edge_dim,
            pooling=pooling,
            n_heads=n_heads,
            device=self.device,
        )
        super(SE3TransformerModel, self).__init__(model,
                                                  loss=loss,
                                                  device=self.device,
                                                  output_types=output_types,
                                                  **kwargs)

    def _prepare_batch(self, batch: Tuple[Any, Any,
                                          Any]) -> Tuple[List, List, List]:
        """
        Prepares a batch of data for the SE3Transformer model.

        This function processes DGL graphs from a DeepChem dataset batch,
        converting the graphs and labels into tensors suitable for training.

        Parameters
        ----------
        batch : Tuple[Any, Any, Any]
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

    def save(self):
        """Saves model to disk using joblib."""
        save_to_disk(self.model, self.get_model_filename(self.model_dir))

    def reload(self):
        """Loads model from joblib file on disk."""
        self.model = load_from_disk(self.get_model_filename(self.model_dir))
