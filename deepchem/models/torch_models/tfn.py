from deepchem.models.torch_models import TorchModel
from deepchem.models.losses import L1Loss
from typing import List, Optional, Union, Dict, Tuple, Any
from deepchem.models.torch_models.layers import SE3GraphConv, SE3GraphNorm, Fiber, SE3MaxPooling
from deepchem.utils.equivariance_utils import get_equivariant_basis_and_r
from deepchem.utils.data_utils import load_from_disk, save_to_disk
import logging
try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:
    raise ImportError('These classes require PyTorch to be installed.')

logger = logging.getLogger(__name__)
LayerType = Union[SE3GraphConv, SE3GraphNorm, SE3MaxPooling, nn.Module]


class TFN(nn.Module):
    """
    Tensor Field Networks (TFNs) are neural networks designed to process 3D geometric
    data by ensuring rotational and translational equivariance using tensor representations
    aligned with SE(3) symmetry group.

    TFNs are designed to be equivariant under 3D rigid body motions (rotations and translations),
    making it particularly suitable for molecular modeling, protein design, and other physics-informed tasks.

    Parameters
    ----------
    num_layers : int
        Number of equivariant layers in the GCN.
    atom_feature_size : int
        Dimensionality of the input atomic features.
    num_channels : int
        Number of channels per degree in intermediate SE(3) tensors.
    n_tasks : int, optional
        Number of regression/classification tasks (default is 1).
    num_nlayers : int, optional
        Number of normalization layers GCN (default is 1).
    num_degrees : int, optional
        Maximum degree (L) of the SE(3) tensors (default is 4).
    edge_dim : int, optional
        Dimensionality of edge features (default is 4).
    device : torch.device, optional
        Device to run the model on (default is CPU).

    Example
    -------
    >>> import torch
    >>> import dgl
    >>> from deepchem.models.torch_models import TFN
    >>> g = dgl.graph(([0, 1, 2], [1, 2, 0]))   # simple 3-node cycle
    >>> g.ndata["x"] = torch.randn(3, 6)        # node features
    >>> g.edata["edge_attr"] = torch.randn(3, 3)
    >>> g.edata["w"] = torch.randn(3, 4)        # matches edge_dim=4
    >>> model = TFN(
    ...     num_layers=2,
    ...     atom_feature_size=6,
    ...     num_channels=8,
    ...     n_tasks=1,
    ...     num_nlayers=1,
    ...     num_degrees=3,
    ...     edge_dim=4,
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
                 device: Optional[torch.device] = torch.device('cpu'),
                 **kwargs):
        super().__init__()

        # Store hyperparameters
        self.num_layers = num_layers
        self.num_nlayers = num_nlayers
        self.num_channels = num_channels
        self.num_degrees = num_degrees
        self.edge_dim = edge_dim
        self.n_tasks = n_tasks
        self.device = device
        self.num_channels_out = num_channels * num_degrees

        # Define SE(3) fibers (degree to channel mapping)
        self.fibers = {
            'in': Fiber(1, atom_feature_size),
            'mid': Fiber(num_degrees, self.num_channels),
            'out': Fiber(1, self.num_channels_out)
        }

        # Build network blocks
        blocks = self._build_gcn(self.fibers, self.n_tasks)
        self.equiv_graph_conv_block, self.pool_block, self.fc_block = blocks

    def _build_gcn(
            self, fibers: Dict[str, Fiber],
            out_dim: int) -> Tuple[nn.ModuleList, nn.ModuleList, nn.ModuleList]:
        """
        Build the Tensor field networks block.

        Parameters
        ----------
        fibers : dict
            Dictionary of input, intermediate, and output fibers.
        out_dim : int
            Output dimension of the final prediction layer.

        Returns
        -------
        equiv_graph_conv_block : nn.ModuleList
            List of graph convolution SE(3)-equivariant layers.
        pool_block : nn.ModuleList
            List of SE(3) max pooling layer.
        FCblock : nn.ModuleList
            List of fully connected layers after pooling.
        """

        equiv_graph_conv_block: List[LayerType] = []
        fin = fibers['in']
        for i in range(self.num_layers - 1):
            equiv_graph_conv_block.append(
                SE3GraphConv(fin,
                             fibers['mid'],
                             self_interaction=True,
                             edge_dim=self.edge_dim))
            equiv_graph_conv_block.append(
                SE3GraphNorm(fibers['mid'], num_layers=self.num_nlayers))
            fin = fibers['mid']
        equiv_graph_conv_block.append(
            SE3GraphConv(fibers['mid'],
                         fibers['out'],
                         self_interaction=True,
                         edge_dim=self.edge_dim))

        pool_block = [SE3MaxPooling()]

        fc_block: List[Union[nn.Linear, nn.ReLU]] = []
        fc_block.append(nn.Linear(self.num_channels_out, self.num_channels_out))
        fc_block.append(nn.ReLU(inplace=True))
        fc_block.append(nn.Linear(self.num_channels_out, out_dim))

        return nn.ModuleList(equiv_graph_conv_block), nn.ModuleList(
            pool_block), nn.ModuleList(fc_block)

    def forward(self, G: DGLGraph) -> Union[Dict[str, Any], torch.Tensor]:
        """
        Forward pass through the TFN model.

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
        h = {
            '0': G.ndata['x'].unsqueeze(-1)
        }  # inject input features as degree-0

        for layer in self.equiv_graph_conv_block:
            h = layer(h, G=G, r=r, basis=basis)

        for layer in self.pool_block:
            h = layer(h, G)

        for layer in self.fc_block:
            h = layer(h)

        return h


class TFNModel(TorchModel):
    """
    TFNModel Deepchem Wrapper.

    This class wraps the TFN model for compatibility with DeepChem's model interface.

    Parameters
    ----------
    num_layers : int
        Number of SE(3) graph convolution layers.
    atom_feature_size : int
        Dimensionality of atom features.
    num_channels : int
        Number of channels for the hidden layers.
    num_nlayers : int, optional (default=1)
        Number of normalization layers for graph convolution block.
    num_degrees : int, optional (default=4)
        Degree of SE(3)-equivariant features.
    edge_dim : int, optional (default=4)
        Dimensionality of edge features.
    device : torch.device, optional
        The device (CPU or GPU) on which the model will run.
    **kwargs :
        Additional arguments for TorchModel.

    Example
    -------
    >>> import torch
    >>> from deepchem.models.torch_models import TFNModel
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
    >>> labels = np.random.rand(len(mols_g), 12)
    >>> weights = np.ones_like(labels)
    >>> dataset = dc.data.NumpyDataset(X=mols_g, y=labels, w=weights)
    >>> model = TFNModel(
    ...    num_layers=7,
    ...    atom_feature_size=6,
    ...    num_channels=32,
    ...    num_nlayers=1,
    ...    num_degrees=4,
    ...    edge_dim=4,
    ...    batch_size=12,)
    >>> loss = model.fit(dataset, nb_epoch=1)
    >>> dir_path = "cache"
    >>> if os.path.exists(dir_path) and os.path.isdir(dir_path):
    ...     shutil.rmtree(dir_path)
    """

    def __init__(self,
                 num_layers: int,
                 atom_feature_size: int,
                 num_channels: int,
                 num_nlayers: int = 1,
                 num_degrees: int = 4,
                 edge_dim: int = 4,
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

        # Initialize TFN model
        model = TFN(
            num_layers=num_layers,
            atom_feature_size=atom_feature_size,
            num_channels=num_channels,
            num_nlayers=num_nlayers,
            num_degrees=num_degrees,
            edge_dim=edge_dim,
            device=self.device,
        )
        super(TFNModel, self).__init__(model,
                                       loss=loss,
                                       device=self.device,
                                       output_types=output_types,
                                       **kwargs)

    def _prepare_batch(self, batch: Tuple[Any, Any,
                                          Any]) -> Tuple[List, List, List]:
        """
        Prepares a batch of data for the TFN model.

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
        _, labels, weights = super(TFNModel, self)._prepare_batch(
            ([], labels, weights))

        return inputs, labels, weights

    def save(self):
        """Saves model to disk using joblib."""
        save_to_disk(self.model, self.get_model_filename(self.model_dir))

    def reload(self):
        """Loads model from joblib file on disk."""
        self.model = load_from_disk(self.get_model_filename(self.model_dir))
