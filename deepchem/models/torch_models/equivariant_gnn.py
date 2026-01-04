"""
Canonical Equivariant Graph Neural Network Model for DeepChem.

This module provides a user-friendly, end-to-end SE(3)-equivariant GNN model
built on DeepChem's existing equivariant infrastructure.
"""

from deepchem.models.torch_models import TorchModel
from deepchem.models.losses import L2Loss, SparseSoftmaxCrossEntropy
from deepchem.utils.data_utils import load_from_disk, save_to_disk
from typing import List, Optional, Union, Dict, Tuple, Any, Literal
import logging

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:
    raise ImportError('These classes require PyTorch to be installed.')

try:
    from deepchem.models.torch_models.layers import (
        SE3GraphConv, SE3ResidualAttention, SE3GraphNorm,
        SE3AvgPooling, SE3MaxPooling, Fiber
    )
    from deepchem.utils.equivariance_utils import get_equivariant_basis_and_r
except ImportError:
    pass

logger = logging.getLogger(__name__)


class EquivariantGNN(nn.Module):
    """
    SE(3)-Equivariant Graph Neural Network.

    A canonical, configurable equivariant GNN that supports both convolution-based
    and attention-based message passing. This model is designed to work with
    3D molecular structures and maintains equivariance under rotations and
    translations.

    The architecture consists of:
    1. Input embedding layer (SE3GraphConv)
    2. Configurable equivariant blocks (convolution or attention)
    3. Invariant pooling for graph-level predictions
    4. Fully connected output head

    Parameters
    ----------
    num_layers : int
        Number of equivariant message passing layers.
    atom_feature_size : int
        Dimensionality of input atomic features.
    num_channels : int
        Number of channels per degree in intermediate SE(3) tensors.
    n_tasks : int, optional (default=1)
        Number of output tasks (regression or classification targets).
    num_degrees : int, optional (default=4)
        Maximum degree (L) of SE(3) tensors. Higher values capture more
        angular information but increase computation.
    edge_dim : int, optional (default=4)
        Dimensionality of edge features.
    block_type : {'conv', 'attention'}, optional (default='attention')
        Type of equivariant block to use:
        - 'conv': SE3GraphConv layers (faster, simpler)
        - 'attention': SE3ResidualAttention layers (more expressive)
    pooling : {'avg', 'max'}, optional (default='avg')
        Global pooling strategy for graph-level predictions.
    n_heads : int, optional (default=1)
        Number of attention heads (only used when block_type='attention').
    div : float, optional (default=2.0)
        Channel division factor in attention layers.
    use_norm : bool, optional (default=True)
        Whether to apply SE(3)-equivariant normalization after each block.
    dropout : float, optional (default=0.0)
        Dropout rate for the output head.

    Examples
    --------
    >>> import torch
    >>> import dgl
    >>> from deepchem.models.torch_models.equivariant_gnn import EquivariantGNN
    >>> g = dgl.graph(([0, 1, 2], [1, 2, 0]))
    >>> g.ndata["x"] = torch.randn(3, 6)
    >>> g.edata["edge_attr"] = torch.randn(3, 3)
    >>> g.edata["w"] = torch.randn(3, 4)
    >>> model = EquivariantGNN(
    ...     num_layers=2,
    ...     atom_feature_size=6,
    ...     num_channels=16,
    ...     block_type='conv'
    ... )
    >>> out = model(g)
    >>> out.shape
    torch.Size([1, 1])

    References
    ----------
    .. [1] Fuchs et al. "SE(3)-Transformers: 3D Roto-Translation Equivariant
           Attention Networks." NeurIPS 2020.
    .. [2] Satorras et al. "E(n) Equivariant Graph Neural Networks." ICML 2021.
    """

    def __init__(
        self,
        num_layers: int,
        atom_feature_size: int,
        num_channels: int,
        n_tasks: int = 1,
        num_degrees: int = 4,
        edge_dim: int = 4,
        block_type: Literal['conv', 'attention'] = 'attention',
        pooling: Literal['avg', 'max'] = 'avg',
        n_heads: int = 1,
        div: float = 2.0,
        use_norm: bool = True,
        dropout: float = 0.0,
        **kwargs
    ):
        super().__init__()

        self.num_layers = num_layers
        self.num_channels = num_channels
        self.num_degrees = num_degrees
        self.edge_dim = edge_dim
        self.block_type = block_type
        self.pooling = pooling
        self.n_heads = n_heads
        self.div = div
        self.use_norm = use_norm
        self.n_tasks = n_tasks

        # Define SE(3) fibers
        self.fibers = {
            'in': Fiber(1, atom_feature_size),
            'mid': Fiber(num_degrees, num_channels),
            'out': Fiber(1, num_degrees * num_channels)
        }

        # Build network
        self.blocks = self._build_blocks()
        self.pooling_layer = self._build_pooling()
        self.output_head = self._build_output_head(dropout)

    def _build_blocks(self) -> nn.ModuleList:
        """Build the equivariant message passing blocks."""
        blocks: List[nn.Module] = []
        fin = self.fibers['in']

        for i in range(self.num_layers):
            fout = self.fibers['mid']

            if self.block_type == 'attention':
                blocks.append(
                    SE3ResidualAttention(
                        fin, fout,
                        edge_dim=self.edge_dim,
                        div=self.div,
                        n_heads=self.n_heads
                    )
                )
            else:  # conv
                blocks.append(
                    SE3GraphConv(
                        fin, fout,
                        self_interaction=True,
                        edge_dim=self.edge_dim
                    )
                )

            if self.use_norm:
                blocks.append(SE3GraphNorm(fout))

            fin = fout

        # Final projection to output fiber
        blocks.append(
            SE3GraphConv(
                self.fibers['mid'],
                self.fibers['out'],
                self_interaction=True,
                edge_dim=self.edge_dim
            )
        )

        return nn.ModuleList(blocks)

    def _build_pooling(self) -> nn.Module:
        """Build the invariant pooling layer."""
        if self.pooling == 'avg':
            return SE3AvgPooling()
        elif self.pooling == 'max':
            return SE3MaxPooling()
        else:
            raise ValueError(f"Unsupported pooling type: {self.pooling}")

    def _build_output_head(self, dropout: float) -> nn.Sequential:
        """Build the fully connected output head."""
        hidden_dim = self.fibers['out'].n_features
        return nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, self.n_tasks)
        )

    def forward(self, G) -> torch.Tensor:
        """
        Forward pass through the equivariant GNN.

        Parameters
        ----------
        G : dgl.DGLGraph
            Input graph with node features in G.ndata['x'] and edge features
            in G.edata['edge_attr']. Edge weights should be in G.edata['w'].

        Returns
        -------
        torch.Tensor
            Output predictions of shape [batch_size, n_tasks].
        """
        # Compute equivariant basis and distances
        basis, r = get_equivariant_basis_and_r(
            G, self.num_degrees - 1, compute_gradients=False
        )

        # Initialize node features as degree-0 fiber
        h: Dict[str, torch.Tensor] = {'0': G.ndata['x'].unsqueeze(-1)}

        # Apply equivariant blocks
        for layer in self.blocks:
            h = layer(h, G=G, r=r, basis=basis)

        # Pool to graph-level representation
        h = self.pooling_layer(h, G=G, r=r, basis=basis)

        # Apply output head
        out = self.output_head(h)

        return out


class EquivariantGNNModel(TorchModel):
    """
    DeepChem TorchModel wrapper for EquivariantGNN.

    This class provides a user-friendly interface for training and evaluating
    SE(3)-equivariant graph neural networks within DeepChem's standard workflow.
    It handles batching, loss computation, and device placement automatically.

    Parameters
    ----------
    num_layers : int
        Number of equivariant message passing layers.
    atom_feature_size : int
        Dimensionality of input atomic features.
    num_channels : int
        Number of channels per degree in intermediate SE(3) tensors.
    n_tasks : int, optional (default=1)
        Number of output tasks.
    num_degrees : int, optional (default=4)
        Maximum degree of SE(3) tensors.
    edge_dim : int, optional (default=4)
        Dimensionality of edge features.
    block_type : {'conv', 'attention'}, optional (default='attention')
        Type of equivariant block ('conv' or 'attention').
    pooling : {'avg', 'max'}, optional (default='avg')
        Global pooling strategy.
    n_heads : int, optional (default=1)
        Number of attention heads.
    mode : {'regression', 'classification'}, optional (default='regression')
        Task mode determining loss function.
    dropout : float, optional (default=0.0)
        Dropout rate for output head.
    device : torch.device, optional
        Device to run the model on.
    **kwargs
        Additional arguments passed to TorchModel.

    Examples
    --------
    >>> import numpy as np
    >>> import deepchem as dc
    >>> from rdkit import Chem
    >>> smiles = ["CCO", "CC(=O)O", "C1=CC=CC=C1"]
    >>> featurizer = dc.feat.EquivariantGraphFeaturizer(
    ...     fully_connected=False, embeded=True
    ... )
    >>> mols = [featurizer.featurize(Chem.MolFromSmiles(s))[0] for s in smiles]
    >>> labels = np.random.rand(len(mols), 1)
    >>> dataset = dc.data.NumpyDataset(X=mols, y=labels)
    >>> model = EquivariantGNNModel(
    ...     num_layers=2,
    ...     atom_feature_size=6,
    ...     num_channels=16,
    ...     block_type='conv',
    ...     batch_size=2
    ... )
    >>> loss = model.fit(dataset, nb_epoch=1)

    Notes
    -----
    This model expects inputs featurized with `EquivariantGraphFeaturizer`.
    The featurizer should be configured with `embeded=True` to generate
    3D coordinates.

    References
    ----------
    .. [1] Fuchs et al. "SE(3)-Transformers: 3D Roto-Translation Equivariant
           Attention Networks." NeurIPS 2020.
    """

    def __init__(
        self,
        num_layers: int,
        atom_feature_size: int,
        num_channels: int,
        n_tasks: int = 1,
        num_degrees: int = 4,
        edge_dim: int = 4,
        block_type: Literal['conv', 'attention'] = 'attention',
        pooling: Literal['avg', 'max'] = 'avg',
        n_heads: int = 1,
        mode: Literal['regression', 'classification'] = 'regression',
        dropout: float = 0.0,
        device: Optional[torch.device] = None,
        **kwargs
    ):
        # Set device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda:0')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device

        # Set loss based on mode
        self.mode = mode
        if mode == 'regression':
            loss = L2Loss()
        else:
            loss = SparseSoftmaxCrossEntropy()

        # Build model
        model = EquivariantGNN(
            num_layers=num_layers,
            atom_feature_size=atom_feature_size,
            num_channels=num_channels,
            n_tasks=n_tasks,
            num_degrees=num_degrees,
            edge_dim=edge_dim,
            block_type=block_type,
            pooling=pooling,
            n_heads=n_heads,
            dropout=dropout
        )

        super(EquivariantGNNModel, self).__init__(
            model,
            loss=loss,
            device=self.device,
            output_types=['prediction'],
            **kwargs
        )

    def _prepare_batch(
        self, batch: Tuple[Any, Any, Any]
    ) -> Tuple[Any, List, List]:
        """
        Prepare a batch of data for the model.

        Parameters
        ----------
        batch : tuple
            A batch of (X, y, w) from the dataset.

        Returns
        -------
        tuple
            Prepared (inputs, labels, weights) for model training.
        """
        try:
            import dgl
        except ModuleNotFoundError:
            raise ImportError('This model requires DGL to be installed.')

        inputs, labels, weights = batch

        # Convert GraphData objects to DGL graphs
        dgl_graphs = []
        for graph in inputs[0]:
            dgl_g = graph.to_dgl_graph()
            dgl_g.edata['w'] = torch.tensor(
                graph.edge_weights, dtype=torch.float32
            )
            dgl_graphs.append(dgl_g)

        # Batch graphs
        batched_graph = dgl.batch(dgl_graphs).to(self.device)

        # Prepare labels and weights
        _, labels, weights = super(EquivariantGNNModel, self)._prepare_batch(
            ([], labels, weights)
        )

        return batched_graph, labels, weights

    def save(self):
        """Save model to disk."""
        save_to_disk(self.model, self.get_model_filename(self.model_dir))

    def reload(self):
        """Load model from disk."""
        self.model = load_from_disk(self.get_model_filename(self.model_dir))
