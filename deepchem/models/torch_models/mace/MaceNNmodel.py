# Core imports
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter

# RDKit
from rdkit import Chem
from rdkit.Chem import AllChem

# DeepChem
import deepchem as dc
from deepchem.models import TorchModel
from deepchem.models.losses import Loss
from deepchem.data import Dataset, NumpyDataset

# Typing
from typing import Tuple, Optional, List, Dict, Any, Sequence, Iterator

# Utilities
import logging
import time
import pickle
import math
from tqdm import tqdm
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

from .MaceInteraction import EquivariantMACEInteractionClean
from .RadialBasis import RadialBasis


class MACEClean(nn.Module):
    """
    Examples
    --------
    >>> import torch
    >>> from deepchem.models.torch_models.mace import MACEClean
    >>> 
    >>> model = MACEClean(hidden_dim=64, num_interactions=3)
    >>> 
    >>> z = torch.tensor([6, 1, 1, 1, 1])  # atomic numbers
    >>> pos = torch.randn(5, 3)            # positions
    >>> edge_index = torch.tensor([[0, 0, 0, 0],
    ...                            [1, 2, 3, 4]])
    >>> batch = torch.zeros(5, dtype=torch.long)
    >>> 
    >>> energy, _ = model(z, pos, edge_index, batch)
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_interactions: int = 4,
        num_basis: int = 16,
        cutoff: float = 5.0
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cutoff = cutoff

        # Atom embedding
        self.atom_embedding = nn.Embedding(100, hidden_dim)

        # Radial basis
        self.radial_basis = RadialBasis(num_basis=num_basis, cutoff=cutoff)

        # Interaction layers
        self.interactions = nn.ModuleList([
            EquivariantMACEInteractionClean(hidden_dim, num_basis)
            for _ in range(num_interactions)
        ])

        # Readout network
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(
        self,
        z: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # Initial scalar and vector features
        s = self.atom_embedding(z)  # (N, hidden_dim)
        v = torch.zeros(
            z.size(0),
            self.hidden_dim,
            3,
            device=z.device
        )  # (N, hidden_dim, 3)

        # Edge features
        row, col = edge_index
        edge_vec = pos[row] - pos[col]  # (E, 3)
        edge_dist = torch.norm(edge_vec, dim=-1, keepdim=True)  # (E, 1)

        edge_attr = self.radial_basis(edge_dist)  # (E, num_basis)
        edge_vec_normalized = edge_vec / (edge_dist + 1e-8)

        # Message passing
        for interaction in self.interactions:
            s, v = interaction(
                s,
                v,
                edge_index,
                edge_attr,
                edge_vec_normalized
            )

        # Pooling
        from torch_geometric.nn import global_add_pool
        s_pooled = global_add_pool(s, batch)

        # Energy prediction
        energy = self.readout(s_pooled)

        return energy, v