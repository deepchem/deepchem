
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

class EquivariantMACEInteractionClean(MessagePassing):
    """Clean E(3)-equivariant interaction layer."""

    def __init__(self, hidden_dim: int, num_basis: int) -> None:
        super().__init__(aggr="add", node_dim=0)
        self.hidden_dim: int = hidden_dim

        # Scalar network
        self.scalar_message_net: nn.Module = nn.Sequential(
            nn.Linear(hidden_dim * 2 + num_basis, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 3),
        )

        # Vector network
        self.vector_message_net: nn.Module = nn.Linear(
            hidden_dim, hidden_dim * 2, bias=False
        )

        # Update network
        self.scalar_update_net: nn.Module = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        s: torch.Tensor,
        v: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_vec: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        s : torch.Tensor
            Scalar node features, shape (N, hidden_dim)
        v : torch.Tensor
            Vector node features, shape (N, hidden_dim, 3)
        edge_index : torch.Tensor
            Edge indices, shape (2, E)
        edge_attr : torch.Tensor
            Radial basis features, shape (E, num_basis)
        edge_vec : torch.Tensor
            Normalized edge vectors, shape (E, 3)

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Updated scalar and vector features.
        """

        row, col = edge_index

        # Gather features
        s_i: torch.Tensor = s[row]
        s_j: torch.Tensor = s[col]
        v_j: torch.Tensor = v[col]

        if edge_attr.dim() == 3:
            edge_attr = edge_attr.squeeze(-1)

        # Scalar message
        scalar_input: torch.Tensor = torch.cat(
            [s_i, s_j, edge_attr], dim=-1
        )
        scalar_out: torch.Tensor = self.scalar_message_net(scalar_input)

        msg, gate, filt = torch.chunk(scalar_out, 3, dim=-1)
        ds: torch.Tensor = msg * torch.sigmoid(gate)

        # Vector message
        vec_weights: torch.Tensor = self.vector_message_net(filt)
        w1, w2 = torch.chunk(vec_weights, 2, dim=-1)

        dv: torch.Tensor = (
            w1.unsqueeze(-1) * v_j
            + w2.unsqueeze(-1) * edge_vec.unsqueeze(1)
        )

        # Aggregate
        ds_agg: torch.Tensor = scatter(
            ds, row, dim=0, dim_size=s.size(0), reduce="add"
        )
        dv_agg: torch.Tensor = scatter(
            dv, row, dim=0, dim_size=v.size(0), reduce="add"
        )

        # Update
        s_out: torch.Tensor = s + self.scalar_update_net(
            torch.cat([s, ds_agg], dim=-1)
        )
        v_out: torch.Tensor = v + dv_agg

        return s_out, v_out



