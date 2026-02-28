
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print(" All imports successful!")
print(f"   PyTorch version: {torch.__version__}")
print(f"   DeepChem version: {dc.__version__}")
print(f"   Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")



class EquivariantMACEInteractionClean(MessagePassing):
    """Clean E(3)-equivariant layer - FIXED."""

    def __init__(self, hidden_dim: int, num_basis: int):
        super().__init__(aggr='add', node_dim=0)
        self.hidden_dim = hidden_dim

        # Scalar network
        self.scalar_message_net = nn.Sequential(
            nn.Linear(hidden_dim * 2 + num_basis, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 3)
        )

        # Vector network
        self.vector_message_net = nn.Linear(hidden_dim, hidden_dim * 2, bias=False)

        # Update
        self.scalar_update_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, s, v, edge_index, edge_attr, edge_vec):
        # Manual message passing
        row, col = edge_index

        # Gather features
        s_i = s[row]
        s_j = s[col]
        v_j = v[col]

        # FIX: Squeeze edge_attr if it has extra dimension
        if edge_attr.dim() == 3:
            edge_attr = edge_attr.squeeze(-1)

        # Scalar message
        scalar_input = torch.cat([s_i, s_j, edge_attr], dim=-1)
        scalar_out = self.scalar_message_net(scalar_input)
        msg, gate, filt = torch.chunk(scalar_out, 3, dim=-1)
        ds = msg * torch.sigmoid(gate)

        # Vector message
        vec_weights = self.vector_message_net(filt)
        w1, w2 = torch.chunk(vec_weights, 2, dim=-1)
        dv = w1.unsqueeze(-1) * v_j + w2.unsqueeze(-1) * edge_vec.unsqueeze(1)

        # Aggregate
        from torch_geometric.utils import scatter
        ds_agg = scatter(ds, row, dim=0, dim_size=s.size(0), reduce='add')
        dv_agg = scatter(dv, row, dim=0, dim_size=v.size(0), reduce='add')

        # Update
        s_out = s + self.scalar_update_net(torch.cat([s, ds_agg], dim=-1))
        v_out = v + dv_agg

        return s_out, v_out


