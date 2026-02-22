
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

from MaceInteraction import EquivariantMACEInteractionClean
from RadialBasis import RadialBasis

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print(" All imports successful!")
print(f"   PyTorch version: {torch.__version__}")
print(f"   DeepChem version: {dc.__version__}")
print(f"   Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
print("MACE WITH CLEAN LAYERS")


class MACEClean(nn.Module):
    """Clean MACE without flatten issues."""

    def __init__(self, hidden_dim=128, num_interactions=4, num_basis=16, cutoff=5.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cutoff = cutoff

        # Embeddings
        self.atom_embedding = nn.Embedding(100, hidden_dim)

        # Radial basis
        self.radial_basis = RadialBasis(num_basis=num_basis, cutoff=cutoff)

        # Interactions
        self.interactions = nn.ModuleList([
            EquivariantMACEInteractionClean(hidden_dim, num_basis)
            for _ in range(num_interactions)
        ])

        # Readout
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, z, pos, edge_index, batch):
        # Initialize
        s = self.atom_embedding(z)
        v = torch.zeros(z.size(0), self.hidden_dim, 3, device=z.device)

        # Compute edge features
        row, col = edge_index
        edge_vec = pos[row] - pos[col]
        edge_dist = torch.norm(edge_vec, dim=-1, keepdim=True)
        edge_attr = self.radial_basis(edge_dist)
        edge_vec_normalized = edge_vec / (edge_dist + 1e-8)

        # Message passing
        for interaction in self.interactions:
            s, v = interaction(s, v, edge_index, edge_attr, edge_vec_normalized)

        # Pool and readout
        from torch_geometric.nn import global_add_pool
        s_pooled = global_add_pool(s, batch)
        energy = self.readout(s_pooled)

        return energy, v

print(" Clean MACE model ready")
