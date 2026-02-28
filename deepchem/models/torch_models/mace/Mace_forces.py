# Imports
import deepchem as dc
import numpy as np
import time
from rdkit import Chem
from rdkit.Chem import AllChem
import pickle
import os

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
print(f" PyTorch version: {torch.__version__}")
print(f" DeepChem version: {dc.__version__}")
print(f" Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")



print(" MACE MODEL WITH FORCE COMPUTATION")


class MACEWithForcesFixed(nn.Module):
    """MACE model that properly handles gradients for forces."""

    def __init__(self, mace_model):
        super().__init__()
        self.mace = mace_model

    def forward(self, z, pos, edge_index, batch, compute_forces=True):
        """Forward pass maintaining gradients."""

        if compute_forces and not pos.requires_grad:
            pos = pos.requires_grad_(True)

        # Recompute edges WITH gradients
        row, col = edge_index
        edge_vec = pos[row] - pos[col]
        edge_dist = torch.norm(edge_vec, dim=-1, keepdim=True)

        # Get embeddings
        s = self.mace.atom_embedding(z)
        num_atoms = z.size(0)
        v = torch.zeros(num_atoms, self.mace.hidden_dim, 3, device=z.device)

        # Radial basis
        edge_attr = self.mace.radial_basis(edge_dist)
        edge_vec_normalized = edge_vec / (edge_dist + 1e-8)

        # Message passing
        for interaction in self.mace.interactions:
            s, v = interaction(s, v, edge_index, edge_attr, edge_vec_normalized)

        # Readout
        from torch_geometric.nn import global_add_pool
        s_pooled = global_add_pool(s, batch)
        energy = self.mace.readout(s_pooled)

        if not compute_forces:
            return energy, None

        # Compute forces
        forces = -torch.autograd.grad(
            outputs=energy.sum(),
            inputs=pos,
            create_graph=True,
            retain_graph=True,
            allow_unused=False
        )[0]

        return energy, forces

# Loss function
def combined_loss_with_force_labels(energy_pred, energy_target, forces_pred, forces_target,
                                     energy_weight=1.0, force_weight=100.0):
    energy_loss = nn.MSELoss()(energy_pred.squeeze(), energy_target.squeeze())
    force_loss = nn.MSELoss()(forces_pred, forces_target)
    total_loss = energy_weight * energy_loss + force_weight * force_loss
    return total_loss, energy_loss, force_loss

