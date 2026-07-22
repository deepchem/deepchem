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
from torch_geometric.nn import global_add_pool
from typing import Tuple, Optional


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


class MACEWithForcesFixed(nn.Module):
    """
Examples
--------
>>> import torch
>>> from deepchem.models.torch_models.mace import MACEWithForcesFixed
>>> 
>>> base_model = MACEClean(hidden_dim=64, num_interactions=3)
>>> model = MACEWithForcesFixed(base_model)
>>> 
>>> z = torch.tensor([6, 1, 1, 1, 1])
>>> pos = torch.randn(5, 3, requires_grad=True)
>>> edge_index = torch.tensor([[0, 0, 0, 0],
...                            [1, 2, 3, 4]])
>>> batch = torch.zeros(5, dtype=torch.long)
>>> 
>>> energy, forces = model(z, pos, edge_index, batch, compute_forces=True)
"""

    def __init__(self, mace_model: nn.Module) -> None:
        super().__init__()
        self.mace = mace_model

    def forward(
        self,
        z: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        compute_forces: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass maintaining gradients."""

        if compute_forces and not pos.requires_grad:
            pos = pos.requires_grad_(True)

        # Recompute edges with gradients
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
def combined_loss_with_force_labels(
    energy_pred: torch.Tensor,
    energy_target: torch.Tensor,
    forces_pred: torch.Tensor,
    forces_target: torch.Tensor,
    energy_weight: float = 1.0,
    force_weight: float = 100.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    energy_loss = nn.MSELoss()(energy_pred.squeeze(), energy_target.squeeze())
    force_loss = nn.MSELoss()(forces_pred, forces_target)
    total_loss = energy_weight * energy_loss + force_weight * force_loss

    return total_loss, energy_loss, force_loss


