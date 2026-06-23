
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

class MACEWithForcesClean(nn.Module):
    """Force prediction wrapper for clean MACE."""

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

        if compute_forces:
            pos.requires_grad_(True)

        energy, _ = self.mace(z, pos, edge_index, batch)

        if not compute_forces:
            return energy, None

        forces = -torch.autograd.grad(
            outputs=energy.sum(),
            inputs=pos,
            create_graph=True,
            retain_graph=True
        )[0]

        return energy, forces
