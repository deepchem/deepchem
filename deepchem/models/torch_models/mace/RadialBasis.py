
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


class RadialBasis(nn.Module):
    """Bessel radial basis functions - FIXED OUTPUT SHAPE."""

    def __init__(self, num_basis: int = 8, cutoff: float = 5.0):
        super().__init__()
        self.num_basis = num_basis
        self.cutoff = cutoff

        # Bessel frequencies
        frequencies = torch.pi * torch.arange(1, num_basis + 1)
        self.register_buffer('frequencies', frequencies)

    def forward(self, distances):
        """
        Args:
            distances: (E, 1) edge distances

        Returns:
            basis: (E, num_basis) - FIXED: No extra dimension!
        """
        # Cutoff envelope
        envelope = 1.0 - torch.sigmoid(5.0 * (distances - self.cutoff))

        # Bessel basis
        d = distances  # (E, 1)
        basis = torch.sin(self.frequencies * d) / d  # (E, 1) * (num_basis,) -> (E, num_basis)

        # Apply envelope and SQUEEZE
        output = basis * envelope  # (E, num_basis)

        return output  # (E, num_basis) - 2D output!

