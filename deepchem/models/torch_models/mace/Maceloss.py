
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

class MACELoss(Loss):
    """MSE loss for MACE model.

    Computes mean squared error between predicted
    and true molecular energies.
    """

    def __init__(self) -> None:
        super().__init__()

    def _create_pytorch_loss(self) -> torch.nn.Module:
        """Create and return PyTorch  MSE loss module."""
        return torch.nn.MSELoss()



