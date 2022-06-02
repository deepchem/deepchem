"""
Normalizing flows for transforming probability distributions using PyTorch.
"""

import numpy as np
import logging
from typing import List, Iterable, Optional, Tuple, Sequence, Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

import deepchem as dc
from deepchem.models.losses import Loss
from deepchem.models.models import Model
from deepchem.models.torch_models import TorchModel
from deepchem.models.optimizers import Optimizer, Adam
from deepchem.utils.typing import OneOrMany
from deepchem.utils.data_utils import load_from_disk, save_to_disk

logger = logging.getLogger(__name__)

class Affine(nn.Module):
  """Class which performs the Affine transformation.

  This transformation is based on the affinity of the base distribution with
  the target distribution. A geometric transformation is applied where
  the parameters performs changes on the scale and shift of a function
  (inputs).
  
  Normalizing Flow transformations must be bijective in order to compute
  the logartihm of jacobian's determinat. For this reason, transformations
  must perform a forward and inverse pass.

  """

  def __init__(self, dim: int)-> None:
      """Create a Affine transform layer.

      Parameters
      ----------
      dim: int
        Value of the Nth dimenssion of the dataset. 

      """

      super().__init__()
      self.dim = dim
      self.scale = nn.Parameter(torch.zeros(self.dim))
      self.shift = nn.Parameter(torch.zeros(self.dim))

  def forward(self, x):
      y = torch.exp(self.scale)*x +self.shift
      det_jacobian = torch.exp(self.scale.sum())
      log_det_jacobian = torch.ones(y.shape[0]) * torch.log(det_jacobian)

      return y, log_det_jacobian

  def inverse(self, y):
      x = (y - self.shift)/torch.exp(self.scale)
      det_jacobian = 1/torch.exp(self.scale.sum())
      inverse_log_det_jacobian = torch.ones(y.shape[0]) * torch.log(det_jacobian)
        
      return x, inverse_log_det_jacobian