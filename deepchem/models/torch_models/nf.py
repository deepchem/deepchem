"""
Normalizing flows for transforming probability distributions with PyTorch.
"""

import numpy as np
import logging
from typing import List, Iterable, Optional, Tuple, Sequence, Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.distributions.multivariate_normal import MultivariateNormal

import deepchem as dc
from deepchem.models.losses import Loss
from deepchem.models.models import Model
from deepchem.models.torch_models import TorchModel
from deepchem.models.optimizers import Optimizer, Adam
from deepchem.utils.typing import OneOrMany
from deepchem.utils.data_utils import load_from_disk, save_to_disk

logger = logging.getLogger(__name__)

class Affine(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.scale = nn.Parameter(torch.zeros(self.dim))
        self.shift = nn.Parameter(torch.zeros(self.dim))

    def _forward(self, x):
        y = torch.exp(self.scale)*x +self.shift
        det_jacobian = torch.exp(self.scale.sum())
        log_det_jacobian = torch.ones(y.shape[0]) * torch.log(det_jacobian)

        return y, log_det_jacobian

    def _inverse(self, y):
        x = (y - self.shift)/torch.exp(self.scale)
        det_jacobian = 1/torch.exp(self.scale.sum())
        inverse_log_det_jacobian = torch.one(y.shape[0] * torch.log(det_jacobian))
        
        return x, inverse_log_det_jacobian

class NormalizingFlow(nn.Module):
    def __init__(self, transform, base_distribution, dim):
        super().__init__()
        self.dim = dim
        self.tranforms = nn.ModuleList(transform)
        self.base_distribution = base_distribution
    
    def _log_prob(self, inputs):
        log_prob = torch.zeros(inputs.shape[0])
        for biject in reversed(self.tranforms):
            inputs, inverse_log_det_jacobian = biject.inverse(inputs)
            log_prob += inverse_log_det_jacobian

        return log_prob
    
    def _sample(self, n_samples):
        outputs = self.base_distribution.sample((n_samples, ))
        log_prob = self.base_distribution.log_prob(outputs)

        for biject in self.transforms:
            output, log_det_jacobian = biject.forward(output)
            log_prob += log_det_jacobian

        return output, log_prob

class NormalizingFlowModel (TorchModel):
    def __init__(self, model: NormalizingFlow, **kwargs) -> None:
        self.nll_loss_fn = lambda input, labels, weights: self.create_nll(input)

        super(NormalizingFlowModel, self).__init__(
        model=model, loss=self.nll_loss_fn, **kwargs)

        self.flow = self.model.transforms
        # self.model = model

    def create_nll(self, input: OneOrMany[torch.Tensor]) -> torch.Tensor:
        
        log_prob = self.flow.log_probability(input, training = True)
        loss = -log_prob.mean(0)

        return loss

