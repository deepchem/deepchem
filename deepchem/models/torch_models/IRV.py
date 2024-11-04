import deepchem as dc
from deepchem.models.torch_models.torch_model import TorchModel
from deepchem.models.losses import SigmoidCrossEntropy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np    
from typing import List, Union, Callable, Optional

device = torch.device("mps" if torch.has_mps else "cpu") 

class Slice(nn.Module):
    """ Choose a slice of input on the last axis given order,
    Suppose input x has two dimensions,
    output f(x) = x[:, slice_num:slice_num+1]
    """

    def __init__(self, slice_num: int, axis=1):
        """
        Parameters
        ----------
        slice_num: int
            index of slice number
        axis: int
            axis id
        """
        super(Slice, self).__init__()
        self.slice_num = slice_num
        self.axis = axis

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        slice_num = self.slice_num
        axis = self.axis
        return inputs.index_select(axis, torch.tensor([slice_num], device=device))

class IRVLayer(nn.Module):
    """ Core layer of IRV classifier, architecture described in:
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2750043/
    
    The paper introduces the Influence Relevance Voter (IRV), a novel machine learning model 
    designed for virtual high-throughput screening (vHTS).vHTS predicts the biological activity 
    of chemical compounds using computational methods, reducing the need for expensive experimental
    screening.
    
    The IRV model extends the k-Nearest Neighbors (kNN) algorithm by improving how neighbors 
    influence predictions. Instead of treating all neighbors equally, IRV assigns each neighbor
    a relevance score based on its similarity to the query compound.This similarity is calculated
    using molecular fingerprint comparisons between the query compound and its neighbors,allowing
    more relevant neighbors to have a greater impact on the prediction.
    """

    def __init__(self, n_tasks, K, penalty):
        """
        Parameters
        ----------
        n_tasks: int
            Number of tasks
        K: int
            Number of nearest neighbours used in classification
        penalty: float
            Amount of penalty (l2 or l1 applied)
        """
        super(IRVLayer, self).__init__()
        self.n_tasks = n_tasks
        self.K = K
        self.penalty = penalty

        # Initialize weights and biases
        self.V = nn.Parameter(torch.tensor([0.01, 1.], dtype=torch.float32))
        self.W = nn.Parameter(torch.tensor([1., 1.], dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor([0.01], dtype=torch.float32))
        self.b2 = nn.Parameter(torch.tensor([0.01], dtype=torch.float32))

    def forward(self, inputs: torch.Tensor ) -> List[torch.Tensor]:
        K = self.K
        outputs = []
        for count in range(self.n_tasks):
            # Similarity values
            similarity = inputs[:, 2 * K * count:(2 * K * count + K)]
            # Labels for all top K similar samples
            ys = inputs[:, (2 * K * count + K):2 * K * (count + 1)].int()

            R = self.b + self.W[0] * similarity + self.W[1] * torch.arange(1, K + 1, dtype=torch.float32, device=device)
            R = torch.sigmoid(R)
            z = torch.sum(R * self.V[ys], dim=1) + self.b2
            outputs.append(z.view(-1, 1))
        outputs= torch.cat(outputs, dim=1)
       
        logits = []
        predictions=outputs
        outputs=[]
        for task in range(self.n_tasks):
            task_output = Slice(task, 1)(predictions)
            sigmoid = torch.sigmoid(task_output)
            logits.append(task_output)
            outputs.append(sigmoid)
        outputs = torch.stack(outputs, dim=1)
        outputs2 = 1 - outputs
        outputs = [
            torch.cat([outputs2, outputs], dim=2),
            logits[0] if len(logits) == 1 else torch.cat(logits, dim=1)
        ]
        return outputs

class MultitaskIRVClassifier(TorchModel):

    def __init__(self, n_tasks, K=10, penalty=0.0, mode="classification", **kwargs):
        """Initialize MultitaskIRVClassifier

        Parameters
        ----------
        n_tasks: int
            Number of tasks
        K: int
            Number of nearest neighbours used in classification
        penalty: float
            Amount of penalty (l2 or l1 applied)
        """
        #super(MultitaskIRVClassifier, self).__init__()
        self._built = False
        self.n_tasks = n_tasks
        self.K = K
        self.n_features = 2 * self.K * self.n_tasks
        self.penalty = penalty
        
        # Define the IRVLayer
        self.irv_layer = IRVLayer(self.n_tasks, self.K, self.penalty).to(device)
        
        # Define the Slice layers
        self._built = True
        self.model = self
        
        regularization_loss: Optional[Callable]
        if self.penalty != 0.0:
            weights = list(self.irv_layer.parameters())
            regularization_loss = lambda: self.penalty * torch.sum(  # noqa: E731
                    torch.stack([torch.square(w).sum() for w in weights]))
        else:
            regularization_loss = None

        super(MultitaskIRVClassifier, self).__init__(self.irv_layer,
                                           loss=SigmoidCrossEntropy(),
                                           output_types=['prediction', 'loss'],
                                           regularization_loss=regularization_loss,
                                           **kwargs)
        
    def forward(self, mol_features: torch.Tensor) -> List[torch.Tensor]:
        mol_features = mol_features.to(device)
        outputs = self.irv_layer(mol_features)
        return outputs

import warnings  # noqa: E402


class TensorflowMultitaskIRVClassifier(MultitaskIRVClassifier):

    def __init__(self, *args, **kwargs):

        warnings.warn(
            "TensorflowMultitaskIRVClassifier is deprecated and has been renamed to MultitaskIRVClassifier",
            FutureWarning)

        super(TensorflowMultitaskIRVClassifier, self).__init__(*args, **kwargs)