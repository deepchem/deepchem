from deepchem.models.torch_models.torch_model import TorchModel
from deepchem.models.losses import SigmoidCrossEntropy

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class IRVLayer(nn.Module):
    """ 
    Core layer of IRV classifier, architecture described in:
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


    def forward(self, inputs):
        K = self.K
        outputs = []
        for count in range(self.n_tasks):
            # Similarity values
            similarity = inputs[:, 2 * K * count:(2 * K * count + K)]
            # Labels for all top K similar samples
            ys = inputs[:, (2 * K * count + K):2 * K * (count + 1)].int()

            R = self.b + self.W[0] * similarity + self.W[1] * torch.arange(1, K + 1, dtype=torch.float32)
            R = torch.sigmoid(R)
            z = torch.sum(R * self.V[ys], dim=1) + self.b2
            outputs.append(z.view(-1, 1))
        
            l2_loss = (torch.sum(self.W ** 2) / 2 + torch.sum(self.V ** 2) / 2 +
                    torch.sum(self.b ** 2) / 2 + torch.sum(self.b2 ** 2) / 2) * self.penalty

        self.add_loss(l2_loss)
        
        return torch.cat(outputs, dim=1)
    
    def add_loss(self, loss):
        if not hasattr(self, 'loss'):
            self.loss = loss
        else:
            self.loss += loss

class Slice(nn.Module):
    """ Choose a slice of input on the last axis given order,
    Suppose input x has two dimensions,
    output f(x) = x[:, slice_num:slice_num+1]
    """

    def __init__(self, slice_num, axis=1):
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

    def forward(self, inputs):
        slice_num = self.slice_num
        axis = self.axis
        return inputs.index_select(axis, torch.tensor([slice_num]))
    
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
        
        self.n_tasks = n_tasks
        self.K = K
        self.n_features = 2 * self.K * self.n_tasks
        self.penalty = penalty

        self.irv_layer = IRVLayer(self.n_tasks, self.K, self.penalty)
        self.slices = nn.ModuleList([Slice(task, 1) for task in range(self.n_tasks)])
    
        predictions = self.irv_layer()
        logits = []
        outputs = []
        for task in range(self.n_tasks):
            task_output = self.slices[task](predictions)
            sigmoid = torch.sigmoid(task_output)
            logits.append(task_output)
            outputs.append(sigmoid)
        outputs = torch.stack(outputs, dim=1)
        outputs2 = 1 - outputs
        outputs = [
            torch.cat([outputs2, outputs], dim=2),
            logits[0] if len(logits) == 1 else torch.cat(logits, dim=1)
        ]
        
        super(MultitaskIRVClassifier, self).__init__(self.IRVLayer,
                                           loss=SigmoidCrossEntropy(),
                                           output_types=['prediction', 'loss'],
                                           **kwargs)

   
import warnings  # noqa: E402


class TensorflowMultitaskIRVClassifier(MultitaskIRVClassifier):

    def __init__(self, *args, **kwargs):

        warnings.warn(
            "TensorflowMultitaskIRVClassifier is deprecated and has been renamed to MultitaskIRVClassifier",
            FutureWarning)

        super(TensorflowMultitaskIRVClassifier, self).__init__(*args, **kwargs)
