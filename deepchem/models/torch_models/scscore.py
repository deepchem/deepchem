from typing import Callable, List, Optional, Union
import numpy as np
import torch
from deepchem.data import NumpyDataset
from deepchem.feat import CircularFingerprint
from deepchem.models import TorchModel
from deepchem.models.losses import HingeLoss, Loss
import torch.nn as nn
import torch.nn.functional as F


class ScScore(nn.Module):
    """
    Builder class for ScScore Model.

    The SCScore model is a feed forward neural network model based on the work of Coley et al. [1]_ that predicts the synthetic complexity score (SCScore) of molecules and correlates it with the expected number of reaction steps required to produce the given target molecule.
    It is trained on a dataset of over 12 million reactions from the Reaxys database to impose a pairwise inequality constraint enforcing that on average the products of published chemical reactions should be more synthetically complex than their corresponding reactants.
    The learned metric (SCScore) exhibits highly desirable nonlinear behavior, particularly in recognizing increases in synthetic complexity throughout a number of linear synthetic routes.
    """

    def __init__(self, n_features=1024, layer_sizes=[300, 300, 300, 300, 300], dropout=0.0, score_scale=5):
        super(ScScore, self).__init__()
        self.dropout = dropout
        self.score_scale = score_scale
        self.layer_sizes = layer_sizes
        self.n_features = n_features

        input_size = self.layer_sizes[0]
        self.input_layer = nn.Linear(self.n_features, input_size)

        self.hidden_layers = nn.ModuleList()
        for layer_size in self.layer_sizes[1:]:
            self.hidden_layers.append(nn.Linear(input_size, layer_size))
            input_size = layer_size
        
        self.output_layer = nn.Linear(self.layer_sizes[-1], 1)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        if self.dropout > 0.0:
            x = F.dropout(x, p=self.dropout)
        
        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x))

            if self.dropout > 0.0:
                x = F.dropout(x, p=self.dropout)
        
        output = F.sigmoid(self.output_layer(x))
        
        output = 1 + (self.score_scale - 1) * output
        return output

class ScScoreModel(TorchModel):
    """
    The SCScore model is a neural network model based on the work of Coley et al. [1]_ that predicts the synthetic complexity score (SCScore) of molecules and correlates it with the expected number of reaction steps required to produce the given target molecule.
    It is trained on a dataset of over 12 million reactions from the Reaxys database to impose a pairwise inequality constraint enforcing that on average the products of published chemical reactions should be more synthetically complex than their corresponding reactants.
    The learned metric (SCScore) exhibits highly desirable nonlinear behavior, particularly in recognizing increases in synthetic complexity throughout a number of linear synthetic routes.
    The SCScore model can accurately predict the synthetic complexity of a variety of molecules, including both drug-like and natural product molecules.
    SCScore has the potential to be a valuable tool for chemists who are working on drug discovery and other areas of chemistry.

    The learned metric (SCScore) exhibits highly desirable nonlinear behavior, particularly in recognizing increases in synthetic complexity throughout a number of linear synthetic routes.

    Our model uses hingeloss instead of the shifted relu loss as in the supplementary material [2]_ provided by the author.
    This could cause differentiation issues with compounds that are "close" to each other in "complexity".

    References
    ----------
    .. [1] Coley, C. W., Rogers, L., Green, W., & Jensen, K. F. (2018). "SCScore: Synthetic Complexity Learned from a Reaction Corpus". Journal of Chemical Information and Modeling, 58(2), 252-261. https://doi.org/10.1021/acs.jcim.7b00622

    .. [2] Coley, C. W., Rogers, L., Green, W., & Jensen, K. F. (2018). Supplementary material to "SCScore: Synthetic Complexity Learned from a Reaction Corpus". Journal of Chemical Information and Modeling, 58(2), 252-261. https://github.com/connorcoley/scscore
    """

    def __init__(self, 
                 n_features=1024, 
                 layer_sizes=[300,300,300,300], 
                 dropout=0.0, 
                 score_scale=5, 
                 **kwargs):
        """
        Parameters
        ----------
        n_features: int
            number of features per molecule
        layer_sizes: list of int
            size of each hidden layer
        dropouts: float
            droupout to apply to each hidden layer
        kwargs
            This takes all kwargs as TensorGraph
        """
        
        model = ScScore(n_features, layer_sizes, dropout, score_scale)
        loss = HingeLoss()
        super(ScScoreModel, self).__init__(model, 
                                           loss=loss, 
                                           **kwargs)
    
