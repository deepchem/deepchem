import torch.nn as nn
import torch
from deepchem.models.torch_models import TorchModel
from deepchem.models.losses import SigmoidCrossEntropy
from deepchem.models.optimizers import Adam
from deepchem.models.optimizers import ExponentialDecay
from typing import List
from deepchem.models.torch_models import MultilayerPerceptron
import random
import numpy as np


class ActinnClassifier(nn.Module):

    def __init__(self, output_dim=None, input_size=None):
        """
        The Classifer class: We are developing a model similar to ACTINN for good accuracy
        """
        if output_dim == None or input_size == None:
            raise ValueError('Must explicitly declare input dim (num features) and output dim (number of classes)')

        super(ActinnClassifier, self).__init__()
        self.inp_dim = input_size
        self.out_dim = output_dim

        # feed forward layers
        self.classifier_sequential = nn.Sequential(
                                        nn.Linear(self.inp_dim, 100),
                                        nn.ReLU(),

                                        nn.Linear(100, 50),
                                        nn.ReLU(),

                                        nn.Linear(50, 25),
                                        nn.ReLU(),

                                        nn.Linear(25, output_dim)
                                        )

    def forward(self, x):
        """
        Forward pass of the classifier
        """
        out = self.classifier_sequential(x)
        return out

class ACTINNModel(TorchModel):
    def __init__(self, output_dim = None, input_size = None, **kwargs):
        
        self.model = ActinnClassifier(output_dim, input_size)
                                            
        cf_optimizer = Adam(learning_rate=0.0001, 
                            beta1=0.9, 
                            beta2=0.999, 
                            epsilon=1e-08, 
                            weight_decay=0.005, 
                            )

        cf_decayRate = 0.95
        cf_lr_scheduler = ExponentialDecay(initial_rate=0.0001, decay_rate=cf_decayRate, decay_steps=1000)
        super(ACTINNModel,
              self).__init__(self.model,
                             loss=self.loss_fn,
                             optimizer=cf_optimizer,
                             learning_rate=cf_lr_scheduler,
                             output_types=['prediction'],
                             **kwargs)

    def loss_fn(self, outputs: List, labels: List[torch.Tensor],
                    weights: List[torch.Tensor]) -> torch.Tensor:
        outputs = outputs[0]
        labels = labels[0][:,0]
        return nn.CrossEntropyLoss()(outputs,labels)
