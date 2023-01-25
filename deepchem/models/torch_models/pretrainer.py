import deepchem as dc
from deepchem.models import TorchModel
from deepchem.models.losses import L2Loss, L1Loss
import torch.nn as nn
import numpy as np

class PretrainableTorchModel(TorchModel):
    @property
    def embedding():
        return NotImplementedError("Subclass must define the embedding")

    def build_embedding(self):
        return NotImplementedError("Subclass must define the embedding")

    def build_head(self):
        return NotImplementedError("Subclass must define the head")

    def build_model(self):
        return NotImplementedError("Subclass must define the model")

class Pretrainer(TorchModel):
    """Abstract pretrainer class. This class is meant to be subclassed for pretraining TorchModels."""

    def __init__(self, torchmodel: PretrainableTorchModel, **kwargs):
        super().__init__(torchmodel.model, torchmodel.loss, **kwargs)

    @property
    def embedding(self):
        return NotImplementedError("Subclass must define the embedding")

    def build_pretrain_loss(self):
        return NotImplementedError("Subclass must define the pretrain loss")
    
