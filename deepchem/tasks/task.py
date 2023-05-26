from typing import Any
import numpy as np
from torch.functional import F

import deepchem as dc
from deepchem.feat.molecule_featurizers import SNAPFeaturizer
from deepchem.models.torch_models.gnn import GNNModular


class Task():
    pass


class Regression(Task):

    def __init__(self, num_tasks, **kwargs):
        self.output_dim = num_tasks
        self.criterion = F.mse_loss

    def 


featurizer = SNAPFeaturizer()
smiles = ["C1=CC=CC=C1", "C1=CC=CC=C1C=O", "C1=CC=CC=C1C(=O)O"]
features = featurizer.featurize(smiles)
dataset = dc.data.NumpyDataset(features, np.zeros(len(features)))
task = Regression(1)
model = GNNModular(task)
loss = model.fit(dataset, nb_epoch=1)
