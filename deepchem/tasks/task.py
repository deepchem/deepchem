import numpy as np
from torch.functional import F

import deepchem as dc
from deepchem.feat.molecule_featurizers import SNAPFeaturizer
from deepchem.models.torch_models.gnn import GNNModular


class Task():
    pass


class Regression(Task):

    def __init__(self, modular, num_tasks, **kwargs):
        self.model = modular
        self.output_dim = num_tasks
        self.criterion = F.mse_loss

    def loader(self, inputs, labels):
        pass


featurizer = SNAPFeaturizer()
smiles = ["C1=CC=CC=C1", "C1=CC=CC=C1C=O", "C1=CC=CC=C1C(=O)O"]
features = featurizer.featurize(smiles)
dataset = dc.data.NumpyDataset(features, np.zeros(len(features)))
model = GNNModular(task=Regression)
loss = model.fit(dataset, nb_epoch=1)
