import numpy as np

# from torch.functional import F
import deepchem as dc
from deepchem.feat.molecule_featurizers import SNAPFeaturizer

# from deepchem.models.losses import SoftmaxCrossEntropy
from deepchem.models.torch_models.gnn import GNNModular
from deepchem.tasks.task import Classification, Regression

featurizer = SNAPFeaturizer()
smiles = ["C1=CC=CC=C1", "C1=CC=CC=C1C=O", "C1=CC=CC=C1C(=O)O"]
features = featurizer.featurize(smiles)
dataset = dc.data.NumpyDataset(features, np.zeros(len(features)))
pt_task = Regression(1)
model = GNNModular(pt_task)
pt_loss = model.fit(dataset, nb_epoch=1)

ft_task = Classification(1, 2)
model.change_task(ft_task)
ft_loss = model.fit(dataset, nb_epoch=1)
