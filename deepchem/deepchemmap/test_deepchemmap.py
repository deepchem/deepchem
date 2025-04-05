import os
import deepchem as dc
import numpy as np
import pytest
import tempfile
from flaky import flaky
from deepchem.data import NumpyDataset, CSVLoader
from deepchem.feat import ConvMolFeaturizer
from deepchem.metrics import Metric, roc_auc_score, mean_absolute_error
from deepchem.molnet import load_bace_classification, load_delaney
from deepchem.utils.data_utils import download_url, get_data_dir
from deepchem import deepchemmap
from deepchem.models.optimizers import Optimizer,AdamW

try:
    import torch
    from deepchem.models.torch_models import GraphConvModel
    has_torch = True
except:
    has_torch = False

@flaky
@pytest.mark.torch
def test_load_param_dict():
    tasks, datasets, transformers = dc.molnet.load_tox21(featurizer='GraphConv')
    train_dataset, valid_dataset, test_dataset = datasets
    n_tasks = len(tasks)
    num_features = train_dataset.X[0].get_atom_features().shape[1]
    model = dc.models.torch_models.GraphConvModel(n_tasks, mode='classification',number_input_features=[num_features,64],device = 'cpu')
    model.fit(train_dataset, nb_epoch=1)
    model.save_pretrained("save_graph")
    param_path = os.path.join("Save_graph", "parameters.json")
    param_dict = deepchemmap.Map.load_param_dict(param_path)
    assert param_dict == model.param_dict