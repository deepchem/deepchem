import pytest
import tempfile
from flaky import flaky
import numpy as np

import deepchem as dc
from deepchem.feat import MolGraphConvFeaturizer
from deepchem.models.tests.test_graph_models import get_dataset

try:
    import dgl  # noqa: F401
    import dgllife  # noqa: F401
    import torch  # noqa: F401
    from deepchem.models import GCNModel
    has_torch_and_dgl = True
except:
    has_torch_and_dgl = False


@pytest.mark.torch
def test_gcn_regression():
    # load datasets
    featurizer = MolGraphConvFeaturizer()
    tasks, dataset, transformers, metric = get_dataset('regression',
                                                       featurizer=featurizer)

    # initialize models
    n_tasks = len(tasks)
    model = GCNModel(mode='regression',
                     n_tasks=n_tasks,
                     number_atom_features=30,
                     batch_size=10,
                     learning_rate=0.001)

    # overfit test
    model.fit(dataset, nb_epoch=300)
    scores = model.evaluate(dataset, [metric], transformers)
    assert scores['mean_absolute_error'] < 0.5

    # test on a small MoleculeNet dataset
    from deepchem.molnet import load_delaney

    tasks, all_dataset, transformers = load_delaney(featurizer=featurizer)
    train_set, _, _ = all_dataset
    model = dc.models.GCNModel(n_tasks=len(tasks),
                               graph_conv_layers=[2],
                               residual=False,
                               predictor_hidden_feats=2)
    model.fit(train_set, nb_epoch=1)


@flaky
@pytest.mark.torch
def test_gcn_classification():
    # load datasets
    featurizer = MolGraphConvFeaturizer()
    tasks, dataset, transformers, metric = get_dataset('classification',
                                                       featurizer=featurizer)

    # initialize models
    n_tasks = len(tasks)
    model = GCNModel(mode='classification',
                     n_tasks=n_tasks,
                     number_atom_features=30,
                     batch_size=10,
                     learning_rate=0.0003)

    # overfit test
    model.fit(dataset, nb_epoch=70)
    scores = model.evaluate(dataset, [metric], transformers)
    assert scores['mean-roc_auc_score'] >= 0.85

    # test on a small MoleculeNet dataset
    from deepchem.molnet import load_bace_classification

    tasks, all_dataset, transformers = load_bace_classification(
        featurizer=featurizer)
    train_set, _, _ = all_dataset
    model = dc.models.GCNModel(mode='classification',
                               n_tasks=len(tasks),
                               graph_conv_layers=[2],
                               residual=False,
                               predictor_hidden_feats=2)
    model.fit(train_set, nb_epoch=1)


@flaky
@pytest.mark.torch
def test_gcn_reload():
    # load datasets
    featurizer = MolGraphConvFeaturizer()
    tasks, dataset, transformers, metric = get_dataset('classification',
                                                       featurizer=featurizer)

    # initialize models
    n_tasks = len(tasks)
    model_dir = tempfile.mkdtemp()
    model = GCNModel(mode='classification',
                     n_tasks=n_tasks,
                     number_atom_features=30,
                     model_dir=model_dir,
                     batch_size=10,
                     learning_rate=0.0003)

    model.fit(dataset, nb_epoch=70)
    scores = model.evaluate(dataset, [metric], transformers)
    assert scores['mean-roc_auc_score'] >= 0.85

    reloaded_model = GCNModel(mode='classification',
                              n_tasks=n_tasks,
                              number_atom_features=30,
                              model_dir=model_dir,
                              batch_size=10,
                              learning_rate=0.0003)
    reloaded_model.restore()

    pred_mols = ["CCCC", "CCCCCO", "CCCCC"]
    X_pred = featurizer(pred_mols)
    random_dataset = dc.data.NumpyDataset(X_pred)
    original_pred = model.predict(random_dataset)
    reload_pred = reloaded_model.predict(random_dataset)
    assert np.all(original_pred == reload_pred)
