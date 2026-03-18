import pytest
import tempfile

import numpy as np

import deepchem as dc
from deepchem.feat import PagtnMolGraphFeaturizer
from deepchem.models.tests.test_graph_models import get_dataset

try:
    import dgl  # noqa: F401
    import dgllife  # noqa: F401
    import torch  # noqa: F401
    from deepchem.models import PagtnModel
    has_torch_and_dgl = True
except:
    has_torch_and_dgl = False


@pytest.mark.torch
def test_pagtn_regression():
    # load datasets
    featurizer = PagtnMolGraphFeaturizer(max_length=5)
    tasks, dataset, transformers, metric = get_dataset('regression',
                                                       featurizer=featurizer)

    # initialize models
    n_tasks = len(tasks)
    model = PagtnModel(mode='regression', n_tasks=n_tasks, batch_size=16)

    # overfit test
    model.fit(dataset, nb_epoch=150)
    scores = model.evaluate(dataset, [metric], transformers)
    assert scores['mean_absolute_error'] < 0.65

    # test on a small MoleculeNet dataset
    from deepchem.molnet import load_delaney

    tasks, all_dataset, transformers = load_delaney(featurizer=featurizer)
    train_set, _, _ = all_dataset
    model = PagtnModel(mode='regression', n_tasks=n_tasks, batch_size=16)
    model.fit(train_set, nb_epoch=1)


@pytest.mark.torch
def test_pagtn_classification():
    # load datasets
    featurizer = PagtnMolGraphFeaturizer(max_length=5)
    tasks, dataset, transformers, metric = get_dataset('classification',
                                                       featurizer=featurizer)

    # initialize models
    n_tasks = len(tasks)
    model = PagtnModel(mode='classification', n_tasks=n_tasks, batch_size=16)

    # overfit test
    model.fit(dataset, nb_epoch=100)
    scores = model.evaluate(dataset, [metric], transformers)
    assert scores['mean-roc_auc_score'] >= 0.85

    # test on a small MoleculeNet dataset
    from deepchem.molnet import load_bace_classification

    tasks, all_dataset, transformers = load_bace_classification(
        featurizer=featurizer)
    train_set, _, _ = all_dataset
    model = PagtnModel(mode='classification', n_tasks=len(tasks), batch_size=16)
    model.fit(train_set, nb_epoch=1)


@pytest.mark.torch
def test_pagtn_reload():
    # load datasets
    featurizer = PagtnMolGraphFeaturizer(max_length=5)
    tasks, dataset, transformers, metric = get_dataset('classification',
                                                       featurizer=featurizer)

    # initialize models
    n_tasks = len(tasks)
    model_dir = tempfile.mkdtemp()
    model = PagtnModel(mode='classification',
                       n_tasks=n_tasks,
                       model_dir=model_dir,
                       batch_size=16)

    model.fit(dataset, nb_epoch=100)
    scores = model.evaluate(dataset, [metric], transformers)
    assert scores['mean-roc_auc_score'] >= 0.85

    reloaded_model = PagtnModel(mode='classification',
                                n_tasks=n_tasks,
                                model_dir=model_dir,
                                batch_size=16)
    reloaded_model.restore()

    pred_mols = ["CCCC", "CCCCCO", "CCCCC"]
    X_pred = featurizer(pred_mols)
    random_dataset = dc.data.NumpyDataset(X_pred)
    original_pred = model.predict(random_dataset)
    reload_pred = reloaded_model.predict(random_dataset)
    assert np.all(original_pred == reload_pred)
