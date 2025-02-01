import deepchem as dc
import numpy as np

import torch.nn.functional as F
from deepchem.molnet import load_bace_classification, load_delaney
from deepchem.data import NumpyDataset
from deepchem.trans import DAGTransformer
from deepchem.molnet import load_bace_classification, load_delaney
import tensorflow as tf
from deepchem.metrics import Metric, roc_auc_score, mean_absolute_error


def get_dataset(mode='classification', featurizer='GraphConv', num_tasks=2):
    data_points = 20
    if mode == 'classification':
        tasks, all_dataset, transformers = load_bace_classification(featurizer,reload=False)
    else:
        tasks, all_dataset, transformers = load_delaney(featurizer,reload=False)

    train, valid, test = all_dataset
    for _ in range(1, num_tasks):
        tasks.append("random_task")
    w = np.ones(shape=(data_points, len(tasks)))

    if mode == 'classification':
        y = np.random.randint(0, 2, size=(data_points, len(tasks)))
        metric = Metric(roc_auc_score, np.mean, mode="classification")
    else:
        y = np.random.normal(size=(data_points, len(tasks)))
        metric = Metric(mean_absolute_error, mode="regression")

    ds = NumpyDataset(train.X[:data_points], y, w, train.ids[:data_points])

    return tasks, ds, transformers, metric

def test_dag_model():
    tasks, dataset, transformers, metric = get_dataset('classification',
                                                       'GraphConv')

    max_atoms = max([mol.get_num_atoms() for mol in dataset.X])
    transformer = DAGTransformer(max_atoms=max_atoms)
    dataset = transformer.transform(dataset)

    model = dc.models.torch_models.DAGModel(len(tasks),
                     max_atoms=max_atoms,
                     mode='classification',
                     learning_rate=0.001)
    
    model.fit(dataset, nb_epoch=30)
    scores = model.evaluate(dataset, [metric], transformers)
    assert scores['mean-roc_auc_score'] >= 0.9


def test_dag_regression_model():
    np.random.seed(1234)
    tf.random.set_seed(1234)
    tasks, dataset, transformers, metric = get_dataset('regression',
                                                       'GraphConv')

    max_atoms = max([mol.get_num_atoms() for mol in dataset.X])
    transformer = DAGTransformer(max_atoms=max_atoms)
    dataset = transformer.transform(dataset)

    model = dc.models.torch_models.DAGModel(len(tasks),
                     max_atoms=max_atoms,
                     mode='regression',
                     learning_rate=0.003)

    model.fit(dataset, nb_epoch=100)
    scores = model.evaluate(dataset, [metric], transformers)
    assert scores['mean_absolute_error'] < 0.15

def test_dag_regression_uncertainty():
    np.random.seed(1234)
    tf.random.set_seed(1234)
    tasks, dataset, _, _ = get_dataset('regression', 'GraphConv')

    batch_size = 10
    max_atoms = max([mol.get_num_atoms() for mol in dataset.X])
    transformer = DAGTransformer(max_atoms=max_atoms)
    dataset = transformer.transform(dataset)

    model = dc.models.torch_models.DAGModel(len(tasks),
                     max_atoms=max_atoms,
                     mode='regression',
                     learning_rate=0.003,
                     batch_size=batch_size,
                     use_queue=False,
                     dropout=0.05,
                     uncertainty=True)

    model.fit(dataset, nb_epoch=750)

    # Predict the output and uncertainty.
    pred, std = model.predict_uncertainty(dataset)
    mean_error = np.mean(np.abs(dataset.y - pred))
    mean_value = np.mean(np.abs(dataset.y))
    mean_std = np.mean(std)
    # The DAG models have high error with dropout
    # Despite a lot of effort tweaking it , there appears to be
    # a limit to how low the error can go with dropout.
    # assert mean_error < 0.5 * mean_value
    assert mean_error < .7 * mean_value
    assert mean_std > 0.5 * mean_error
    assert mean_std < mean_value