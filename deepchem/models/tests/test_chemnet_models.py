import os
import numpy as np
import tempfile

import pytest
from flaky import flaky
import deepchem as dc
from deepchem.feat import create_char_to_idx, SmilesToSeq, SmilesToImage
from deepchem.molnet.load_function.chembl25_datasets import CHEMBL25_TASKS

try:
    from deepchem.models import Smiles2Vec, ChemCeption
    has_tensorflow = True
except:
    has_tensorflow = False


@pytest.mark.tensorflow
def get_dataset(mode="classification",
                featurizer="smiles2seq",
                max_seq_len=20,
                data_points=10,
                n_tasks=5):
    dataset_file = os.path.join(os.path.dirname(__file__), "assets",
                                "chembl_25_small.csv")

    if featurizer == "smiles2seq":
        max_len = 250
        pad_len = 10
        char_to_idx = create_char_to_idx(dataset_file,
                                         max_len=max_len,
                                         smiles_field="smiles")
        feat = SmilesToSeq(char_to_idx=char_to_idx,
                           max_len=max_len,
                           pad_len=pad_len)

    elif featurizer == "smiles2img":
        img_size = 80
        img_spec = "engd"
        res = 0.5
        feat = SmilesToImage(img_size=img_size, img_spec=img_spec, res=res)

    loader = dc.data.CSVLoader(tasks=CHEMBL25_TASKS,
                               smiles_field='smiles',
                               featurizer=feat)
    dataset = loader.create_dataset(inputs=[dataset_file],
                                    shard_size=10000,
                                    data_dir=tempfile.mkdtemp())

    w = np.ones(shape=(data_points, n_tasks))

    if mode == 'classification':
        y = np.random.randint(0, 2, size=(data_points, n_tasks))
        metric = dc.metrics.Metric(dc.metrics.roc_auc_score,
                                   np.mean,
                                   mode="classification")
    else:
        y = np.random.normal(size=(data_points, n_tasks))
        metric = dc.metrics.Metric(dc.metrics.mean_absolute_error,
                                   mode="regression")

    if featurizer == "smiles2seq":
        dataset = dc.data.NumpyDataset(dataset.X[:data_points, :max_seq_len], y,
                                       w, dataset.ids[:data_points])
    else:
        dataset = dc.data.NumpyDataset(dataset.X[:data_points], y, w,
                                       dataset.ids[:data_points])

    if featurizer == "smiles2seq":
        return dataset, metric, char_to_idx
    else:
        return dataset, metric


@pytest.mark.slow
@pytest.mark.tensorflow
def test_chemception_regression():
    n_tasks = 5
    dataset, metric = get_dataset(mode="regression",
                                  featurizer="smiles2img",
                                  n_tasks=n_tasks)
    model = ChemCeption(n_tasks=n_tasks,
                        img_spec="engd",
                        model_dir=None,
                        mode="regression")
    model.fit(dataset, nb_epoch=300)
    scores = model.evaluate(dataset, [metric], [])
    assert scores['mean_absolute_error'] < 0.1


@pytest.mark.slow
@pytest.mark.tensorflow
def test_chemception_classification():
    n_tasks = 5
    dataset, metric = get_dataset(mode="classification",
                                  featurizer="smiles2img",
                                  n_tasks=n_tasks)
    model = ChemCeption(n_tasks=n_tasks,
                        img_spec="engd",
                        model_dir=None,
                        mode="classification")
    model.fit(dataset, nb_epoch=300)
    scores = model.evaluate(dataset, [metric], [])
    assert scores['mean-roc_auc_score'] >= 0.9


@pytest.mark.slow
@pytest.mark.tensorflow
def test_smiles_to_vec_regression():
    n_tasks = 5
    max_seq_len = 20
    dataset, metric, char_to_idx = get_dataset(mode="regression",
                                               featurizer="smiles2seq",
                                               n_tasks=n_tasks,
                                               max_seq_len=max_seq_len)
    model = Smiles2Vec(char_to_idx=char_to_idx,
                       max_seq_len=max_seq_len,
                       use_conv=True,
                       n_tasks=n_tasks,
                       model_dir=None,
                       mode="regression")
    model.fit(dataset, nb_epoch=500)
    scores = model.evaluate(dataset, [metric], [])
    assert scores['mean_absolute_error'] < 0.1


@pytest.mark.slow
@pytest.mark.tensorflow
def test_smiles_to_vec_classification():
    n_tasks = 5
    max_seq_len = 20
    dataset, metric, char_to_idx, = get_dataset(mode="classification",
                                                featurizer="smiles2seq",
                                                n_tasks=n_tasks,
                                                max_seq_len=max_seq_len)
    model = Smiles2Vec(char_to_idx=char_to_idx,
                       max_seq_len=max_seq_len,
                       use_conv=True,
                       n_tasks=n_tasks,
                       model_dir=None,
                       mode="classification")
    model.fit(dataset, nb_epoch=500)
    scores = model.evaluate(dataset, [metric], [])
    assert scores['mean-roc_auc_score'] >= 0.9


@flaky
@pytest.mark.slow
@pytest.mark.tensorflow
def test_chemception_fit_with_augmentation():
    n_tasks = 5
    dataset, metric = get_dataset(mode="classification",
                                  featurizer="smiles2img",
                                  n_tasks=n_tasks)
    model = ChemCeption(n_tasks=n_tasks,
                        img_spec="engd",
                        model_dir=None,
                        augment=True,
                        mode="classification")
    model.fit(dataset, nb_epoch=300)
    scores = model.evaluate(dataset, [metric], [])
    assert scores['mean-roc_auc_score'] >= 0.9
