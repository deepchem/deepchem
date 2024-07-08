import os
import numpy as np
import tempfile
import pytest

import deepchem as dc
from deepchem.feat import create_char_to_idx, SmilesToSeq
from deepchem.molnet.load_function.chembl25_datasets import CHEMBL25_TASKS

try:
    import torch
except ModuleNotFoundError:
    pass


def get_dataset(mode="regression",
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

    loader = dc.data.CSVLoader(tasks=CHEMBL25_TASKS,
                               feature_field='smiles',
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


@pytest.mark.torch
def test_smiles2vec_model():
    from deepchem.models.torch_models import Smiles2Vec

    n_tasks = 10
    max_seq_len = 20

    _, _, char_to_idx = get_dataset(
        mode="regression",
        featurizer="smiles2seq",
        n_tasks=n_tasks,
        max_seq_len=max_seq_len,
    )
    model = Smiles2Vec(char_to_idx)
    input = torch.randint(low=0, high=len(char_to_idx), size=(1, max_seq_len))
    # Ex: input = torch.tensor([[32,32,32,32,32,32,25,29,15,17,29,29,32,32,32,32,32,32,32,32]])

    logits = model.forward(input)
    assert np.shape(logits) == (1, n_tasks, 1)
