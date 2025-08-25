import pytest
import numpy as np
import deepchem as dc
import os
from deepchem.feat import SmilesToImage
from deepchem.molnet.load_function.chembl25_datasets import CHEMBL25_TASKS
import tempfile
try:
    import torch
    from deepchem.models.torch_models import ChemCeption
    from deepchem.models.torch_models import ChemceptionModel
    has_torch = True
except:
    has_torch = False


def get_chemception_dataset(mode="classification", data_points=10, n_tasks=5):
    dataset_file = os.path.join(os.path.dirname(__file__), "assets",
                                "chembl_25_small.csv")

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

    dataset = dc.data.NumpyDataset(dataset.X[:data_points], y, w,
                                   dataset.ids[:data_points])

    return dataset, metric


@pytest.mark.torch
def test_chemception_forward():
    base_filters = 16
    img_size = 80
    n_tasks = 10
    n_classes = 2

    model = ChemCeption(img_spec="std",
                        img_size=img_size,
                        base_filters=base_filters,
                        inception_blocks={
                            "A": 3,
                            "B": 3,
                            "C": 3
                        },
                        n_tasks=n_tasks,
                        n_classes=n_classes,
                        augment=False,
                        mode="classification")

    smiles = ['CC(=O)OC1=CC=CC=C1C(=O)O']
    featurizer = dc.feat.SmilesToImage(img_size=80, img_spec='std')
    images = featurizer.featurize(smiles)
    image = torch.tensor(images, dtype=torch.float32)
    image = image.permute(0, 3, 1, 2)
    output = model(image)

    assert np.shape(output) == (1, n_tasks, n_classes)


@pytest.mark.torch
def test_chemception_base_forward():
    img_size = 80
    n_tasks = 5
    n_classes = 2
    data_points = 20
    smiles = ["CCO", "CCN", "CCC", "CCCl", "CNO"] * (data_points // 5)
    y = np.random.randint(0, n_classes,
                          size=(len(smiles), n_tasks)).astype(np.float32)

    featurizer = SmilesToImage(img_size=img_size, res=img_size)
    X = featurizer.featurize(smiles)
    X = np.transpose(X, (0, 3, 1, 2))

    dataset = dc.data.NumpyDataset(X, y)

    model = ChemceptionModel(img_size=img_size,
                             n_tasks=n_tasks,
                             n_classes=n_classes,
                             mode="regression",
                             augment=True)

    model.fit(dataset, nb_epoch=5)
    preds = model.predict_on_batch(dataset.X)
    assert preds.shape == (data_points, n_tasks)


@pytest.mark.torch
def test_chemception_reload():
    n_tasks = 5
    img_size = 80
    model_dir = tempfile.mkdtemp()

    dataset, metric = get_chemception_dataset(mode="regression",
                                              img_size=img_size,
                                              n_tasks=n_tasks,
                                              data_points=10)

    model = ChemceptionModel(img_size=img_size,
                             n_tasks=n_tasks,
                             model_dir=model_dir,
                             mode="regression",
                             augment=True)
    model.fit(dataset, nb_epoch=10)
    scores = model.evaluate(dataset, [metric], [])

    reloaded_model = ChemceptionModel(img_size=img_size,
                                      n_tasks=n_tasks,
                                      model_dir=model_dir,
                                      mode="regression",
                                      augment=True)
    reloaded_model.restore()
    reloaded_scores = reloaded_model.evaluate(dataset, [metric], [])
    assert scores == reloaded_scores
