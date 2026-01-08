import pytest
import numpy as np
import deepchem as dc
from deepchem.feat import SmilesToImage

try:
    import torch
    from deepchem.models.torch_models import ChemCeption
    from deepchem.models.torch_models import ChemCeptionModular
    has_torch = True
except:
    has_torch = False


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
def test_chemceptionModular_forward():
    n_samples = 6
    img_size = 80
    n_tasks = 10
    n_classes = 2
    smiles_list = ["CCO", "CC(=O)O", "c1ccccc1", "CCN", "C1CCCCC1", "O=C=O"]

    y_pretrain = np.random.randint(0, n_classes,
                                   (n_samples, n_tasks)).astype(np.float32)
    y_finetune = np.random.randint(0, n_classes,
                                   (n_samples, n_tasks)).astype(np.float32)

    featurizer = SmilesToImage(img_size=img_size, img_spec='std')
    X_images = featurizer.featurize(smiles_list)
    X_images = np.array([img.squeeze() for img in X_images])[:,
                                                             np.newaxis, :, :]

    dataset_pt = dc.data.NumpyDataset(X_images, y_pretrain)
    dataset_ft = dc.data.NumpyDataset(X_images, y_finetune)

    pretrain_model = ChemCeptionModular(task='pretraining',
                                        img_size=img_size,
                                        n_tasks=n_tasks,
                                        n_classes=n_classes,
                                        mode='classification',
                                        learning_rate=1e-4,
                                        device="cpu")

    finetune_model = ChemCeptionModular(task='classification',
                                        img_size=img_size,
                                        n_tasks=n_tasks,
                                        n_classes=n_classes,
                                        mode='classification',
                                        learning_rate=1e-4,
                                        device="cpu")

    pretrain_model.fit(dataset_pt, nb_epoch=1)
    finetune_model.load_from_pretrained(
        pretrain_model, components=pretrain_model.components.keys())
    finetune_model.fit(dataset_ft, nb_epoch=1)

    smiles = ['CC(=O)OC1=CC=CC=C1C(=O)O']  # aspirin
    featurizer = SmilesToImage(img_size=80, img_spec='std')
    images = featurizer.featurize(smiles)

    image = torch.tensor(images, dtype=torch.float32)
    image = image.permute(0, 3, 1, 2).to(device='cpu')

    with torch.no_grad():
        output = finetune_model.model(image)

    assert np.shape(output) == (1, n_tasks, n_classes)
