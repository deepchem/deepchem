import pytest
import numpy as np
import deepchem as dc
try:
    import torch
    from deepchem.models.torch_models import ChemCeption
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