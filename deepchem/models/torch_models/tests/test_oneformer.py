import pytest
import numpy as np
from deepchem.data import ImageDataset
try:
    import torch
    from deepchem.models.torch_models.oneformer import OneFormer
    from deepchem.models.torch_models import HuggingFaceModel
except ModuleNotFoundError:
    pass

@pytest.mark.torch
def test_oneformer_train():
    model = OneFormer(model_path='shi-labs/oneformer_ade20k_swin_tiny',
                      segmentation_task="semantic",
                      torch_dtype=torch.float16,
                      batch_size=1)
    X = np.random.randint(0, 255, (3, 224, 224, 3))
    y = np.random.randint(0, 1, (3, 224, 224))

    dataset = ImageDataset(X, y)
    avg_loss = model.fit(dataset, nb_epoch=2)

    assert isinstance(model, HuggingFaceModel)
    assert isinstance(avg_loss, float)


@pytest.mark.torch
def test_oneformer_predict():
    model = OneFormer(model_path='shi-labs/oneformer_ade20k_swin_tiny',
                      segmentation_task="semantic",
                      torch_dtype=torch.float16,
                      batch_size=1)
    X = np.random.randint(0, 255, (3, 224, 224, 3))
    y = np.random.randint(0, 1, (3, 224, 224))

    dataset = ImageDataset(X, y)
    preds = model.predict(dataset)
    preds = np.array(preds)

    assert isinstance(model, HuggingFaceModel)
    assert np.array(preds).shape == y.shape
