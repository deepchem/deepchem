import pytest
import numpy as np
import deepchem as dc
import tempfile


@pytest.mark.torch
def test_UNetModel():
    from deepchem.models.torch_models import UNetModel

    input_samples = np.random.randn(5, 3, 32, 32).astype(
        np.float32)  # 5 RGB 32x32 pixel input images
    output_samples = np.random.randn(5, 1, 32, 32).astype(
        np.float32)  # 5 grey scale 32x32 pixel output segmentation masks

    np_dataset = dc.data.NumpyDataset(input_samples, output_samples)

    unet_model = UNetModel(in_channels=3, out_channels=1)

    unet_model.fit(np_dataset, nb_epoch=1)
    pred = unet_model.predict(np_dataset)

    assert pred.shape == output_samples.shape


@pytest.mark.torch
def test_restore_UNetModel():
    from deepchem.models.torch_models import UNetModel

    input_samples = np.random.randn(5, 3, 10, 10).astype(
        np.float32)  # 5 RGB 10x10 pixel input images
    output_samples = np.random.randn(5, 1, 10, 10).astype(
        np.float32)  # 5 grey scale 10x10 pixel output segmentation masks

    np_dataset = dc.data.NumpyDataset(input_samples, output_samples)

    model_dir = tempfile.mkdtemp()
    unet_model = UNetModel(in_channels=3, out_channels=1, model_dir=model_dir)

    unet_model.fit(np_dataset, nb_epoch=1)
    pred = unet_model.predict(np_dataset)

    reloaded_model = UNetModel(in_channels=3,
                               out_channels=1,
                               model_dir=model_dir)

    pred = unet_model.predict(np_dataset)
    reloaded_pred = reloaded_model.predict(np_dataset)

    assert len(pred) == len(reloaded_pred)
    assert np.allclose(pred, reloaded_pred, atol=1e-04)
