import pytest
import numpy as np
import deepchem as dc
import tempfile


@pytest.mark.torch
def test_unet_forward():
    from deepchem.models.torch_models import UNetModel

    # 5 RGB 16x16 pixel input images and 5 grey scale 16x16 pixel output segmentation masks
    input_samples = np.random.randn(5, 3, 16, 16).astype(np.float32)
    output_samples = np.random.rand(5, 1, 16, 16).astype(np.float32)

    # Model works with ImageDataset as well as NumpyDataset.
    # Using NumpyDataset for testing
    np_dataset = dc.data.NumpyDataset(input_samples, output_samples)

    unet_model = UNetModel(in_channels=3, out_channels=1)

    unet_model.fit(np_dataset, nb_epoch=1)
    pred = unet_model.predict(np_dataset)

    assert pred.shape == output_samples.shape


@pytest.mark.torch
def test_unet_restore():
    from deepchem.models.torch_models import UNetModel

    # 5 RGB 16x16 pixel input images and 5 grey scale 16x16 pixel output segmentation masks
    input_samples = np.random.randn(5, 3, 16, 16).astype(np.float32)
    output_samples = np.random.rand(5, 1, 16, 16).astype(np.float32)

    # Using ImageDataset for testing
    np_dataset = dc.data.ImageDataset(input_samples, output_samples)

    model_dir = tempfile.mkdtemp()
    unet_model = UNetModel(in_channels=3, out_channels=1, model_dir=model_dir)

    unet_model.fit(np_dataset, nb_epoch=1)
    pred = unet_model.predict(np_dataset)

    reloaded_model = UNetModel(in_channels=3,
                               out_channels=1,
                               model_dir=model_dir)

    reloaded_model.restore()

    pred = unet_model.predict(np_dataset)
    reloaded_pred = reloaded_model.predict(np_dataset)

    assert len(pred) == len(reloaded_pred)
    assert np.allclose(pred, reloaded_pred, atol=1e-04)


@pytest.mark.torch
def test_unet_overfit():
    from deepchem.models.torch_models import UNetModel

    # 5 RGB 16x16 pixel input images and 5 grey scale 16x16 pixel output segmentation masks
    input_samples = np.random.randn(5, 3, 16, 16).astype(np.float32)
    output_samples = np.random.rand(5, 1, 16, 16).astype(np.float32)

    # Using ImageDataset for testing
    np_dataset = dc.data.NumpyDataset(input_samples, output_samples)

    regression_metric = dc.metrics.Metric(dc.metrics.mean_squared_error,
                                          mode='regression')

    model_dir = tempfile.mkdtemp()
    unet_model = UNetModel(in_channels=3, out_channels=1, model_dir=model_dir)

    unet_model.fit(np_dataset, nb_epoch=100)
    pred = unet_model.predict(np_dataset)

    scores = regression_metric.compute_metric(np_dataset.y.reshape(5, -1),
                                              pred.reshape(5, -1))

    assert scores < 0.05, "Failed to overfit"
