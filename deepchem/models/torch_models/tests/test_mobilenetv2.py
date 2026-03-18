import pytest
import numpy as np
import tempfile

try:
    import torch
    has_torch = True
except ModuleNotFoundError:
    has_torch = False
    pass


@pytest.mark.torch
def test_mobilenetv2_overfit_classification():
    """
    Test the MobileNetV2 model's ability to overfit on a small classification dataset
    """
    from deepchem.data import NumpyDataset
    input_size = 96
    n_classes = 2
    in_channels = 3
    np.random.seed(123)

    input_samples = np.random.randn(3, in_channels, input_size,
                                    input_size).astype(np.float32)
    output_samples = np.array([0, 1, 0]).astype(np.int64)

    dataset = NumpyDataset(X=input_samples,
                           y=np.reshape(output_samples, (-1, 1)))

    from deepchem.models.torch_models.mobilenetv2 import MobileNetV2Model
    model_dir = tempfile.mkdtemp()
    mobilenet_model = MobileNetV2Model(
        n_tasks=1,
        in_channels=in_channels,
        input_size=input_size,
        mode="classification",
        n_classes=n_classes,
        model_dir=model_dir,
        learning_rate=0.00001,
    )

    mobilenet_model.fit(dataset, nb_epoch=200)
    pred = mobilenet_model.predict(dataset)

    pred_labels = np.argmax(pred, axis=1)
    true_labels = output_samples

    correct_predictions = np.sum(pred_labels == true_labels)
    total_predictions = len(true_labels)
    accuracy = correct_predictions / total_predictions

    # Assert the accuracy is high, indicating overfitting
    assert accuracy > 0.9, "Failed to overfit on small dataset"


@pytest.mark.torch
def test_mobilenetv2_regression_overfit():
    """
    Test the MobileNetV2 model's ability to overfit on a small regression dataset
    """
    from deepchem.data import NumpyDataset

    input_size = 96
    in_channels = 6
    n_tasks = 1
    np.random.seed(123)
    input_samples = np.random.randn(4, in_channels, input_size,
                                    input_size).astype(np.float32)
    output_samples = np.array([0.5, 2.3, 1.7, 3.9]).astype(np.float32)
    weights = np.ones((4, 1))

    dataset = NumpyDataset(X=input_samples,
                           y=np.reshape(output_samples, (-1, 1)),
                           w=weights)

    from deepchem.models.torch_models.mobilenetv2 import MobileNetV2Model
    model_dir = tempfile.mkdtemp()
    mobilenet_model = MobileNetV2Model(
        n_tasks=n_tasks,
        in_channels=in_channels,
        input_size=input_size,
        mode="regression",  # Use regression mode
        model_dir=model_dir,
        learning_rate=0.0001,
    )
    mobilenet_model.fit(dataset, nb_epoch=200)
    pred = mobilenet_model.predict(dataset)
    squared_errors = np.square(pred.flatten() - output_samples)
    mse = np.mean(squared_errors)

    mae = np.mean(np.abs(pred.flatten() - output_samples))

    # For overfitting, expect very low error
    assert mse < 0.1, "Failed to overfit regression dataset: MSE too high"
    assert mae < 0.2, "Failed to overfit regression dataset: MAE too high"


@pytest.mark.torch
def test_mobilenetv2_regression_forward():
    """
    Test the MobileNetV2 model's forward pass on a small regression dataset
    """
    from deepchem.data import NumpyDataset
    input_size = 96
    in_channels = 6
    n_tasks = 1

    np.random.seed(123)
    input_samples = np.random.randn(4, in_channels, input_size,
                                    input_size).astype(np.float32)
    output_samples = np.array([0.5, 2.3, 1.7, 3.9]).astype(np.float32)
    weights = np.ones((4, 1))

    dataset = NumpyDataset(X=input_samples,
                           y=np.reshape(output_samples, (-1, 1)),
                           w=weights)

    from deepchem.models.torch_models.mobilenetv2 import MobileNetV2Model
    model_dir = tempfile.mkdtemp()
    mobilenet_model = MobileNetV2Model(
        n_tasks=n_tasks,
        in_channels=in_channels,
        input_size=input_size,
        mode="regression",  # Use regression mode
        model_dir=model_dir,
        learning_rate=0.0001,
    )

    mobilenet_model.fit(dataset, nb_epoch=1)

    pred = mobilenet_model.predict(dataset)

    assert pred.shape == (4, 1), f"Unexpected prediction shape: {pred.shape}"


@pytest.mark.torch
def test_mobilenetv2_classification_forward():
    """
    Test the MobileNetV2 model's forward pass on a small classificationdataset
    """
    from deepchem.data import NumpyDataset
    input_size = 96
    n_classes = 2
    in_channels = 3
    np.random.seed(123)

    input_samples = np.random.randn(3, in_channels, input_size,
                                    input_size).astype(np.float32)
    output_samples = np.array([0, 1, 0]).astype(np.int64)

    dataset = NumpyDataset(X=input_samples,
                           y=np.reshape(output_samples, (-1, 1)))

    from deepchem.models.torch_models.mobilenetv2 import MobileNetV2Model
    model_dir = tempfile.mkdtemp()
    mobilenet_model = MobileNetV2Model(
        n_tasks=1,
        in_channels=in_channels,
        input_size=input_size,
        mode="classification",
        n_classes=n_classes,
        model_dir=model_dir,
        learning_rate=0.01,
    )
    mobilenet_model.fit(dataset, nb_epoch=1)

    pred = mobilenet_model.predict(dataset)

    assert pred.shape == (
        3, n_classes), f"Unexpected prediction shape: {pred.shape}"


@pytest.mark.torch
def test_inverted_residual_forward():
    """
    Test the forward pass of the InvertedResidual class
    """
    # Test cases with different configurations
    test_cases = [
        # inp, oup, stride, expand_ratio, input_shape
        (16, 16, 1, 1, (2, 16, 32, 32)),  # Residual connection case
        (16, 24, 1, 6, (2, 16, 32, 32)),  # No residual, expand ratio > 1
        (24, 32, 2, 6, (2, 24, 32, 32)),  # Stride 2, expand ratio > 1
        (32, 32, 1, 1, (2, 32, 16, 16)),  # Residual, smallest expand ratio
    ]

    for inp, oup, stride, expand_ratio, input_shape in test_cases:
        x = torch.randn(input_shape)

        from deepchem.models.torch_models.mobilenetv2 import InvertedResidual
        block = InvertedResidual(inp, oup, stride, expand_ratio)

        output = block(x)
        expected_h = input_shape[2] // stride
        expected_w = input_shape[3] // stride
        expected_shape = (input_shape[0], oup, expected_h, expected_w)

        # Assert output shape
        assert output.shape == expected_shape, \
            f"Expected shape {expected_shape}, got {output.shape} for params: " \
            f"inp={inp}, oup={oup}, stride={stride}, expand_ratio={expand_ratio}"

        # Check if residual connection is used when expected
        if stride == 1 and inp == oup:
            assert block.use_res_connect, "Residual connection should be used"
        else:
            assert not block.use_res_connect, "Residual connection should not be used"

        # For residual blocks, verify output differs from input
        if block.use_res_connect:
            # The difference should not be zero (but may be small)
            diff = torch.sum(torch.abs(output - x))
            assert diff > 0, "Residual block output should not be identical to input"


@pytest.mark.torch
def test_mobilenetv2_restore():
    """
    Test saving and restoring the MobileNetV2 model
    """
    from deepchem.data import NumpyDataset

    input_size = 96  # Must be divisible by 32
    in_channels = 6
    n_classes = 3
    batch_size = 5

    input_samples = np.random.randn(batch_size, in_channels, input_size,
                                    input_size).astype(np.float32)
    output_samples = np.random.randint(0, n_classes,
                                       (batch_size,)).astype(np.int64)

    dataset = NumpyDataset(X=input_samples,
                           y=np.reshape(output_samples, (-1, 1)))

    from deepchem.models.torch_models.mobilenetv2 import MobileNetV2Model
    model_dir = tempfile.mkdtemp()
    mobilenet_model = MobileNetV2Model(n_tasks=1,
                                       in_channels=in_channels,
                                       input_size=input_size,
                                       mode="classification",
                                       n_classes=n_classes,
                                       model_dir=model_dir)

    mobilenet_model.fit(dataset, nb_epoch=1)
    pred_before_restore = mobilenet_model.predict(dataset)

    mobilenet_model.save()
    reloaded_model = MobileNetV2Model(n_tasks=1,
                                      in_channels=in_channels,
                                      input_size=input_size,
                                      mode="classification",
                                      n_classes=n_classes,
                                      model_dir=model_dir)
    reloaded_model.restore()
    pred_after_restore = reloaded_model.predict(dataset)

    # Ensure predictions before and after restoring are identical
    assert np.allclose(pred_before_restore, pred_after_restore, atol=1e-04)

    # Verify prediction shape is correct
    assert pred_after_restore.shape == (batch_size, n_classes), \
        f"Expected prediction shape {(batch_size, n_classes)}, got {pred_after_restore.shape}"
