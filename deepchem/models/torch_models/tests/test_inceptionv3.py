import pytest
import numpy as np
import deepchem as dc
import tempfile

try:
    import torch
    has_torch = True
except ModuleNotFoundError:
    has_torch = False
    pass


def one_hot_encode(labels, num_classes):
    return np.eye(num_classes)[labels]


@pytest.mark.torch
def test_inceptionv3_forward():
    """
    Test the forward pass of the InceptionV3 model
    """
    from deepchem.models.torch_models.inceptionv3 import InceptionV3Model

    # Generate random data for 5 samples with the input shape of (6, 299, 299)
    input_shape = (5, 6, 299, 299)
    input_samples = np.random.randn(*input_shape).astype(np.float32)
    output_samples = np.random.randint(0, 3, (5,)).astype(
        np.int64)  # Random labels for 3 classes

    # Manually one-hot encode the labels
    one_hot_output_samples = one_hot_encode(output_samples, 3)

    dataset = dc.data.ImageDataset(input_samples, one_hot_output_samples)

    # Instantiate and run forward pass on InceptionV3 model
    inception_model = InceptionV3Model(n_tasks=3)
    inception_model.fit(dataset, nb_epoch=1)
    predictions = inception_model.predict(dataset)

    # Ensure predictions shape matches the expected output shape
    assert predictions.shape == (5, 3)


@pytest.mark.torch
def test_inceptionv3_restore():
    """
    Test saving and restoring the InceptionV3 model
    """
    from deepchem.models.torch_models.inceptionv3 import InceptionV3Model

    # Generate random data for testing model saving and loading
    input_shape = (5, 6, 299, 299)
    input_samples = np.random.randn(*input_shape).astype(np.float32)
    output_samples = np.random.randint(0, 3, (5,)).astype(
        np.int64)  # Random labels for 3 classes

    # Manually one-hot encode the labels
    one_hot_output_samples = one_hot_encode(output_samples, 3)

    dataset = dc.data.ImageDataset(input_samples, one_hot_output_samples)

    # Initialize model and set a temporary directory for saving
    model_dir = tempfile.mkdtemp()
    inception_model = InceptionV3Model(n_tasks=3, model_dir=model_dir)

    # Train and get predictions from the model
    inception_model.fit(dataset, nb_epoch=1)
    pred_before_restore = inception_model.predict(dataset)

    # Save and restore model, then compare predictions
    inception_model.save()
    reloaded_model = InceptionV3Model(n_tasks=3, model_dir=model_dir)
    reloaded_model.restore()
    pred_after_restore = reloaded_model.predict(dataset)

    # Ensure predictions before and after restoring are close
    assert np.allclose(pred_before_restore, pred_after_restore, atol=1e-04)


@pytest.mark.torch
def test_basic_conv2d():
    """
    Test the forward pass of the BasicConv2d layer
    """
    from deepchem.models.torch_models.inceptionv3 import BasicConv2d

    conv_layer = BasicConv2d(in_channels=6,
                             out_channels=32,
                             kernel_size=3,
                             stride=2)

    # Create an input tensor with the shape (N, 6, 299, 299)
    batch_size = 5
    input_tensor = torch.randn(batch_size, 6, 299, 299)

    # Run forward pass
    output_tensor = conv_layer(input_tensor)

    # Expected output shape after convolution
    expected_output_shape = (batch_size, 32, 149, 149)

    # Verify the output shape matches expected shape
    assert output_tensor.shape == expected_output_shape


@pytest.mark.torch
def test_InceptionA():
    """
    Test the forward pass of the InceptionA layer
    """
    from deepchem.models.torch_models.inceptionv3 import InceptionA

    mixed_5b = InceptionA(192, pool_features=32)

    # Create an input tensor with the shape (N, 192, 35, 35)
    batch_size = 5
    input_tensor = torch.randn(batch_size, 192, 35, 35)

    # Run forward pass
    output_tensor = mixed_5b(input_tensor)

    # Expected output shape after convolution
    expected_output_shape = (batch_size, 256, 35, 35)

    # Verify the output shape matches expected shape
    assert output_tensor.shape == expected_output_shape


@pytest.mark.torch
def test_InceptionB():
    """
    Test the forward pass of the InceptionB layer
    """
    from deepchem.models.torch_models.inceptionv3 import InceptionB

    mixed_6a = InceptionB(288)

    # Create an input tensor with the shape (N, 288, 35, 35)
    batch_size = 5
    input_tensor = torch.randn(batch_size, 288, 35, 35)

    # Run forward pass
    output_tensor = mixed_6a(input_tensor)

    # Expected output shape after convolution
    expected_output_shape = (batch_size, 768, 17, 17)

    # Verify the output shape matches expected shape
    assert output_tensor.shape == expected_output_shape


@pytest.mark.torch
def test_InceptionC():
    """
    Test the forward pass of the InceptionC layer
    """
    from deepchem.models.torch_models.inceptionv3 import InceptionC

    mixed_6b = InceptionC(768, channels_7x7=128)

    # Create an input tensor with the shape (N, 768, 17, 17)
    batch_size = 5
    input_tensor = torch.randn(batch_size, 768, 17, 17)

    # Run forward pass
    output_tensor = mixed_6b(input_tensor)

    # Expected output shape after convolution
    expected_output_shape = (batch_size, 768, 17, 17)

    # Verify the output shape matches expected shape
    assert output_tensor.shape == expected_output_shape


@pytest.mark.torch
def test_InceptionD():
    """
    Test the forward pass of the InceptionD layer
    """
    from deepchem.models.torch_models.inceptionv3 import InceptionD

    mixed_7a = InceptionD(768)

    # Create an input tensor with the shape (N, 768, 17, 17)
    batch_size = 5
    input_tensor = torch.randn(batch_size, 768, 17, 17)

    # Run forward pass
    output_tensor = mixed_7a(input_tensor)

    # Expected output shape after convolution
    expected_output_shape = (batch_size, 1280, 8, 8)

    # Verify the output shape matches expected shape
    assert output_tensor.shape == expected_output_shape


@pytest.mark.torch
def test_InceptionE():
    """
    Test the forward pass of the InceptionE layer
    """
    from deepchem.models.torch_models.inceptionv3 import InceptionE

    mixed_7b = InceptionE(1280)

    # Create an input tensor with the shape (N, 1280, 8, 8)
    batch_size = 5
    input_tensor = torch.randn(batch_size, 1280, 8, 8)

    # Run forward pass
    output_tensor = mixed_7b(input_tensor)

    # Expected output shape after convolution
    expected_output_shape = (batch_size, 2048, 8, 8)

    # Verify the output shape matches expected shape
    assert output_tensor.shape == expected_output_shape


@pytest.mark.torch
def test_InceptionAux():
    """
    Test the forward pass of the InceptionAux layer
    """
    from deepchem.models.torch_models.inceptionv3 import InceptionAux

    num_classes = 3
    aux = InceptionAux(768, num_classes)

    # Create an input tensor with the shape (N, 768, 17, 17)
    batch_size = 5
    input_tensor = torch.randn(batch_size, 768, 17, 17)

    # Run forward pass
    output_tensor = aux(input_tensor)

    # Expected output shape after convolution
    expected_output_shape = (batch_size, num_classes)

    # Verify the output shape matches expected shape
    assert output_tensor.shape == expected_output_shape


@pytest.mark.torch
def test_inceptionv3_overfit():
    """
    Test the InceptionV3 model's ability to overfit on a small dataset
    """
    from sklearn.metrics import accuracy_score
    from deepchem.models.torch_models.inceptionv3 import InceptionV3Model

    # Generate a small dataset to test overfitting
    input_shape = (3, 3, 299, 299)
    input_samples = np.random.randn(*input_shape).astype(np.float32)
    output_samples = np.array([0, 1,
                               2]).astype(np.int64)  # One sample per class
    one_hot_output_samples = one_hot_encode(output_samples, 3)

    dataset = dc.data.ImageDataset(input_samples, one_hot_output_samples)

    # Initialize model and set a temporary directory for saving
    model_dir = tempfile.mkdtemp()
    inception_model = InceptionV3Model(n_tasks=3,
                                       in_channels=3,
                                       warmup_steps=0,
                                       learning_rate=0.1,
                                       decay_rate=1,
                                       dropout_rate=0.0,
                                       model_dir=model_dir)

    # Train for many epochs to test overfitting capability
    inception_model.fit(dataset, nb_epoch=100)

    # Check performance on the small dataset
    pred = inception_model.predict(dataset)

    # Ensure predictions are in the correct shape to match the number
    # of classes
    assert pred.shape == (3, 3), f"Unexpected prediction shape: {pred.shape}"

    # Convert predictions and labels to one-hot format for metric computation
    pred_labels = np.argmax(pred, axis=1)
    true_labels = output_samples

    # Calculate accuracy using direct comparison
    accuracy = accuracy_score(true_labels, pred_labels)

    # Assert the accuracy is high, indicating overfitting
    assert accuracy > 0.9, "Failed to overfit on small dataset"
