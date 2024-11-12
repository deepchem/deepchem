import pytest
import numpy as np
import deepchem as dc
import tempfile


def one_hot_encode(labels, num_classes):
    return np.eye(num_classes)[labels]


@pytest.mark.torch
def test_inceptionv3_forward():
    from deepchem.models.torch_models.inceptionv3 import InceptionV3Model

    # Generate random data for 5 samples with the input shape of (6, 100, 221)
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
def test_inceptionv3_overfit():
    from deepchem.models.torch_models.inceptionv3 import InceptionV3Model

    # Generate a small dataset to test overfitting
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

    # Train for many epochs to test overfitting capability
    inception_model.fit(dataset, nb_epoch=20)

    # Check performance on the small dataset
    pred = inception_model.predict(dataset)

    # Ensure predictions are in the correct shape to match the number of classes
    assert pred.shape == (5, 3), f"Unexpected prediction shape: {pred.shape}"

    # Convert predictions and labels to one-hot format for metric computation
    pred_labels = np.argmax(pred, axis=1)
    true_labels = output_samples

    # Calculate accuracy using direct comparison
    accuracy = np.mean(pred_labels == true_labels)

    # Assert the accuracy is high, indicating overfitting
    assert accuracy > 0.9, "Failed to overfit on small dataset"
