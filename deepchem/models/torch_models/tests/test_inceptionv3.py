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
    input_shape = (5, 6, 100, 221)
    input_samples = np.random.randn(*input_shape).astype(np.float32)
    output_samples = np.random.randint(0, 3, (5,)).astype(np.int64)  # Random labels for 3 classes

    # Manually one-hot encode the labels
    one_hot_output_samples = one_hot_encode(output_samples, 3)

    dataset = dc.data.ImageDataset(input_samples, one_hot_output_samples)

    # Instantiate and run forward pass on InceptionV3 model
    inception_model = InceptionV3Model(input_shape=(6, 100, 221), n_tasks=3)
    inception_model.fit(dataset, nb_epoch=1)
    predictions = inception_model.predict(dataset)

    # Ensure predictions shape matches the expected output shape
    assert predictions.shape == (5, 3)

@pytest.mark.torch
def test_inceptionv3_restore():
    from deepchem.models.torch_models.inceptionv3 import InceptionV3Model

    # Generate random data for testing model saving and loading
    input_shape = (5, 6, 100, 221)
    input_samples = np.random.randn(*input_shape).astype(np.float32)
    output_samples = np.random.randint(0, 3, (5,)).astype(np.int64)  # Random labels for 3 classes

    # Manually one-hot encode the labels
    one_hot_output_samples = one_hot_encode(output_samples, 3)

    dataset = dc.data.ImageDataset(input_samples, one_hot_output_samples)

    # Initialize model and set a temporary directory for saving
    model_dir = tempfile.mkdtemp()
    inception_model = InceptionV3Model(input_shape=(6, 100, 221), n_tasks=3, model_dir=model_dir)

    # Train and get predictions from the model
    inception_model.fit(dataset, nb_epoch=1)
    pred_before_restore = inception_model.predict(dataset)

    # Save and restore model, then compare predictions
    inception_model.save()
    reloaded_model = InceptionV3Model(input_shape=(6, 100, 221), n_tasks=3, model_dir=model_dir)
    reloaded_model.restore()
    pred_after_restore = reloaded_model.predict(dataset)

    # Ensure predictions before and after restoring are close
    assert np.allclose(pred_before_restore, pred_after_restore, atol=1e-04)



@pytest.mark.torch
def test_inceptionv3_overfit():
    from deepchem.models.torch_models.inceptionv3 import InceptionV3Model

    # Generate a small dataset to test overfitting
    input_shape = (5, 6, 100, 221)
    input_samples = np.random.randn(*input_shape).astype(np.float32)
    output_samples = np.random.randint(0, 3, (5,)).astype(np.int64)  # Random labels for 3 classes

    # Manually one-hot encode the labels for 3 classes
    one_hot_output_samples = np.eye(3)[output_samples]  # This ensures one-hot encoding for 3 classes

    # Create a DeepChem ImageDataset
    dataset = dc.data.ImageDataset(input_samples, one_hot_output_samples)

    # Initialize model with n_tasks=3 to correspond to three output classes
    model_dir = tempfile.mkdtemp()
    inception_model = InceptionV3Model(input_shape=(6, 100, 221), n_tasks=3, model_dir=model_dir)

    # Train the model to test overfitting
    inception_model.fit(dataset, nb_epoch=100)

    # Generate predictions
    pred = inception_model.predict(dataset)

    # Confirm that predictions have the expected shape
    assert pred.shape == (5, 3), f"Unexpected prediction shape: {pred.shape}"

    # Calculate accuracy, ensuring we use the integer labels
    accuracy_metric = dc.metrics.Metric(dc.metrics.accuracy_score)
    scores = accuracy_metric.compute_metric(output_samples, np.argmax(pred, axis=1))

    # Assert that the model achieves a high accuracy to confirm overfitting
    assert scores > 0.95, "Failed to overfit on the small dataset"
