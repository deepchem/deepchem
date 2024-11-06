import pytest
import numpy as np
import deepchem as dc
import tempfile
import torch

@pytest.mark.torch
def test_inceptionv3_forward():
    from deepchem.models.torch_models.inceptionv3 import InceptionV3Model

    # Generate random data for 5 samples with the input shape of (6, 100, 221)
    input_shape = (5, 6, 100, 221)
    input_samples = np.random.randn(*input_shape).astype(np.float32)
    output_samples = np.random.randint(0, 3, (5,)).astype(np.int64)  # Random labels for 3 classes

    dataset = dc.data.ImageDataset(input_samples, output_samples)

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

    dataset = dc.data.ImageDataset(input_samples, output_samples)

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

    dataset = dc.data.ImageDataset(input_samples, output_samples)

    # Initialize model and set a temporary directory for saving
    model_dir = tempfile.mkdtemp()
    inception_model = InceptionV3Model(input_shape=(6, 100, 221), n_tasks=3, model_dir=model_dir)

    # Train for many epochs to test overfitting capability
    inception_model.fit(dataset, nb_epoch=200)

    # Check performance on the small dataset
    pred = inception_model.predict(dataset)
    accuracy_metric = dc.metrics.Metric(dc.metrics.accuracy_score)
    scores = accuracy_metric.compute_metric(output_samples, np.argmax(pred, axis=1))

    # Assert the accuracy is high, indicating overfitting
    assert scores > 0.95, "Failed to overfit on small dataset"
