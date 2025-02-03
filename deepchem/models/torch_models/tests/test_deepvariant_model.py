import os
import numpy as np
import pytest
import deepchem as dc
import tempfile
from deepchem.models.torch_models.deepvariant_model import DeepVariant


def one_hot_encode(labels, num_classes):
    return np.eye(num_classes)[labels]


@pytest.mark.torch
def test_deepvariant_vcf():
    """Test VCF generation from predictions."""

    n_samples = 5
    n_features = 6
    n_tasks = 3
    img_size = 299

    # Generate dummy dataset
    X = np.random.rand(n_samples, n_features, img_size,
                       img_size).astype(np.float32)
    y = []
    for i in range(n_samples):
        y.append({
            'chrom': 'chr1',
            'pos': 1000 + i,
            'ref_allele': 'A',
            'alt_allele': 'T',
            'depth': 30
        })

    dataset = dc.data.datasets.NumpyDataset(X, y)
    model = DeepVariant(n_tasks=n_tasks, in_channels=n_features)

    # Create temp file
    tmpfile = tempfile.NamedTemporaryFile(delete=False)
    fname = tmpfile.name

    try:
        model.call_variants(dataset, fname, "test_sample")
        with open(fname) as f:
            lines = f.readlines()
            assert any("##fileformat=VCFv4.2" in line for line in lines)
            assert sum(
                1 for line in lines if not line.startswith("#")) == n_samples
    finally:
        os.unlink(fname)


@pytest.mark.torch
def test_deepvariant_restore():
    """
    Test training, saving and restoring the DeepVariant model
    """

    # Generate random data for testing model saving and loading
    input_shape = (5, 6, 299, 299)
    input_samples = np.random.randn(*input_shape).astype(np.float32)
    output_samples = np.random.randint(0, 3, (5,)).astype(np.int64)

    # Manually one-hot encode the labels
    one_hot_output_samples = one_hot_encode(output_samples, 3)

    dataset = dc.data.ImageDataset(input_samples, one_hot_output_samples)

    # Initialize model and set a temporary directory for saving
    model_dir = tempfile.mkdtemp()
    model = dc.models.DeepVariant(n_tasks=3, model_dir=model_dir)

    # Train and get predictions from the model
    model.fit(dataset, nb_epoch=1)
    pred_before_restore = model.predict(dataset)

    # Save and restore model, then compare predictions
    model.save()
    reloaded_model = dc.models.DeepVariant(n_tasks=3, model_dir=model_dir)
    reloaded_model.restore()
    pred_after_restore = reloaded_model.predict(dataset)

    # Ensure predictions before and after restoring are close
    assert np.allclose(pred_before_restore, pred_after_restore, atol=1e-04)
