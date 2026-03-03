"""
Unit tests for DCTorchSymbolicModel.

Includes:
- Overfit test (model can memorize tiny dataset)
- Save/reload test (predictions preserved)
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import deepchem as dc

from deepchem.models.torch_models import DCTorchSymbolicModel


# --------------------------------------------------
# Regression overfit test
# --------------------------------------------------

def test_symbolic_regression_overfit():
    """Model should overfit tiny regression dataset."""

    # tiny linear dataset
    X = np.random.randn(20, 3)
    y = 2.0 * X[:, 0] - 1.5 * X[:, 1] + 0.3

    dataset = dc.data.NumpyDataset(X, y)

    model = DCTorchSymbolicModel(
        learning_rate=0.01,
        batch_size=20,
        task_type="regression",
    )

    model.fit(dataset, nb_epoch=2000)

    preds = model.predict(dataset).flatten()
    rmse = np.sqrt(np.mean((preds - y) ** 2))

    # should nearly memorize
    assert rmse < 1e-2


# --------------------------------------------------
# Classification overfit test
# --------------------------------------------------

def test_symbolic_classification_overfit():
    """Model should overfit tiny classification dataset."""

    # separable dataset
    X = np.random.randn(30, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(float)

    dataset = dc.data.NumpyDataset(X, y)

    model = DCTorchSymbolicModel(
        learning_rate=0.01,
        batch_size=30,
        task_type="classification",
    )

    model.fit(dataset, nb_epoch=2000)

    logits = model.predict(dataset).flatten()
    probs = 1 / (1 + np.exp(-logits))

    preds = (probs > 0.5).astype(float)
    acc = np.mean(preds == y)

    assert acc > 0.95


# --------------------------------------------------
# Save / reload test
# --------------------------------------------------

def test_symbolic_save_reload(tmpdir):
    """Model predictions should match after save/load."""

    X = np.random.randn(25, 4)
    y = X[:, 0] - 0.5 * X[:, 2]

    dataset = dc.data.NumpyDataset(X, y)

    model = DCTorchSymbolicModel(
        learning_rate=0.01,
        batch_size=25,
        task_type="regression",
    )

    model.fit(dataset, nb_epoch=1000)

    preds_before = model.predict(dataset)

    # save
    model_dir = str(tmpdir.mkdir("symbolic"))
    model.save_checkpoint(model_dir)

    # reload
    new_model = DCTorchSymbolicModel(
        learning_rate=0.01,
        batch_size=25,
        task_type="regression",
        model_dir=model_dir,
    )

    new_model.restore()

    preds_after = new_model.predict(dataset)

    assert np.allclose(preds_before, preds_after)