"""Test normalization of input."""

import numpy as np

import deepchem as dc
from deepchem.metrics import to_one_hot
from deepchem.metrics import from_one_hot
from deepchem.metrics import threshold_predictions
from deepchem.metrics import handle_classification_mode
from deepchem.metrics import normalize_prediction_shape
from deepchem.metrics import normalize_weight_shape


def test_one_hot():
    """Test the one hot encoding."""
    y = np.array([0, 0, 1, 0, 1, 1, 0])
    y_hot = to_one_hot(y)
    expected = np.array([[1, 0], [1, 0], [0, 1], [1, 0], [0, 1], [0, 1], [1,
                                                                          0]])
    yp = from_one_hot(y_hot)
    assert np.array_equal(expected, y_hot)
    assert np.array_equal(y, yp)


def test_handle_classification_mode_direct():
    """Test proper thresholding."""
    y = np.random.rand(10, 2)
    y = y / np.sum(y, axis=1)[:, np.newaxis]
    y = np.expand_dims(y, 1)
    y_expected = y
    y_out = handle_classification_mode(y, "direct")
    assert y_out.shape == (10, 1, 2)
    assert np.array_equal(y_out, y_expected)


def test_handle_classification_mode_threshold():
    """Test proper thresholding."""
    y = np.random.rand(10, 2)
    y = y / np.sum(y, axis=1)[:, np.newaxis]
    y = np.expand_dims(y, 1)
    y_expected = np.argmax(np.squeeze(y), axis=1)[:, np.newaxis]
    y_out = handle_classification_mode(y, "threshold", threshold_value=0.5)
    assert y_out.shape == (10, 1)
    assert np.array_equal(y_out, y_expected)


def test_handle_classification_mode_threshold_nonstandard():
    """Test proper thresholding."""
    y = np.random.rand(10, 2)
    y = y / np.sum(y, axis=1)[:, np.newaxis]
    y_expected = np.where(y[:, 1] >= 0.3, np.ones(10), np.zeros(10))[:,
                                                                     np.newaxis]
    y = np.expand_dims(y, 1)
    y_out = handle_classification_mode(y, "threshold", threshold_value=0.3)
    assert y_out.shape == (10, 1)
    assert np.array_equal(y_out, y_expected)


def test_handle_classification_mode_threshold_one_hot():
    """Test proper thresholding."""
    y = np.random.rand(10, 2)
    y = y / np.sum(y, axis=1)[:, np.newaxis]
    y = np.expand_dims(y, 1)
    y_expected = np.expand_dims(
        to_one_hot(np.argmax(np.squeeze(y), axis=1), n_classes=2), 1)
    y_out = handle_classification_mode(y,
                                       "threshold-one-hot",
                                       threshold_value=0.5)
    assert y_out.shape == (10, 1, 2)
    assert np.array_equal(y_out, y_expected)


def test_threshold_predictions_binary():
    """Test thresholding of binary predictions."""
    # Get a random prediction matrix
    y = np.random.rand(10, 2)
    y = y / np.sum(y, axis=1)[:, np.newaxis]
    y_thresh = threshold_predictions(y, 0.5)
    assert y_thresh.shape == (10,)
    assert (y_thresh == np.argmax(y, axis=1)).all()


def test_threshold_predictions_multiclass():
    """Test thresholding of multiclass predictions."""
    y = np.random.rand(10, 5)
    y = y / np.sum(y, axis=1)[:, np.newaxis]
    y_thresh = threshold_predictions(y)
    assert y_thresh.shape == (10,)
    assert (y_thresh == np.argmax(y, axis=1)).all()


def test_normalize_1d_classification_binary():
    """Tests 1d classification normalization."""
    y = np.array([0, 0, 1, 0, 1, 1, 0])
    expected = np.array([[[1., 0.]], [[1., 0.]], [[0., 1.]], [[1., 0.]],
                         [[0., 1.]], [[0., 1.]], [[1., 0.]]])
    y_out = normalize_prediction_shape(y,
                                       mode="classification",
                                       n_tasks=1,
                                       n_classes=2)
    assert y_out.shape == (7, 1, 2)
    assert np.array_equal(expected, y_out)


def test_normalize_1d_classification_multiclass():
    """Tests 1d classification normalization."""
    y = np.random.randint(5, size=(200,))
    y_expected = np.expand_dims(to_one_hot(y, n_classes=5), 1)
    y_out = normalize_prediction_shape(y,
                                       mode="classification",
                                       n_tasks=1,
                                       n_classes=5)
    assert y_out.shape == (200, 1, 5)
    assert np.array_equal(y_expected, y_out)


def test_normalize_1d_classification_multiclass_explicit_nclasses():
    """Tests 1d classification normalization."""
    y = np.random.randint(5, size=(10,))
    y_expected = np.expand_dims(to_one_hot(y, n_classes=10), 1)
    y_out = normalize_prediction_shape(y,
                                       mode="classification",
                                       n_classes=10,
                                       n_tasks=1)
    assert y_out.shape == (10, 1, 10)
    assert np.array_equal(y_expected, y_out)


def test_normalize_2d_classification_binary():
    """Tests 2d classification normalization."""
    # Of shape (N, n_classes)
    y = np.random.randint(2, size=(10, 1))
    y_expected = np.expand_dims(dc.metrics.to_one_hot(np.squeeze(y)), 1)
    y_out = normalize_prediction_shape(y,
                                       mode="classification",
                                       n_tasks=1,
                                       n_classes=2)
    assert y_out.shape == (10, 1, 2)
    assert np.array_equal(y_expected, y_out)


def test_normalize_3d_classification_binary():
    """Tests 1d classification normalization."""
    # Of shape (N, 1, n_classes)
    y = np.random.randint(2, size=(10,))
    y = dc.metrics.to_one_hot(y, n_classes=2)
    y = np.expand_dims(y, 1)
    y_expected = y
    y_out = normalize_prediction_shape(y,
                                       mode="classification",
                                       n_tasks=1,
                                       n_classes=2)
    assert y_out.shape == (10, 1, 2)
    assert np.array_equal(y_expected, y_out)


def test_normalize_1d_regression():
    """Tests 1d regression normalization."""
    y = np.random.rand(10)
    y_expected = y[:, np.newaxis]
    y_out = normalize_prediction_shape(y, mode="regression", n_tasks=1)
    assert y_out.shape == (10, 1)
    assert np.array_equal(y_expected, y_out)


def test_normalize_2d_regression():
    """Tests 2d regression normalization."""
    y = np.random.rand(10, 5)
    y_expected = y
    y_out = normalize_prediction_shape(y, mode="regression", n_tasks=5)
    assert y_out.shape == (10, 5)
    assert np.array_equal(y_expected, y_out)


def test_normalize_3d_regression():
    """Tests 3d regression normalization."""
    y = np.random.rand(10, 5, 1)
    y_expected = np.squeeze(y)
    y_out = normalize_prediction_shape(y, mode="regression", n_tasks=5)
    assert y_out.shape == (10, 5)
    assert np.array_equal(y_expected, y_out)


def test_scalar_weight_normalization():
    """Test normalization of weights."""
    w_out = normalize_weight_shape(w=5, n_samples=10, n_tasks=5)
    assert w_out.shape == (10, 5)
    assert np.all(w_out == 5 * np.ones((10, 5)))


def test_1d_weight_normalization():
    """Test normalization of weights."""
    w = np.random.rand(10)
    # This has w for each task.
    w_expected = np.array([w, w, w, w, w]).T
    w_out = normalize_weight_shape(w, n_samples=10, n_tasks=5)
    assert w_out.shape == (10, 5)
    assert np.all(w_out == w_expected)


def test_2d_weight_normalization():
    """Test normalization of weights."""
    w = np.random.rand(10, 5)
    w_out = normalize_weight_shape(w, n_samples=10, n_tasks=5)
    assert w_out.shape == (10, 5)
    assert np.all(w_out == w)
