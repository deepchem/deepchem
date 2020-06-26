"""Test normalization of input."""

import numpy as np
import unittest
import deepchem as dc
from deepchem.metrics import to_one_hot
from deepchem.metrics import from_one_hot
from deepchem.metrics import normalize_prediction_shape
from deepchem.metrics import normalize_weight_shape

class TestNormalization(unittest.TestCase):
  """
  Tests that input normalization works as expected.
  """

  def test_one_hot(self):
    """Test the one hot encoding."""
    y = np.array([0, 0, 1, 0, 1, 1, 0])
    y_hot = to_one_hot(y)
    expected = np.array([[1, 0], [1, 0], [0, 1], [1, 0], [0, 1], [0, 1], [1,
                                                                          0]])
    yp = from_one_hot(y_hot)
    assert np.array_equal(expected, y_hot)
    assert np.array_equal(y, yp)

  def test_normalize_scalar_classification_binary(self):
    """Tests 1d classification normalization."""
    y = 1 
    y_out = normalize_prediction_shape(y, mode="classification")
    assert y_out.shape == (1, 1, 2)

  def test_normalize_1d_classification_binary(self):
    """Tests 1d classification normalization."""
    y = np.random.randint(2, size=(10,))
    y_out = normalize_prediction_shape(y, mode="classification")
    assert y_out.shape == (10, 1, 2)

  def test_normalize_1d_classification_multiclass(self):
    """Tests 1d classification normalization."""
    y = np.random.randint(5, size=(200,))
    y_out = normalize_prediction_shape(y, mode="classification")
    assert y_out.shape == (200, 1, 5)

  def test_normalize_1d_classification_multiclass_explicit_nclasses(self):
    """Tests 1d classification normalization."""
    y = np.random.randint(5, size=(10,))
    y_out = normalize_prediction_shape(y, mode="classification", n_classes=10)
    assert y_out.shape == (10, 1, 10)

  def test_normalize_2d_classification_binary(self):
    """Tests 2d classification normalization."""
    # Of shape (N, n_classes)
    y = np.random.randint(2, size=(10,))
    y = dc.metrics.to_one_hot(y, n_classes=2)
    y_out = normalize_prediction_shape(y, mode="classification")
    assert y_out.shape == (10, 1, 2)

  def test_normalize_3d_classification_binary(self):
    """Tests 1d classification normalization."""
    # Of shape (N, 1, n_classes)
    y = np.random.randint(2, size=(10,))
    y = dc.metrics.to_one_hot(y, n_classes=2)
    y = np.expand_dims(y, 1)
    y_out = normalize_prediction_shape(y, mode="classification")
    assert y_out.shape == (10, 1, 2)

  def test_normalize_scalar_regression(self):
    """Tests scalar regression normalization."""
    y = 4.0 
    y_out = normalize_prediction_shape(y, mode="regression")
    assert y_out.shape == (1, 1)

  def test_normalize_1d_regression(self):
    """Tests 1d regression normalization."""
    y = np.random.rand(10)
    y_out = normalize_prediction_shape(y, mode="regression")
    assert y_out.shape == (10, 1)

  def test_normalize_2d_regression(self):
    """Tests 2d regression normalization."""
    y = np.random.rand(10, 5)
    y_out = normalize_prediction_shape(y, mode="regression")
    assert y_out.shape == (10, 5)

  def test_normalize_3d_regression(self):
    """Tests 3d regression normalization."""
    y = np.random.rand(10, 5, 1)
    y_out = normalize_prediction_shape(y, mode="regression")
    assert y_out.shape == (10, 5)

  def test_scalar_weight_normalization(self):
    """Test normalization of weights."""
    w_out = normalize_weight_shape(w=5, n_samples=10, n_tasks=5)
    assert w_out.shape == (10, 5)
    assert np.all(w_out == 5 * np.ones((10, 5)))
    
  def test_1d_weight_normalization(self):
    """Test normalization of weights."""
    w = np.random.rand(10)
    # This has w for each task.
    w_out_correct = np.array([w, w, w, w, w]).T
    w_out = normalize_weight_shape(w, n_samples=10, n_tasks=5)
    assert w_out.shape == (10, 5)
    assert np.all(w_out == w_out_correct)
    
  def test_2d_weight_normalization(self):
    """Test normalization of weights."""
    w = np.random.rand(10, 5)
    w_out = normalize_weight_shape(w, n_samples=10, n_tasks=5)
    assert w_out.shape == (10, 5)
    assert np.all(w_out == w)
