import deepchem.models.losses as losses
import unittest
import numpy as np

try:
  import tensorflow as tf
  has_tensorflow = True
except:
  has_tensorflow = False

try:
  import torch
  has_pytorch = True
except:
  has_pytorch = False


class TestLosses(unittest.TestCase):
  """Test loss functions."""

  @unittest.skipIf(not has_tensorflow, 'TensorFlow is not installed')
  def test_l1_loss_tf(self):
    """Test L1Loss."""
    loss = losses.L1Loss()
    outputs = tf.constant([[0.1, 0.8], [0.4, 0.6]])
    labels = tf.constant([[0.0, 1.0], [1.0, 0.0]])
    result = loss._compute_tf_loss(outputs, labels).numpy()
    expected = [[0.1, 0.2], [0.6, 0.6]]
    assert np.allclose(expected, result)

  @unittest.skipIf(not has_pytorch, 'PyTorch is not installed')
  def test_l1_loss_pytorch(self):
    """Test L1Loss."""
    loss = losses.L1Loss()
    outputs = torch.tensor([[0.1, 0.8], [0.4, 0.6]])
    labels = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    result = loss._create_pytorch_loss()(outputs, labels).numpy()
    expected = [[0.1, 0.2], [0.6, 0.6]]
    assert np.allclose(expected, result)

  @unittest.skipIf(not has_tensorflow, 'TensorFlow is not installed')
  def test_l2_loss_tf(self):
    """Test L2Loss."""
    loss = losses.L2Loss()
    outputs = tf.constant([[0.1, 0.8], [0.4, 0.6]])
    labels = tf.constant([[0.0, 1.0], [1.0, 0.0]])
    result = loss._compute_tf_loss(outputs, labels).numpy()
    expected = [[0.1**2, 0.2**2], [0.6**2, 0.6**2]]
    assert np.allclose(expected, result)

  @unittest.skipIf(not has_pytorch, 'PyTorch is not installed')
  def test_l2_loss_pytorch(self):
    """Test L2Loss."""
    loss = losses.L2Loss()
    outputs = torch.tensor([[0.1, 0.8], [0.4, 0.6]])
    labels = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    result = loss._create_pytorch_loss()(outputs, labels).numpy()
    expected = [[0.1**2, 0.2**2], [0.6**2, 0.6**2]]
    assert np.allclose(expected, result)

  @unittest.skipIf(not has_tensorflow, 'TensorFlow is not installed')
  def test_hinge_loss_tf(self):
    """Test HingeLoss."""
    loss = losses.HingeLoss()
    outputs = tf.constant([[0.1, 0.8], [0.4, 0.6]])
    labels = tf.constant([[1.0, -1.0], [-1.0, 1.0]])
    result = loss._compute_tf_loss(outputs, labels).numpy()
    expected = [np.mean([0.9, 1.8]), np.mean([1.4, 0.4])]
    assert np.allclose(expected, result)

  @unittest.skipIf(not has_pytorch, 'PyTorch is not installed')
  def test_hinge_loss_pytorch(self):
    """Test HingeLoss."""
    loss = losses.HingeLoss()
    outputs = torch.tensor([[0.1, 0.8], [0.4, 0.6]])
    labels = torch.tensor([[1.0, -1.0], [-1.0, 1.0]])
    result = loss._create_pytorch_loss()(outputs, labels).numpy()
    expected = [np.mean([0.9, 1.8]), np.mean([1.4, 0.4])]
    assert np.allclose(expected, result)

  @unittest.skipIf(not has_tensorflow, 'TensorFlow is not installed')
  def test_binary_cross_entropy_tf(self):
    """Test BinaryCrossEntropy."""
    loss = losses.BinaryCrossEntropy()
    outputs = tf.constant([[0.1, 0.8], [0.4, 0.6]])
    labels = tf.constant([[0.0, 1.0], [1.0, 0.0]])
    result = loss._compute_tf_loss(outputs, labels).numpy()
    expected = [
        -np.mean([np.log(0.9), np.log(0.8)]),
        -np.mean([np.log(0.4), np.log(0.4)])
    ]
    assert np.allclose(expected, result)

  @unittest.skipIf(not has_pytorch, 'PyTorch is not installed')
  def test_binary_cross_entropy_pytorch(self):
    """Test BinaryCrossEntropy."""
    loss = losses.BinaryCrossEntropy()
    outputs = torch.tensor([[0.1, 0.8], [0.4, 0.6]])
    labels = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    result = loss._create_pytorch_loss()(outputs, labels).numpy()
    expected = [
        -np.mean([np.log(0.9), np.log(0.8)]),
        -np.mean([np.log(0.4), np.log(0.4)])
    ]
    assert np.allclose(expected, result)

  @unittest.skipIf(not has_tensorflow, 'TensorFlow is not installed')
  def test_categorical_cross_entropy_tf(self):
    """Test CategoricalCrossEntropy."""
    loss = losses.CategoricalCrossEntropy()
    outputs = tf.constant([[0.2, 0.8], [0.4, 0.6]])
    labels = tf.constant([[0.0, 1.0], [1.0, 0.0]])
    result = loss._compute_tf_loss(outputs, labels).numpy()
    expected = [-np.log(0.8), -np.log(0.4)]
    assert np.allclose(expected, result)

  @unittest.skipIf(not has_pytorch, 'PyTorch is not installed')
  def test_categorical_cross_entropy_pytorch(self):
    """Test CategoricalCrossEntropy."""
    loss = losses.CategoricalCrossEntropy()
    outputs = torch.tensor([[0.2, 0.8], [0.4, 0.6]])
    labels = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    result = loss._create_pytorch_loss()(outputs, labels).numpy()
    expected = [-np.log(0.8), -np.log(0.4)]
    assert np.allclose(expected, result)

  @unittest.skipIf(not has_tensorflow, 'TensorFlow is not installed')
  def test_sigmoid_cross_entropy_tf(self):
    """Test SigmoidCrossEntropy."""
    loss = losses.SigmoidCrossEntropy()
    y = [[0.1, 0.8], [0.4, 0.6]]
    outputs = tf.constant(y)
    labels = tf.constant([[0.0, 1.0], [1.0, 0.0]])
    result = loss._compute_tf_loss(outputs, labels).numpy()
    sigmoid = 1.0 / (1.0 + np.exp(-np.array(y)))
    expected = [[-np.log(1 - sigmoid[0, 0]), -np.log(sigmoid[0, 1])],
                [-np.log(sigmoid[1, 0]), -np.log(1 - sigmoid[1, 1])]]
    assert np.allclose(expected, result)

  @unittest.skipIf(not has_pytorch, 'PyTorch is not installed')
  def test_sigmoid_cross_entropy_pytorch(self):
    """Test SigmoidCrossEntropy."""
    loss = losses.SigmoidCrossEntropy()
    y = [[0.1, 0.8], [0.4, 0.6]]
    outputs = torch.tensor(y)
    labels = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    result = loss._create_pytorch_loss()(outputs, labels).numpy()
    sigmoid = 1.0 / (1.0 + np.exp(-np.array(y)))
    expected = [[-np.log(1 - sigmoid[0, 0]), -np.log(sigmoid[0, 1])],
                [-np.log(sigmoid[1, 0]), -np.log(1 - sigmoid[1, 1])]]
    assert np.allclose(expected, result)

  @unittest.skipIf(not has_tensorflow, 'TensorFlow is not installed')
  def test_softmax_cross_entropy_tf(self):
    """Test SoftmaxCrossEntropy."""
    loss = losses.SoftmaxCrossEntropy()
    y = np.array([[0.1, 0.8], [0.4, 0.6]])
    outputs = tf.constant(y)
    labels = tf.constant([[0.0, 1.0], [1.0, 0.0]])
    result = loss._compute_tf_loss(outputs, labels).numpy()
    softmax = np.exp(y) / np.expand_dims(np.sum(np.exp(y), axis=1), 1)
    expected = [-np.log(softmax[0, 1]), -np.log(softmax[1, 0])]
    assert np.allclose(expected, result)

  @unittest.skipIf(not has_pytorch, 'PyTorch is not installed')
  def test_softmax_cross_entropy_pytorch(self):
    """Test SoftmaxCrossEntropy."""
    loss = losses.SoftmaxCrossEntropy()
    y = np.array([[0.1, 0.8], [0.4, 0.6]])
    outputs = torch.tensor(y)
    labels = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    result = loss._create_pytorch_loss()(outputs, labels).numpy()
    softmax = np.exp(y) / np.expand_dims(np.sum(np.exp(y), axis=1), 1)
    expected = [-np.log(softmax[0, 1]), -np.log(softmax[1, 0])]
    assert np.allclose(expected, result)

  @unittest.skipIf(not has_tensorflow, 'TensorFlow is not installed')
  def test_sparse_softmax_cross_entropy_tf(self):
    """Test SparseSoftmaxCrossEntropy."""
    loss = losses.SparseSoftmaxCrossEntropy()
    y = np.array([[0.1, 0.8], [0.4, 0.6]])
    outputs = tf.constant(y)
    labels = tf.constant([1, 0])
    result = loss._compute_tf_loss(outputs, labels).numpy()
    softmax = np.exp(y) / np.expand_dims(np.sum(np.exp(y), axis=1), 1)
    expected = [-np.log(softmax[0, 1]), -np.log(softmax[1, 0])]
    assert np.allclose(expected, result)

  @unittest.skipIf(not has_pytorch, 'PyTorch is not installed')
  def test_sparse_softmax_cross_entropy_pytorch(self):
    """Test SparseSoftmaxCrossEntropy."""
    loss = losses.SparseSoftmaxCrossEntropy()
    y = np.array([[0.1, 0.8], [0.4, 0.6]])
    outputs = torch.tensor(y)
    labels = torch.tensor([1, 0])
    result = loss._create_pytorch_loss()(outputs, labels).numpy()
    softmax = np.exp(y) / np.expand_dims(np.sum(np.exp(y), axis=1), 1)
    expected = [-np.log(softmax[0, 1]), -np.log(softmax[1, 0])]
    assert np.allclose(expected, result)
