class Loss:
  """A loss function for use in training models."""

  def _compute_tf_loss(self, output, labels):
    """Compute the loss function for TensorFlow tensors.

    The inputs are tensors containing the model's outputs and the labels for a
    batch.  The return value should be a tensor of shape (batch_size) or
    (batch_size, tasks) containing the value of the loss function on each
    sample or sample/task.

    Parameters
    ----------
    output: tensor
      the output of the model
    labels: tensor
      the expected output

    Returns
    -------
    The value of the loss function on each sample or sample/task pair
    """
    raise NotImplementedError("Subclasses must implement this")

  def _create_pytorch_loss(self):
    """Create a PyTorch loss function."""
    raise NotImplementedError("Subclasses must implement this")


class L1Loss(Loss):
  """The absolute difference between the true and predicted values."""

  def _compute_tf_loss(self, output, labels):
    import tensorflow as tf
    output, labels = _make_tf_shapes_consistent(output, labels)
    output, labels = _ensure_float(output, labels)
    return tf.abs(output - labels)

  def _create_pytorch_loss(self):
    import torch
    return torch.nn.L1Loss(reduction='none')


class L2Loss(Loss):
  """The squared difference between the true and predicted values."""

  def _compute_tf_loss(self, output, labels):
    import tensorflow as tf
    output, labels = _make_tf_shapes_consistent(output, labels)
    output, labels = _ensure_float(output, labels)
    return tf.square(output - labels)

  def _create_pytorch_loss(self):
    import torch
    return torch.nn.MSELoss(reduction='none')


class HingeLoss(Loss):
  """The hinge loss function.

  The 'output' argument should contain logits, and all elements of 'labels'
  should equal 0 or 1.
  """

  def _compute_tf_loss(self, output, labels):
    import tensorflow as tf
    output, labels = _make_tf_shapes_consistent(output, labels)
    return tf.keras.losses.hinge(labels, output)

  def _create_pytorch_loss(self):
    import torch

    def loss(output, labels):
      output, labels = _make_pytorch_shapes_consistent(output, labels)
      return torch.mean(torch.clamp(1 - labels * output, min=0), dim=-1)

    return loss


class BinaryCrossEntropy(Loss):
  """The cross entropy between pairs of probabilities.

  The arguments should each have shape (batch_size) or (batch_size, tasks) and
  contain probabilities.
  """

  def _compute_tf_loss(self, output, labels):
    import tensorflow as tf
    output, labels = _make_tf_shapes_consistent(output, labels)
    output, labels = _ensure_float(output, labels)
    return tf.keras.losses.binary_crossentropy(labels, output)

  def _create_pytorch_loss(self):
    import torch
    bce = torch.nn.BCELoss(reduction='none')

    def loss(output, labels):
      output, labels = _make_pytorch_shapes_consistent(output, labels)
      return torch.mean(bce(output, labels), dim=-1)

    return loss


class CategoricalCrossEntropy(Loss):
  """The cross entropy between two probability distributions.

  The arguments should each have shape (batch_size, classes) or
  (batch_size, tasks, classes), and represent a probability distribution over
  classes.
  """

  def _compute_tf_loss(self, output, labels):
    import tensorflow as tf
    output, labels = _make_tf_shapes_consistent(output, labels)
    output, labels = _ensure_float(output, labels)
    return tf.keras.losses.categorical_crossentropy(labels, output)

  def _create_pytorch_loss(self):
    import torch

    def loss(output, labels):
      output, labels = _make_pytorch_shapes_consistent(output, labels)
      return -torch.sum(labels * torch.log(output), dim=-1)

    return loss


class SigmoidCrossEntropy(Loss):
  """The cross entropy between pairs of probabilities.

  The arguments should each have shape (batch_size) or (batch_size, tasks).  The
  labels should be probabilities, while the outputs should be logits that are
  converted to probabilities using a sigmoid function.
  """

  def _compute_tf_loss(self, output, labels):
    import tensorflow as tf
    output, labels = _make_tf_shapes_consistent(output, labels)
    output, labels = _ensure_float(output, labels)
    return tf.nn.sigmoid_cross_entropy_with_logits(labels, output)

  def _create_pytorch_loss(self):
    import torch
    bce = torch.nn.BCEWithLogitsLoss(reduction='none')

    def loss(output, labels):
      output, labels = _make_pytorch_shapes_consistent(output, labels)
      return bce(output, labels)

    return loss


class SoftmaxCrossEntropy(Loss):
  """The cross entropy between two probability distributions.

  The arguments should each have shape (batch_size, classes) or
  (batch_size, tasks, classes).  The labels should be probabilities, while the
  outputs should be logits that are converted to probabilities using a softmax
  function.
  """

  def _compute_tf_loss(self, output, labels):
    import tensorflow as tf
    output, labels = _make_tf_shapes_consistent(output, labels)
    output, labels = _ensure_float(output, labels)
    return tf.nn.softmax_cross_entropy_with_logits(labels, output)

  def _create_pytorch_loss(self):
    import torch
    ls = torch.nn.LogSoftmax(dim=1)

    def loss(output, labels):
      output, labels = _make_pytorch_shapes_consistent(output, labels)
      return -torch.sum(labels * ls(output), dim=-1)

    return loss


class SparseSoftmaxCrossEntropy(Loss):
  """The cross entropy between two probability distributions.

  The labels should have shape (batch_size) or (batch_size, tasks), and be
  integer class labels.  The outputs have shape (batch_size, classes) or
  (batch_size, tasks, classes) and be logits that are converted to probabilities
  using a softmax function.
  """

  def _compute_tf_loss(self, output, labels):
    import tensorflow as tf
    labels = tf.cast(labels, tf.int32)
    return tf.nn.sparse_softmax_cross_entropy_with_logits(labels, output)

  def _create_pytorch_loss(self):
    import torch
    return torch.nn.CrossEntropyLoss(reduction='none')


def _make_tf_shapes_consistent(output, labels):
  """Try to make inputs have the same shape by adding dimensions of size 1."""
  import tensorflow as tf
  shape1 = output.shape
  shape2 = labels.shape
  len1 = len(shape1)
  len2 = len(shape2)
  if len1 == len2:
    return (output, labels)
  if isinstance(shape1, tf.TensorShape):
    shape1 = tuple(shape1.as_list())
  if isinstance(shape2, tf.TensorShape):
    shape2 = tuple(shape2.as_list())
  if len1 > len2 and all(i == 1 for i in shape1[len2:]):
    for i in range(len1 - len2):
      labels = tf.expand_dims(labels, -1)
    return (output, labels)
  if len2 > len1 and all(i == 1 for i in shape2[len1:]):
    for i in range(len2 - len1):
      output = tf.expand_dims(output, -1)
    return (output, labels)
  raise ValueError("Incompatible shapes for outputs and labels: %s versus %s" %
                   (str(shape1), str(shape2)))


def _make_pytorch_shapes_consistent(output, labels):
  """Try to make inputs have the same shape by adding dimensions of size 1."""
  import torch
  shape1 = output.shape
  shape2 = labels.shape
  len1 = len(shape1)
  len2 = len(shape2)
  if len1 == len2:
    return (output, labels)
  shape1 = tuple(shape1)
  shape2 = tuple(shape2)
  if len1 > len2 and all(i == 1 for i in shape1[len2:]):
    for i in range(len1 - len2):
      labels = torch.unsqueeze(labels, -1)
    return (output, labels)
  if len2 > len1 and all(i == 1 for i in shape2[len1:]):
    for i in range(len2 - len1):
      output = torch.unsqueeze(output, -1)
    return (output, labels)
  raise ValueError("Incompatible shapes for outputs and labels: %s versus %s" %
                   (str(shape1), str(shape2)))


def _ensure_float(output, labels):
  """Make sure the outputs and labels are both floating point types."""
  import tensorflow as tf
  if output.dtype not in (tf.float32, tf.float64):
    output = tf.cast(output, tf.float32)
  if labels.dtype not in (tf.float32, tf.float64):
    labels = tf.cast(labels, tf.float32)
  return (output, labels)
