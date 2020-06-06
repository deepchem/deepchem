import tensorflow as tf


class Loss:
  """A loss function for use in training models."""

  def __call__(self, output, labels):
    """Compute the loss function.

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


class L1Loss(Loss):
  """The absolute difference between the true and predicted values."""

  def __call__(self, output, labels):
    output, labels = _make_shapes_consistent(output, labels)
    output, labels = _ensure_float(output, labels)
    return tf.abs(output - labels)


class L2Loss(Loss):
  """The squared difference between the true and predicted values."""

  def __call__(self, output, labels):
    output, labels = _make_shapes_consistent(output, labels)
    output, labels = _ensure_float(output, labels)
    return tf.square(output - labels)


class HingeLoss(Loss):
  """The hinge loss function.

  The 'output' argument should contain logits, and all elements of 'labels'
  should equal 0 or 1.
  """

  def __call__(self, output, labels):
    output, labels = _make_shapes_consistent(output, labels)
    return tf.keras.losses.hinge(labels, output)


class BinaryCrossEntropy(Loss):
  """The cross entropy between pairs of probabilities.

  The arguments should each have shape (batch_size) or (batch_size, tasks) and
  contain probabilities.
  """

  def __call__(self, output, labels):
    output, labels = _make_shapes_consistent(output, labels)
    output, labels = _ensure_float(output, labels)
    return tf.keras.losses.binary_crossentropy(labels, output)


class CategoricalCrossEntropy(Loss):
  """The cross entropy between two probability distributions.

  The arguments should each have shape (batch_size, classes) or
  (batch_size, tasks, classes), and represent a probability distribution over
  classes.
  """

  def __call__(self, output, labels):
    output, labels = _make_shapes_consistent(output, labels)
    output, labels = _ensure_float(output, labels)
    return tf.keras.losses.categorical_crossentropy(labels, output)


class SigmoidCrossEntropy(Loss):
  """The cross entropy between pairs of probabilities.

  The arguments should each have shape (batch_size) or (batch_size, tasks).  The
  labels should be probabilities, while the outputs should be logits that are
  converted to probabilities using a sigmoid function.
  """

  def __call__(self, output, labels):
    output, labels = _make_shapes_consistent(output, labels)
    output, labels = _ensure_float(output, labels)
    return tf.nn.sigmoid_cross_entropy_with_logits(labels, output)


class SoftmaxCrossEntropy(Loss):
  """The cross entropy between two probability distributions.

  The arguments should each have shape (batch_size, classes) or
  (batch_size, tasks, classes).  The labels should be probabilities, while the
  outputs should be logits that are converted to probabilities using a softmax
  function.
  """

  def __call__(self, output, labels):
    output, labels = _make_shapes_consistent(output, labels)
    output, labels = _ensure_float(output, labels)
    return tf.nn.softmax_cross_entropy_with_logits(labels, output)


class SparseSoftmaxCrossEntropy(Loss):
  """The cross entropy between two probability distributions.

  The labels should have shape (batch_size) or (batch_size, tasks), and be
  integer class labels.  The outputs have shape (batch_size, classes) or
  (batch_size, tasks, classes) and be logits that are converted to probabilities
  using a softmax function.
  """

  def __call__(self, output, labels):
    labels = tf.cast(labels, tf.int32)
    return tf.nn.sparse_softmax_cross_entropy_with_logits(labels, output)


def _make_shapes_consistent(output, labels):
  """Try to make inputs have the same shape by adding dimensions of size 1."""
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


def _ensure_float(output, labels):
  """Make sure the outputs and labels are both floating point types."""
  if output.dtype not in (tf.float32, tf.float64):
    output = tf.cast(output, tf.float32)
  if labels.dtype not in (tf.float32, tf.float64):
    labels = tf.cast(labels, tf.float32)
  return (output, labels)
