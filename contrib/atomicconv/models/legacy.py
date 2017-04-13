"""Helper operations and classes for general model building.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import collections
import pickle
import os
import time
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
import tempfile
from deepchem.models import Model
from deepchem.metrics import from_one_hot
from deepchem.nn import model_ops
from deepchem.models.tensorflow_models import utils as tf_utils
from deepchem.trans import undo_transforms
from deepchem.utils.save import log
from deepchem.utils.evaluate import Evaluator
from deepchem.data import pad_features
from tensorflow.contrib.layers.python.layers import batch_norm


def softmax(x):
  """Simple numpy softmax implementation
  """
  # (n_samples, n_classes)
  if len(x.shape) == 2:
    row_max = np.max(x, axis=1)
    x -= row_max.reshape((x.shape[0], 1))
    x = np.exp(x)
    row_sum = np.sum(x, axis=1)
    x /= row_sum.reshape((x.shape[0], 1))
  # (n_samples, n_tasks, n_classes)
  elif len(x.shape) == 3:
    row_max = np.max(x, axis=2)
    x -= row_max.reshape(x.shape[:2] + (1,))
    x = np.exp(x)
    row_sum = np.sum(x, axis=2)
    x /= row_sum.reshape(x.shape[:2] + (1,))
  return x


class TensorflowGraph(object):
  """Simple class that holds information needed to run Tensorflow graph."""

  def __init__(self, graph, session, name_scopes, output, labels, weights,
               loss):
    self.graph = graph
    self.session = session
    self.name_scopes = name_scopes
    self.output = output
    self.labels = labels
    self.weights = weights
    self.loss = loss

  @staticmethod
  def get_placeholder_scope(graph, name_scopes):
    """Gets placeholder scope."""
    placeholder_root = "placeholders"
    return TensorflowGraph.shared_name_scope(placeholder_root, graph,
                                             name_scopes)

  @staticmethod
  def shared_name_scope(name, graph, name_scopes):
    """Returns a singleton TensorFlow scope with the given name.

    Used to prevent '_1'-appended scopes when sharing scopes with child classes.

    Args:
      name: String. Name scope for group of operations.
    Returns:
      tf.name_scope with the provided name.
    """
    with graph.as_default():
      if name not in name_scopes:
        with tf.name_scope(name) as scope:
          name_scopes[name] = scope
      return tf.name_scope(name_scopes[name])

  @staticmethod
  def get_feed_dict(named_values):
    feed_dict = {}
    placeholder_root = "placeholders"
    for name, value in named_values.items():
      feed_dict['{}/{}:0'.format(placeholder_root, name)] = value
    return feed_dict


class TensorflowGraphModel(Model):
  """Parent class for deepchem Tensorflow models.
  
  Classifier:
    n_classes

  Has the following attributes:

    placeholder_root: String placeholder prefix, used to create
      placeholder_scope.

  Generic base class for defining, training, and evaluating TensorflowGraphs.

  Subclasses must implement the following methods:
    build
    add_output_ops
    add_training_cost 

  Args:
    train: If True, model is in training mode.
    logdir: Directory for output files.
  """

  def __init__(self,
               n_tasks,
               n_features,
               logdir=None,
               layer_sizes=[1000],
               weight_init_stddevs=[.02],
               bias_init_consts=[1.],
               penalty=0.0,
               penalty_type="l2",
               dropouts=[0.5],
               learning_rate=.001,
               momentum=.9,
               optimizer="adam",
               batch_size=50,
               n_classes=2,
               pad_batches=False,
               verbose=True,
               seed=None,
               **kwargs):
    """Constructs the computational graph.

    This function constructs the computational graph for the model. It relies
    subclassed methods (build/cost) to construct specific graphs.

    Parameters
    ----------
    n_tasks: int
      Number of tasks
    n_features: int
      Number of features.
    logdir: str
      Location to save data
    layer_sizes: list
      List of layer sizes.
    weight_init_stddevs: list
      List of standard deviations for weights (sampled from zero-mean
      gaussians). One for each layer.
    bias_init_consts: list
      List of bias initializations. One for each layer.
    penalty: float
      Amount of penalty (l2 or l1 applied)
    penalty_type: str
      Either "l2" or "l1"
    dropouts: list
      List of dropout amounts. One for each layer.
    learning_rate: float
      Learning rate for model.
    momentum: float
      Momentum. Only applied if optimizer=="momentum"
    optimizer: str
      Type of optimizer applied.
    batch_size: int
      Size of minibatches for training.
    n_classes: int
      Number of classes if this is for classification.
      TODO(rbharath): Move this argument to TensorflowClassifier
    verbose: True 
      Perform logging.
    seed: int
      If not none, is used as random seed for tensorflow. 
    """
    # Save hyperparameters
    self.n_tasks = n_tasks
    self.n_features = n_features
    self.layer_sizes = layer_sizes
    self.weight_init_stddevs = weight_init_stddevs
    self.bias_init_consts = bias_init_consts
    self.penalty = penalty
    self.penalty_type = penalty_type
    self.dropouts = dropouts
    self.learning_rate = learning_rate
    self.momentum = momentum
    self.optimizer = optimizer
    self.batch_size = batch_size
    self.n_classes = n_classes
    self.pad_batches = pad_batches
    self.verbose = verbose
    self.seed = seed

    if logdir is not None:
      if not os.path.exists(logdir):
        os.makedirs(logdir)
    else:
      logdir = tempfile.mkdtemp()
    self.logdir = logdir

    # Guard variable to make sure we don't Restore() this model
    # from a disk checkpoint more than once.
    self._restored_model = False
    # Path to save checkpoint files, which matches the
    # replicated supervisor's default path.
    self._save_path = os.path.join(logdir, 'model.ckpt')

    self.train_graph = self.construct_graph(training=True, seed=self.seed)
    self.eval_graph = self.construct_graph(training=False, seed=self.seed)

  def save(self):
    """
    No-op since tf models save themselves during fit()
    """
    pass

  def reload(self):
    """
    Loads model from disk. Thin wrapper around restore() for consistency.
    """
    self.restore()

  def get_num_tasks(self):
    return self.n_tasks

  def construct_graph(self, training, seed):
    """Returns a TensorflowGraph object."""
    graph = tf.Graph()

    # Lazily created by _get_shared_session().
    shared_session = None

    # Cache of TensorFlow scopes, to prevent '_1' appended scope names
    # when subclass-overridden methods use the same scopes.
    name_scopes = {}

    # Setup graph
    with graph.as_default():
      if seed is not None:
        tf.set_random_seed(seed)
      output = self.build(graph, name_scopes, training)
      labels = self.add_label_placeholders(graph, name_scopes)
      weights = self.add_example_weight_placeholders(graph, name_scopes)

    if training:
      loss = self.add_training_cost(graph, name_scopes, output, labels, weights)
    else:
      loss = None
      output = self.add_output_ops(graph, output)  # add softmax heads
    return TensorflowGraph(
        graph=graph,
        session=shared_session,
        name_scopes=name_scopes,
        output=output,
        labels=labels,
        weights=weights,
        loss=loss)

  def add_training_cost(self, graph, name_scopes, output, labels, weights):
    with graph.as_default():
      epsilon = 1e-3  # small float to avoid dividing by zero
      weighted_costs = []  # weighted costs for each example
      gradient_costs = []  # costs used for gradient calculation

      with TensorflowGraph.shared_name_scope('costs', graph, name_scopes):
        for task in range(self.n_tasks):
          task_str = str(task).zfill(len(str(self.n_tasks)))
          with TensorflowGraph.shared_name_scope('cost_{}'.format(task_str),
                                                 graph, name_scopes):
            with tf.name_scope('weighted'):
              weighted_cost = self.cost(output[task], labels[task],
                                        weights[task])
              weighted_costs.append(weighted_cost)

            with tf.name_scope('gradient'):
              # Note that we divide by the batch size and not the number of
              # non-zero weight examples in the batch.  Also, instead of using
              # tf.reduce_mean (which can put ops on the CPU) we explicitly
              # calculate with div/sum so it stays on the GPU.
              gradient_cost = tf.div(
                  tf.reduce_sum(weighted_cost), self.batch_size)
              gradient_costs.append(gradient_cost)

        # aggregated costs
        with TensorflowGraph.shared_name_scope('aggregated', graph,
                                               name_scopes):
          with tf.name_scope('gradient'):
            loss = tf.add_n(gradient_costs)

          # weight decay
          if self.penalty != 0.0:
            penalty = model_ops.weight_decay(self.penalty_type, self.penalty)
            loss += penalty

      return loss

  def fit(self,
          dataset,
          nb_epoch=10,
          max_checkpoints_to_keep=5,
          log_every_N_batches=50,
          **kwargs):
    """Fit the model.

    Parameters
    ---------- 
    dataset: dc.data.Dataset
      Dataset object holding training data 
    nb_epoch: 10
      Number of training epochs.
    max_checkpoints_to_keep: int
      Maximum number of checkpoints to keep; older checkpoints will be deleted.
    log_every_N_batches: int
      Report every N batches. Useful for training on very large datasets,
      where epochs can take long time to finish.

    Raises
    ------
    AssertionError
      If model is not in training mode.
    """
    ############################################################## TIMING
    time1 = time.time()
    ############################################################## TIMING
    log("Training for %d epochs" % nb_epoch, self.verbose)
    with self.train_graph.graph.as_default():
      train_op = self.get_training_op(self.train_graph.graph,
                                      self.train_graph.loss)
      with self._get_shared_session(train=True) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=max_checkpoints_to_keep)
        # Save an initial checkpoint.
        saver.save(sess, self._save_path, global_step=0)
        for epoch in range(nb_epoch):
          avg_loss, n_batches = 0., 0
          for ind, (X_b, y_b, w_b, ids_b) in enumerate(
              # Turns out there are valid cases where we don't want pad-batches
              # on by default.
              #dataset.iterbatches(batch_size, pad_batches=True)):
              dataset.iterbatches(
                  self.batch_size, pad_batches=self.pad_batches)):
            if ind % log_every_N_batches == 0:
              log("On batch %d" % ind, self.verbose)
            # Run training op.
            feed_dict = self.construct_feed_dict(X_b, y_b, w_b, ids_b)
            fetches = self.train_graph.output + [
                train_op, self.train_graph.loss
            ]
            fetched_values = sess.run(fetches, feed_dict=feed_dict)
            output = fetched_values[:len(self.train_graph.output)]
            loss = fetched_values[-1]
            avg_loss += loss
            y_pred = np.squeeze(np.array(output))
            y_b = y_b.flatten()
            n_batches += 1
          saver.save(sess, self._save_path, global_step=epoch)
          avg_loss = float(avg_loss) / n_batches
          log('Ending epoch %d: Average loss %g' % (epoch, avg_loss),
              self.verbose)
        # Always save a final checkpoint when complete.
        saver.save(sess, self._save_path, global_step=epoch + 1)
    ############################################################## TIMING
    time2 = time.time()
    print("TIMING: model fitting took %0.3f s" % (time2 - time1), self.verbose)
    ############################################################## TIMING

  def add_output_ops(self, graph, output):
    """Replace logits with softmax outputs."""
    with graph.as_default():
      softmax = []
      with tf.name_scope('inference'):
        for i, logits in enumerate(output):
          softmax.append(tf.nn.softmax(logits, name='softmax_%d' % i))
      output = softmax
    return output

  def build(self, graph, name_scopes, training):
    """Define the core graph.

    NOTE(user): Operations defined here should be in their own name scope to
    avoid any ambiguity when restoring checkpoints.
    Raises:
      NotImplementedError: if not overridden by concrete subclass.
    """
    raise NotImplementedError('Must be overridden by concrete subclass')

  def construct_feed_dict(self, X_b, y_b=None, w_b=None, ids_b=None):
    """Transform a minibatch of data into a feed_dict.

    Raises:
      NotImplementedError: if not overridden by concrete subclass.
    """
    raise NotImplementedError('Must be overridden by concrete subclass')

  def add_label_placeholders(self, graph, name_scopes):
    """Add Placeholders for labels for each task.

    This method creates the following Placeholders for each task:
      labels_%d: Float label tensor. For classification tasks, this tensor will
        have shape batch_size x n_classes. For regression tasks, this tensor
        will have shape batch_size.

    Raises:
      NotImplementedError: if not overridden by concrete subclass.
    """
    raise NotImplementedError('Must be overridden by concrete subclass')

  def add_example_weight_placeholders(self, graph, name_scopes):
    """Add Placeholders for example weights for each task.

    This method creates the following Placeholders for each task:
      weights_%d: Label tensor with shape batch_size.

    Placeholders are wrapped in identity ops to avoid the error caused by
    feeding and fetching the same tensor.
    """
    weights = []
    placeholder_scope = TensorflowGraph.get_placeholder_scope(graph,
                                                              name_scopes)
    with placeholder_scope:
      for task in range(self.n_tasks):
        weights.append(
            tf.identity(
                tf.placeholder(
                    tf.float32, shape=[None], name='weights_%d' % task)))
    return weights

  def cost(self, output, labels, weights):
    """Calculate single-task training cost for a batch of examples.

    Args:
      output: Tensor with model outputs.
      labels: Tensor with true labels.
      weights: Tensor with shape batch_size containing example weights.

    Returns:
      A tensor with shape batch_size containing the weighted cost for each
      example. For use in subclasses that want to calculate additional costs.
    """
    raise NotImplementedError('Must be overridden by concrete subclass')

  def get_training_op(self, graph, loss):
    """Get training op for applying gradients to variables.

    Subclasses that need to do anything fancy with gradients should override
    this method.

    Returns:
    A training op.
    """
    with graph.as_default():
      opt = model_ops.optimizer(self.optimizer, self.learning_rate,
                                self.momentum)
      return opt.minimize(loss, name='train')

  def _get_shared_session(self, train):
    # allow_soft_placement=True allows ops without a GPU implementation
    # to run on the CPU instead.
    if train:
      if not self.train_graph.session:
        config = tf.ConfigProto(allow_soft_placement=True)
        self.train_graph.session = tf.Session(config=config)
      return self.train_graph.session
    else:
      if not self.eval_graph.session:
        config = tf.ConfigProto(allow_soft_placement=True)
        self.eval_graph.session = tf.Session(config=config)
      return self.eval_graph.session

  def restore(self):
    """Restores the model from the provided training checkpoint.

    Args:
      checkpoint: string. Path to checkpoint file.
    """
    if self._restored_model:
      return
    with self.eval_graph.graph.as_default():
      last_checkpoint = self._find_last_checkpoint()
      # TODO(rbharath): Is setting train=False right here?
      saver = tf.train.Saver()
      saver.restore(self._get_shared_session(train=False), last_checkpoint)
      self._restored_model = True

  def predict(self, dataset, transformers=[]):
    """
    Uses self to make predictions on provided Dataset object.

    Returns:
      y_pred: numpy ndarray of shape (n_samples,)
    """
    y_preds = []
    n_tasks = self.get_num_tasks()
    ind = 0

    for (X_batch, _, _, ids_batch) in dataset.iterbatches(
        self.batch_size, deterministic=True):
      n_samples = len(X_batch)
      y_pred_batch = self.predict_on_batch(X_batch)
      # Discard any padded predictions
      y_pred_batch = y_pred_batch[:n_samples]
      y_pred_batch = np.reshape(y_pred_batch, (n_samples, n_tasks))
      y_pred_batch = undo_transforms(y_pred_batch, transformers)
      y_preds.append(y_pred_batch)
    y_pred = np.vstack(y_preds)

    # The iterbatches does padding with zero-weight examples on the last batch.
    # Remove padded examples.
    n_samples = len(dataset)
    y_pred = np.reshape(y_pred, (n_samples, n_tasks))
    # Special case to handle singletasks.
    if n_tasks == 1:
      y_pred = np.reshape(y_pred, (n_samples,))
    return y_pred

  def predict_proba(self, dataset, transformers=[], n_classes=2):
    """
    TODO: Do transformers even make sense here?

    Returns:
      y_pred: numpy ndarray of shape (n_samples, n_classes*n_tasks)
    """
    y_preds = []
    n_tasks = self.get_num_tasks()

    for (X_batch, y_batch, w_batch, ids_batch) in dataset.iterbatches(
        self.batch_size, deterministic=True):
      n_samples = len(X_batch)
      y_pred_batch = self.predict_proba_on_batch(X_batch)
      y_pred_batch = y_pred_batch[:n_samples]
      y_pred_batch = np.reshape(y_pred_batch, (n_samples, n_tasks, n_classes))
      y_pred_batch = undo_transforms(y_pred_batch, transformers)
      y_preds.append(y_pred_batch)
    y_pred = np.vstack(y_preds)
    # The iterbatches does padding with zero-weight examples on the last batch.
    # Remove padded examples.
    n_samples = len(dataset)
    y_pred = y_pred[:n_samples]
    y_pred = np.reshape(y_pred, (n_samples, n_tasks, n_classes))
    return y_pred

  # TODO(rbharath): Verify this can be safely removed.
  #def evaluate(self, dataset, metrics, transformers=[]):
  #  """
  #  Evaluates the performance of this model on specified dataset.
  #
  #  Parameters
  #  ----------
  #  dataset: dc.data.Dataset
  #    Dataset object.
  #  metric: deepchem.metrics.Metric
  #    Evaluation metric
  #  transformers: list
  #    List of deepchem.transformers.Transformer

  #  Returns
  #  -------
  #  dict
  #    Maps tasks to scores under metric.
  #  """
  #  evaluator = Evaluator(self, dataset, transformers)
  #  scores = evaluator.compute_model_performance(metrics)
  #  return scores

  def _find_last_checkpoint(self):
    """Finds last saved checkpoint."""
    highest_num, last_checkpoint = -np.inf, None
    for filename in os.listdir(self.logdir):
      # checkpoints look like logdir/model.ckpt-N
      # self._save_path is "logdir/model.ckpt"
      if os.path.basename(self._save_path) in filename:
        try:
          N = int(filename.split("-")[1].split(".")[0])
          if N > highest_num:
            highest_num = N
            last_checkpoint = "model.ckpt-" + str(N)
        except ValueError:
          pass
    return os.path.join(self.logdir, last_checkpoint)


class TensorflowClassifier(TensorflowGraphModel):
  """Classification model.

  Subclasses must set the following attributes:
    output: logits op(s) used for computing classification loss and predicted
      class probabilities for each task.
  """

  def get_task_type(self):
    return "classification"

  def cost(self, logits, labels, weights):
    """Calculate single-task training cost for a batch of examples.

    Args:
      logits: Tensor with shape batch_size x n_classes containing logits.
      labels: Tensor with shape batch_size x n_classes containing true labels
        in a one-hot encoding.
      weights: Tensor with shape batch_size containing example weights.

    Returns:
      A tensor with shape batch_size containing the weighted cost for each
      example.
    """
    return tf.multiply(
        tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels),
        weights)

  def add_label_placeholders(self, graph, name_scopes):
    """Add Placeholders for labels for each task.

    This method creates the following Placeholders for each task:
      labels_%d: Label tensor with shape batch_size x n_classes.

    Placeholders are wrapped in identity ops to avoid the error caused by
    feeding and fetching the same tensor.
    """
    placeholder_scope = TensorflowGraph.get_placeholder_scope(graph,
                                                              name_scopes)
    with graph.as_default():
      batch_size = self.batch_size
      n_classes = self.n_classes
      labels = []
      with placeholder_scope:
        for task in range(self.n_tasks):
          labels.append(
              tf.identity(
                  tf.placeholder(
                      tf.float32,
                      shape=[None, n_classes],
                      name='labels_%d' % task)))
      return labels

  def predict_on_batch(self, X):
    """Return model output for the provided input.

    Restore(checkpoint) must have previously been called on this object.

    Args:
      dataset: dc.data.dataset object.

    Returns:
      Tuple of three numpy arrays with shape n_examples x n_tasks (x ...):
        output: Model outputs.
        labels: True labels.
        weights: Example weights.
      Note that the output and labels arrays may be more than 2D, e.g. for
      classifier models that return class probabilities.

    Raises:
      AssertionError: If model is not in evaluation mode.
      ValueError: If output and labels are not both 3D or both 2D.
    """
    len_unpadded = len(X)
    if self.pad_batches:
      X = pad_features(self.batch_size, X)

    if not self._restored_model:
      self.restore()
    with self.eval_graph.graph.as_default():

      # run eval data through the model
      n_tasks = self.n_tasks
      output = []
      with self._get_shared_session(train=False).as_default():
        feed_dict = self.construct_feed_dict(X)
        data = self._get_shared_session(train=False).run(
            self.eval_graph.output, feed_dict=feed_dict)
        batch_output = np.asarray(data[:n_tasks], dtype=float)
        # reshape to batch_size x n_tasks x ...
        if batch_output.ndim == 3:
          batch_output = batch_output.transpose((1, 0, 2))
        elif batch_output.ndim == 2:
          batch_output = batch_output.transpose((1, 0))
        else:
          raise ValueError('Unrecognized rank combination for output: %s' %
                           (batch_output.shape,))
        output.append(batch_output)

        outputs = np.array(
            from_one_hot(np.squeeze(np.concatenate(output)), axis=-1))

    outputs = np.copy(outputs)
    outputs = np.reshape(outputs, (len(X), n_tasks))
    outputs = outputs[:len_unpadded]
    return outputs

  def predict_proba_on_batch(self, X):
    """Return model output for the provided input.

    Restore(checkpoint) must have previously been called on this object.

    Args:
      dataset: dc.data.Dataset object.

    Returns:
      Tuple of three numpy arrays with shape n_examples x n_tasks (x ...):
        output: Model outputs.
      Note that the output arrays may be more than 2D, e.g. for
      classifier models that return class probabilities.

    Raises:
      AssertionError: If model is not in evaluation mode.
      ValueError: If output and labels are not both 3D or both 2D.
    """
    if self.pad_batches:
      X = pad_features(self.batch_size, X)
    if not self._restored_model:
      self.restore()
    with self.eval_graph.graph.as_default():
      # run eval data through the model
      n_tasks = self.n_tasks
      with self._get_shared_session(train=False).as_default():
        feed_dict = self.construct_feed_dict(X)
        data = self._get_shared_session(train=False).run(
            self.eval_graph.output, feed_dict=feed_dict)
        batch_outputs = np.asarray(data[:n_tasks], dtype=float)
        # reshape to batch_size x n_tasks x ...
        if batch_outputs.ndim == 3:
          batch_outputs = batch_outputs.transpose((1, 0, 2))
        elif batch_outputs.ndim == 2:
          batch_outputs = batch_outputs.transpose((1, 0))
        else:
          raise ValueError('Unrecognized rank combination for output: %s ' %
                           (batch_outputs.shape,))

      # Note that softmax is already applied in construct_grpah
      outputs = batch_outputs

    return np.copy(outputs)


class TensorflowRegressor(TensorflowGraphModel):
  """Regression model.

  Subclasses must set the following attributes:
    output: Op(s) used for computing regression loss and predicted regression
      outputs for each task.
  """

  def get_task_type(self):
    return "regressor"

  def add_output_ops(self, graph, output):
    """No-op for regression models since no softmax."""
    return output

  def cost(self, output, labels, weights):
    """Calculate single-task training cost for a batch of examples.

    Args:
      output: Tensor with shape batch_size containing predicted values.
      labels: Tensor with shape batch_size containing true values.
      weights: Tensor with shape batch_size containing example weights.

    Returns:
      A tensor with shape batch_size containing the weighted cost for each
      example.
    """
    return tf.multiply(0.5 * tf.square(output - labels), weights)

  def add_label_placeholders(self, graph, name_scopes):
    """Add Placeholders for labels for each task.

    This method creates the following Placeholders for each task:
      labels_%d: Label tensor with shape batch_size.

    Placeholders are wrapped in identity ops to avoid the error caused by
    feeding and fetching the same tensor.
    """
    placeholder_scope = TensorflowGraph.get_placeholder_scope(graph,
                                                              name_scopes)
    with graph.as_default():
      batch_size = self.batch_size
      labels = []
      with placeholder_scope:
        for task in range(self.n_tasks):
          labels.append(
              tf.identity(
                  tf.placeholder(
                      tf.float32, shape=[None], name='labels_%d' % task)))
    return labels

  def predict_on_batch(self, X):
    """Return model output for the provided input.

    Restore(checkpoint) must have previously been called on this object.

    Args:
      dataset: dc.data.Dataset object.

    Returns:
      Tuple of three numpy arrays with shape n_examples x n_tasks (x ...):
        output: Model outputs.
        labels: True labels.
        weights: Example weights.
      Note that the output and labels arrays may be more than 2D, e.g. for
      classifier models that return class probabilities.

    Raises:
      AssertionError: If model is not in evaluation mode.
      ValueError: If output and labels are not both 3D or both 2D.
    """
    len_unpadded = len(X)
    if self.pad_batches:
      X = pad_features(self.batch_size, X)

    if not self._restored_model:
      self.restore()
    with self.eval_graph.graph.as_default():

      # run eval data through the model
      n_tasks = self.n_tasks
      outputs = []
      with self._get_shared_session(train=False).as_default():
        n_samples = len(X)
        feed_dict = self.construct_feed_dict(X)
        data = self._get_shared_session(train=False).run(
            self.eval_graph.output, feed_dict=feed_dict)
        batch_outputs = np.asarray(data[:n_tasks], dtype=float)
        # reshape to batch_size x n_tasks x ...
        if batch_outputs.ndim == 3:
          batch_outputs = batch_outputs.transpose((1, 0, 2))
        elif batch_outputs.ndim == 2:
          batch_outputs = batch_outputs.transpose((1, 0))
        # Handle edge case when batch-size is 1.
        elif batch_outputs.ndim == 1:
          n_samples = len(X)
          batch_outputs = batch_outputs.reshape((n_samples, n_tasks))
        else:
          raise ValueError('Unrecognized rank combination for output: %s' %
                           (batch_outputs.shape))
        # Prune away any padding that was added
        batch_outputs = batch_outputs[:n_samples]
        outputs.append(batch_outputs)

        outputs = np.squeeze(np.concatenate(outputs))

    outputs = np.copy(outputs)

    # Handle case of 0-dimensional scalar output
    if len(outputs.shape) > 0:
      return outputs[:len_unpadded]
    else:
      outputs = np.reshape(outputs, (1,))
      return outputs


class TensorflowMultiTaskRegressor(TensorflowRegressor):
  """Implements an icml model as configured in a model_config.proto."""

  def build(self, graph, name_scopes, training):
    """Constructs the graph architecture as specified in its config.

    This method creates the following Placeholders:
      mol_features: Molecule descriptor (e.g. fingerprint) tensor with shape
        batch_size x n_features.
    """
    n_features = self.n_features
    placeholder_scope = TensorflowGraph.get_placeholder_scope(graph,
                                                              name_scopes)
    with graph.as_default():
      with placeholder_scope:
        self.mol_features = tf.placeholder(
            tf.float32, shape=[None, n_features], name='mol_features')

      layer_sizes = self.layer_sizes
      weight_init_stddevs = self.weight_init_stddevs
      bias_init_consts = self.bias_init_consts
      dropouts = self.dropouts
      lengths_set = {
          len(layer_sizes),
          len(weight_init_stddevs),
          len(bias_init_consts),
          len(dropouts),
      }
      assert len(lengths_set) == 1, 'All layer params must have same length.'
      n_layers = lengths_set.pop()
      assert n_layers > 0, 'Must have some layers defined.'

      prev_layer = self.mol_features
      prev_layer_size = n_features
      for i in range(n_layers):
        layer = tf.nn.relu(
            model_ops.fully_connected_layer(
                tensor=prev_layer,
                size=layer_sizes[i],
                weight_init=tf.truncated_normal(
                    shape=[prev_layer_size, layer_sizes[i]],
                    stddev=weight_init_stddevs[i]),
                bias_init=tf.constant(
                    value=bias_init_consts[i], shape=[layer_sizes[i]])))
        layer = model_ops.dropout(layer, dropouts[i], training)
        prev_layer = layer
        prev_layer_size = layer_sizes[i]

      output = []
      for task in range(self.n_tasks):
        output.append(
            tf.squeeze(
                model_ops.fully_connected_layer(
                    tensor=prev_layer,
                    size=layer_sizes[i],
                    weight_init=tf.truncated_normal(
                        shape=[prev_layer_size, 1],
                        stddev=weight_init_stddevs[i]),
                    bias_init=tf.constant(value=bias_init_consts[i], shape=[1
                                                                           ]))))
      return output

  def construct_feed_dict(self, X_b, y_b=None, w_b=None, ids_b=None):
    """Construct a feed dictionary from minibatch data.

    TODO(rbharath): ids_b is not used here. Can we remove it?

    Args:
      X_b: np.ndarray of shape (batch_size, n_features)
      y_b: np.ndarray of shape (batch_size, n_tasks)
      w_b: np.ndarray of shape (batch_size, n_tasks)
      ids_b: List of length (batch_size) with datapoint identifiers.
    """
    orig_dict = {}
    orig_dict["mol_features"] = X_b
    for task in range(self.n_tasks):
      if y_b is not None:
        orig_dict["labels_%d" % task] = y_b[:, task]
      else:
        # Dummy placeholders
        orig_dict["labels_%d" % task] = np.squeeze(np.zeros((self.batch_size,)))
      if w_b is not None:
        orig_dict["weights_%d" % task] = w_b[:, task]
      else:
        # Dummy placeholders
        orig_dict["weights_%d" % task] = np.ones((self.batch_size,))
    return TensorflowGraph.get_feed_dict(orig_dict)
