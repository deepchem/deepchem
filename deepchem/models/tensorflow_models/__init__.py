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
from deepchem.models import Model
from deepchem.metrics import from_one_hot
from deepchem.models.tensorflow_models import model_ops
from deepchem.models.tensorflow_models import utils as tf_utils
from deepchem.utils.save import log
from deepchem.datasets import pad_features
from tensorflow.contrib.layers.python.layers import batch_norm

def softmax(x):
  """Simple numpy softmax implementation
  """
  # (n_samples, n_classes)
  if len(x.shape) == 2:
    row_max = np.max(x, axis = 1)
    x -= row_max.reshape((x.shape[0], 1))
    x = np.exp(x)
    row_sum = np.sum(x, axis = 1)
    x /= row_sum.reshape((x.shape[0], 1))
  # (n_samples, n_tasks, n_classes)
  elif len(x.shape) == 3:
    row_max = np.max(x, axis = 2)
    x -= row_max.reshape(x.shape[:2] + (1,))
    x = np.exp(x)
    row_sum = np.sum(x, axis = 2)
    x /= row_sum.reshape(x.shape[:2] + (1,))
  return x

class TensorflowGraph(object):
  """Simple class that holds information needed to run Tensorflow graph."""
  def __init__(self, graph, session, name_scopes, output, labels, weights, loss):
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
    #with graph.as_default():
    #  with tf.name_scope(placeholder_root) as scope:
    #    return scope
    return TensorflowGraph.shared_name_scope(placeholder_root, graph, name_scopes)

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


class TensorflowGraphModel(object):
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

  def __init__(self, n_tasks, n_features, logdir, layer_sizes=[1000],
               weight_init_stddevs=[.02], bias_init_consts=[1.], penalty=0.0,
               dropouts=[0.5], learning_rate=.001, momentum=".9",
               optimizer="adam", batch_size=50, n_classes=2,
               train=True, verbosity=None, **kwargs):
    """Constructs the computational graph.

    Args:
      train: whether model is in train mode
      logdir: Location to save data

    This function constructs the computational graph for the model. It relies
    subclassed methods (build/cost) to construct specific graphs.
    """
    # Save hyperparameters
    self.n_tasks = n_tasks
    self.n_features = n_features
    self.logdir = logdir
    self.layer_sizes = layer_sizes
    self.weight_init_stddevs = weight_init_stddevs
    self.bias_init_consts = bias_init_consts
    self.penalty = penalty
    self.dropouts = dropouts
    self.learning_rate = learning_rate
    self.momentum = momentum
    self.optimizer = optimizer
    self.batch_size = batch_size
    self.n_classes = n_classes
    self.train = train
    self.verbosity = verbosity

    # Guard variable to make sure we don't Restore() this model
    # from a disk checkpoint more than once.
    self._restored_model = False
    # Path to save checkpoint files, which matches the
    # replicated supervisor's default path.
    self._save_path = os.path.join(logdir, 'model.ckpt')

    self.train_graph = self.construct_graph(training=True)
    self.eval_graph = self.construct_graph(training=False)


  def construct_graph(self, training):
    """Returns a TensorflowGraph object."""
    graph = tf.Graph() 

    # Lazily created by _get_shared_session().
    shared_session = None

    # Cache of TensorFlow scopes, to prevent '_1' appended scope names
    # when subclass-overridden methods use the same scopes.
    name_scopes = {}

    # Setup graph
    with graph.as_default():
      output = self.build(graph, name_scopes, training)
      labels = self.add_label_placeholders(graph, name_scopes)
      weights = self.add_example_weight_placeholders(graph, name_scopes)

    if training:
      loss = self.add_training_cost(graph, name_scopes, output, labels, weights)
    else:
      loss = None
      output = self.add_output_ops(graph, output)  # add softmax heads
    return TensorflowGraph(graph=graph,
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
          with TensorflowGraph.shared_name_scope(
              'cost_{}'.format(task_str), graph, name_scopes):
            with tf.name_scope('weighted'):
              weighted_cost = self.cost(output[task], labels[task],
                                        weights[task])
              weighted_costs.append(weighted_cost)

            with tf.name_scope('gradient'):
              # Note that we divide by the batch size and not the number of
              # non-zero weight examples in the batch.  Also, instead of using
              # tf.reduce_mean (which can put ops on the CPU) we explicitly
              # calculate with div/sum so it stays on the GPU.
              gradient_cost = tf.div(tf.reduce_sum(weighted_cost),
                                     self.batch_size)
              gradient_costs.append(gradient_cost)

        # aggregated costs
        with TensorflowGraph.shared_name_scope('aggregated', graph, name_scopes):
          with tf.name_scope('gradient'):
            loss = tf.add_n(gradient_costs)

          # weight decay
          if self.penalty != 0.0:
            penalty = model_ops.weight_decay(self.penalty_type, self.penalty)
            loss += penalty

      return loss 

  def fit(self, dataset, nb_epoch=10, pad_batches=False, shuffle=False,
          max_checkpoints_to_keep=5, log_every_N_batches=50, **kwargs):
    """Fit the model.

    Args:
      dataset: Dataset object that represents data on disk.
      max_checkpoints_to_keep: Integer. Maximum number of checkpoints to keep;
        older checkpoints will be deleted.

    Raises:
      AssertionError: If model is not in training mode.
    """
    ############################################################## TIMING
    time1 = time.time()
    ############################################################## TIMING
    n_datapoints = len(dataset)
    batch_size = self.batch_size
    step_per_epoch = np.ceil(float(n_datapoints)/batch_size)
    log("Training for %d epochs" % nb_epoch, self.verbosity)
    with self.train_graph.graph.as_default():
      train_op = self.get_training_op(
          self.train_graph.graph, self.train_graph.loss)
      with self._get_shared_session(train=True) as sess:
        sess.run(tf.initialize_all_variables())
        saver = tf.train.Saver(max_to_keep=max_checkpoints_to_keep)
        # Save an initial checkpoint.
        saver.save(sess, self._save_path, global_step=0)
        for epoch in range(nb_epoch):
          avg_loss, n_batches = 0., 0
          if shuffle:
            log("About to shuffle dataset before epoch start.", self.verbosity)
            dataset.shuffle()
          for ind, (X_b, y_b, w_b, ids_b) in enumerate(
              dataset.iterbatches(batch_size, pad_batches=True)): # hardcode pad_batches=True to work around limitations in Tensorflow
            if ind % log_every_N_batches == 0:
              log("On batch %d" % ind, self.verbosity)
            # Run training op.
            feed_dict = self.construct_feed_dict(X_b, y_b, w_b, ids_b)
            fetches = self.train_graph.output + [
                train_op, self.train_graph.loss]
            fetched_values = sess.run(
                fetches,
                feed_dict=feed_dict)
            output = fetched_values[:len(self.train_graph.output)]
            loss = fetched_values[-1]
            avg_loss += loss
            y_pred = np.squeeze(np.array(output))
            y_b = y_b.flatten()
            n_batches += 1
          saver.save(sess, self._save_path, global_step=epoch)
          avg_loss = float(avg_loss)/n_batches
          log('Ending epoch %d: Average loss %g' % (epoch, avg_loss), self.verbosity)
        # Always save a final checkpoint when complete.
        saver.save(sess, self._save_path, global_step=epoch+1)
    ############################################################## TIMING
    time2 = time.time()
    print("TIMING: model fitting took %0.3f s" % (time2-time1),
          self.verbosity)
    ############################################################## TIMING

  def predict_on_batch(self, X, pad_batch=False):
    """Return model output for the provided input.

    Restore(checkpoint) must have previously been called on this object.

    Args:
      dataset: deepchem.datasets.dataset object.

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
    if pad_batch:
      X = pad_features(self.batch_size, X)
    
    if not self._restored_model:
      self.restore()
    with self.eval_graph.graph.as_default():

      # run eval data through the model
      n_tasks = self.n_tasks
      output = []
      start = time.time()
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
          raise ValueError(
              'Unrecognized rank combination for output: %s' %
              (batch_output.shape,))
        output.append(batch_output)

        outputs = np.array(from_one_hot(
            np.squeeze(np.concatenate(output)), axis=-1))

    return np.copy(outputs)

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
    placeholder_scope = TensorflowGraph.get_placeholder_scope(graph, name_scopes)
    with placeholder_scope:
      for task in range(self.n_tasks):
        weights.append(tf.identity(
            tf.placeholder(tf.float32, shape=[None],
                           name='weights_%d' % task)))
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
      opt = model_ops.optimizer(self.optimizer, self.learning_rate, self.momentum)
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

      # TODO(rbharath): Is setting train=Falseright here?
      saver = tf.train.Saver()
      saver.restore(self._get_shared_session(train=False),
                    last_checkpoint)
      self._restored_model = True

  def _find_last_checkpoint(self):
    """Finds last saved checkpoint."""
    highest_num, last_checkpoint = -np.inf, None
    for filename in os.listdir(self.logdir):
      # checkpoints look like logdir/model.ckpt-N
      # self._save_path is "logdir/model.ckpt"
      if os.path.basename(self._save_path) in filename:
        try:
          N = int(filename.split("-")[-1])
          if N > highest_num:
            highest_num = N
            last_checkpoint = filename
        except ValueError:
          pass
    return os.path.join(self.logdir, last_checkpoint)
          
class TensorflowClassifier(TensorflowGraphModel):
  """Classification model.

  Subclasses must set the following attributes:
    output: logits op(s) used for computing classification loss and predicted
      class probabilities for each task.

  Class attributes:
    default_metrics: List of metrics to compute by default.
  """

  default_metrics = ['auc']

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
    return tf.mul(tf.nn.softmax_cross_entropy_with_logits(logits, labels),
                  weights)

  def add_label_placeholders(self, graph, name_scopes):
    """Add Placeholders for labels for each task.

    This method creates the following Placeholders for each task:
      labels_%d: Label tensor with shape batch_size x n_classes.

    Placeholders are wrapped in identity ops to avoid the error caused by
    feeding and fetching the same tensor.
    """
    placeholder_scope = TensorflowGraph.get_placeholder_scope(graph, name_scopes)
    with graph.as_default():
      batch_size = self.batch_size 
      n_classes = self.n_classes
      labels = []
      with placeholder_scope:
        for task in range(self.n_tasks):
          labels.append(tf.identity(
              tf.placeholder(tf.float32, shape=[None, n_classes],
                             name='labels_%d' % task)))
      return labels

  def predict_proba_on_batch(self, X, pad_batch=False):
    """Return model output for the provided input.

    Restore(checkpoint) must have previously been called on this object.

    Args:
      dataset: deepchem.datasets.dataset object.

    Returns:
      Tuple of three numpy arrays with shape n_examples x n_tasks (x ...):
        output: Model outputs.
      Note that the output arrays may be more than 2D, e.g. for
      classifier models that return class probabilities.

    Raises:
      AssertionError: If model is not in evaluation mode.
      ValueError: If output and labels are not both 3D or both 2D.
    """
    if pad_batch:
      X = pad_features(self.batch_size, X)
    if not self._restored_model:
      self.restore()
    with self.eval_graph.graph.as_default():
      # run eval data through the model
      n_tasks = self.n_tasks
      outputs = []
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
          raise ValueError(
              'Unrecognized rank combination for output: %s ' %
              (batch_outputs.shape,))
        outputs.append(batch_outputs)

        # TODO(rbharath): This is a bug! We're actually applying softmax twice.
        # I believe this is harmless since softmax of softmax doesn't change
        # properties, but I need to check this...
        # We apply softmax to predictions to get class probabilities.
        outputs = softmax(np.squeeze(np.hstack(outputs)))

    return np.copy(outputs)

class TensorflowRegressor(TensorflowGraphModel):
  """Regression model.

  Subclasses must set the following attributes:
    output: Op(s) used for computing regression loss and predicted regression
      outputs for each task.

  Class attributes:
    default_metrics: List of metrics to compute by default.
  """

  default_metrics = ['r2']

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
    return tf.mul(0.5 * tf.square(output - labels), weights)

  def add_label_placeholders(self, graph, name_scopes):
    """Add Placeholders for labels for each task.

    This method creates the following Placeholders for each task:
      labels_%d: Label tensor with shape batch_size.

    Placeholders are wrapped in identity ops to avoid the error caused by
    feeding and fetching the same tensor.
    """
    placeholder_scope = TensorflowGraph.get_placeholder_scope(graph, name_scopes)
    with graph.as_default():
      batch_size = self.batch_size
      labels = []
      with placeholder_scope:
        for task in range(self.n_tasks):
          labels.append(tf.identity(
              tf.placeholder(tf.float32, shape=[None],
                             name='labels_%d' % task)))
    return labels

  def predict_on_batch(self, X):
    """Return model output for the provided input.

    Restore(checkpoint) must have previously been called on this object.

    Args:
      dataset: deepchem.datasets.dataset object.

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
    if not self._restored_model:
      self.restore()
    with self.train_graph.graph.as_default():

      # run eval data through the model
      n_tasks = self.n_tasks
      outputs = []
      with self._get_shared_session(train=False).as_default():
        n_samples = len(X)
        # TODO(rbharath): Should this be padding there? Shouldn't padding be
        # turned on in predict?
        #################################################################### DEBUG
        # Some tensorflow models can't handle variadic batches,
        # especially models using tf.pack, tf.split. Pad batch-size
        # to handle these cases.
        #X = pad_features(self.batch_size, X)
        #################################################################### DEBUG
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
          raise ValueError(
              'Unrecognized rank combination for output: %s' %
              (batch_outputs.shape))
        # Prune away any padding that was added
        batch_outputs = batch_outputs[:n_samples]
        outputs.append(batch_outputs)

        outputs = np.squeeze(np.concatenate(outputs)) 

    return np.copy(outputs)

class TensorflowModel(Model):
  """
  Abstract base class shared across all Tensorflow models.
  """

  def __init__(self, model, logdir, verbosity=None, **kwargs):
    assert verbosity in [None, "low", "high"]
    self.verbosity = verbosity
    self.model_instance = model
    self.fit_transformers = None
    if not os.path.exists(logdir):
      os.makedirs(logdir)

  def fit(self, dataset, **kwargs):
    """
    Fits TensorflowGraph to data.
    """
    self.model_instance.fit(dataset, **kwargs)

  def predict(self, dataset, transformers=[], batch_size=None,
              pad_batches=False):
    """
    Uses self to make predictions on provided Dataset object.

    This is overridden to make sure the batch size is always valid for Tensorflow.

    Returns:
      y_pred: numpy ndarray of shape (n_samples,)
    """
    return Model.predict(self, dataset, transformers, self.model_instance.batch_size, True)

  def predict_on_batch(self, X, pad_batch=True):
    """
    Makes predictions on batch of data.
    """
    if pad_batch:
      len_unpadded = len(X)
      Xpad = pad_features(self.model_instance.batch_size, X)
      return self.model_instance.predict_on_batch(Xpad)[:len_unpadded]
    else:
      return self.model_instance.predict_on_batch(X)

  def predict_grad_on_batch(self, X):
    """
    Calculates gradient of cost function on batch of data.
    """
    return self.model_instance.predict_grad_on_batch(X)

  def predict_proba_on_batch(self, X, pad_batch=False):
    """
    Makes predictions on batch of data.
    """
    return self.model_instance.predict_proba_on_batch(X, pad_batch=pad_batch)

  def save(self):
    """
    No-op since tf models save themselves during fit()
    """
    pass

  def reload(self):
    """
    Loads model from disk. Thin wrapper around restore() for consistency.
    """
    self.model_instance.restore()

  def get_num_tasks(self):
    return self.model_instance.n_tasks
