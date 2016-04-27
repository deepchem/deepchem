#!/usr/bin/python
#
# Copyright 2015 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Helper operations and classes for general model building.

"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import collections
import cPickle as pickle
import os
import time
import warnings


import numpy as np
import pandas as pd
from sklearn import metrics as sklearn_metrics
import tensorflow as tf

from tensorflow.python.platform import logging

from tensorflow.python.platform import gfile

from deepchem.models import Model
import deepchem.metrics as met 
from deepchem.metrics import from_one_hot
from deepchem.models.tensorflow_models import model_ops
from deepchem.models.tensorflow_models import utils as tf_utils
from deepchem.utils.save import log

class TensorflowGraph(object):
  """Thin wrapper holding a tensorflow graph and a few vars.

  Has the following attributes:

    placeholder_root: String placeholder prefix, used to create
      placeholder_scope.

  Generic base class for defining, training, and evaluating TensorflowGraphs.

  Subclasses must implement the following methods:
    build
    add_output_ops
    add_training_cost 

  Subclasses must set the following attributes:
    loss: Op to calculate training cost used for gradient calculation.
    output: Op(s) for model output for each task.
    labels: Op(s) for true labels for each task.
    weights: Op(s) for example weights for each task.
    global_step: Scalar variable tracking total training/eval steps.
    updates: Op(s) for running updates of e.g. moving averages for batch
      normalization. Should be set to tf.no_op() if no updates are required.

  This base class provides the following attributes:
    model_params: dictionary containing model configuration parameters.
    graph: TensorFlow graph object.
    logdir: Path to the file output directory to store checkpoints etc.
    master: TensorFlow session master specification string.
    num_tasks: Integer number of tasks this model trains/evals on.
    placeholder_scope: name scope where tf.placeholders are defined.
    valid: Placeholder for a boolean tensor with shape batch_size to use as a
      mask when calculating gradient costs.

  Args:
    model_params: dictionary.
    train: If True, model is in training mode.
    logdir: Directory for output files.
  """

  def __init__(self, model_params, logdir, tasks, task_types, train=True,
               verbosity=None):
    """Constructs the computational graph.

    Args:
      train: whether model is in train mode
      model_params: dictionary of model parameters
      logdir: Location to save data

    This function constructs the computational graph for the model. It relies
    subclassed methods (build/cost) to construct specific graphs.
    """
    self.graph = tf.Graph() 
    self.model_params = model_params
    self.logdir = logdir
    self.tasks = tasks
    self.task_types = task_types
    self.num_tasks = len(task_types)
    self.verbosity = verbosity

    # Lazily created by _get_shared_session().
    self._shared_session = None

    # Guard variable to make sure we don't Restore() this model
    # from a disk checkpoint more than once.
    self._restored_model = False

    # Cache of TensorFlow scopes, to prevent '_1' appended scope names
    # when subclass-overridden methods use the same scopes.
    self._name_scopes = {}

    # Path to save checkpoint files, which matches the
    # replicated supervisor's default path.
    self._save_path = os.path.join(logdir, 'model.ckpt')

    with self.graph.as_default():
      model_ops.set_training(train)
      self.placeholder_root = 'placeholders'
      with tf.name_scope(self.placeholder_root) as scope:
        self.placeholder_scope = scope
        self.valid = tf.placeholder(tf.bool,
                                    shape=[model_params["batch_size"]],
                                    name='valid')

    self.setup()
    if train:
      self.add_training_cost()
      self.merge_updates()
    else:
      self.add_output_ops()  # add softmax heads

  def setup(self):
    """Add ops common to training/eval to the graph."""
    with self.graph.as_default():
      with tf.name_scope('core_model'):
        self.build()
      self.add_labels_and_weights()
      self.global_step = tf.Variable(0, name='global_step', trainable=False)

  def _shared_name_scope(self, name):
    """Returns a singleton TensorFlow scope with the given name.

    Used to prevent '_1'-appended scopes when sharing scopes with child classes.

    Args:
      name: String. Name scope for group of operations.
    Returns:
      tf.name_scope with the provided name.
    """
    if name not in self._name_scopes:
      with self.graph.as_default():
        with tf.name_scope(name) as scope:
          self._name_scopes[name] = scope

    return tf.name_scope(self._name_scopes[name])

  def add_training_cost(self):
    with self.graph.as_default():
      self.require_attributes(['output', 'labels', 'weights'])
      epsilon = 1e-3  # small float to avoid dividing by zero
      model_params = self.model_params
      weighted_costs = []  # weighted costs for each example
      gradient_costs = []  # costs used for gradient calculation
      old_costs = []  # old-style cost

      with self._shared_name_scope('costs'):
        for task in xrange(self.num_tasks):
          task_str = str(task).zfill(len(str(self.num_tasks)))
          with self._shared_name_scope('cost_{}'.format(task_str)):
            with tf.name_scope('weighted'):
              weighted_cost = self.cost(self.output[task], self.labels[task],
                                        self.weights[task])
              weighted_costs.append(weighted_cost)

            with tf.name_scope('gradient'):
              # Note that we divide by the batch size and not the number of
              # non-zero weight examples in the batch.  Also, instead of using
              # tf.reduce_mean (which can put ops on the CPU) we explicitly
              # calculate with div/sum so it stays on the GPU.
              gradient_cost = tf.div(tf.reduce_sum(weighted_cost),
                                     model_params["batch_size"])
              gradient_costs.append(gradient_cost)

            with tf.name_scope('old_cost'):
              old_cost = tf.div(
                  tf.reduce_sum(weighted_cost),
                  tf.reduce_sum(self.weights[task]) + epsilon)
              old_costs.append(old_cost)

        # aggregated costs
        with self._shared_name_scope('aggregated'):
          with tf.name_scope('gradient'):
            loss = tf.add_n(gradient_costs)
          with tf.name_scope('old_cost'):
            old_loss = tf.add_n(old_costs)

          # weight decay
          if model_params["penalty"] != 0.0:
            penalty = model_ops.WeightDecay(model_params)
            loss += penalty
            old_loss += penalty

        # loss used for gradient calculation
        self.loss = loss

      return weighted_costs

  def merge_updates(self):
    """Group updates into a single op."""
    with self.graph.as_default():
      updates = tf.get_default_graph().get_collection('updates')
      if updates:
        self.updates = tf.group(*updates, name='updates')
      else:
        self.updates = tf.no_op(name='updates')

  def fit(self,
          dataset,
          summaries=False,
          max_checkpoints_to_keep=5):
    """Fit the model.

    Args:
      dataset: Dataset object that represents data on disk.
      summaries: If True, add summaries for model parameters.
      max_checkpoints_to_keep: Integer. Maximum number of checkpoints to keep;
        older checkpoints will be deleted.

    Raises:
      AssertionError: If model is not in training mode.
    """
    num_datapoints = len(dataset)
    batch_size = self.model_params["batch_size"]
    step_per_epoch = np.ceil(float(num_datapoints)/batch_size)
    nb_epoch = self.model_params["nb_epoch"]
    log("Training for %d epochs" % nb_epoch, self.verbosity)
    with self.graph.as_default():
      assert model_ops.is_training()
      self.require_attributes(['loss', 'global_step', 'updates'])
      train_op = self.get_training_op()
      no_op = tf.no_op()
      tf.train.write_graph(
          tf.get_default_graph().as_graph_def(), self.logdir, 'train.pbtxt')
      with self._get_shared_session() as sess:
        sess.run(tf.initialize_all_variables())
        saver = tf.train.Saver(max_to_keep=max_checkpoints_to_keep)
        # Save an initial checkpoint.
        saver.save(sess, self._save_path, global_step=self.global_step)
        for epoch in range(nb_epoch):
          for (X_b, y_b, w_b, ids_b) in dataset.iterbatches(batch_size):
            # Run training op and compute summaries.
            feed_dict = self.construct_feed_dict(X_b, y_b, w_b, ids_b)
            step, loss, _ = sess.run(
                [train_op.values()[0], self.loss, self.updates],
                feed_dict=feed_dict)
          # Save model checkpoints at end of epoch
          saver.save(sess, self._save_path, global_step=self.global_step)
          log('Ending epoch %d: loss %g' % (epoch, loss), self.verbosity)
        # Always save a final checkpoint when complete.
        saver.save(sess, self._save_path, global_step=self.global_step)

  def predict_on_batch(self, X):
    """Return model output for the provided input.

    Restore(checkpoint) must have previously been called on this object.

    Args:
      dataset: deepchem.datasets.dataset object.

    Returns:
      Tuple of three numpy arrays with shape num_examples x num_tasks (x ...):
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
    with self.graph.as_default():
      assert not model_ops.is_training()
      self.require_attributes(['output', 'labels', 'weights'])

      # run eval data through the model
      num_tasks = self.num_tasks
      output, labels, weights = [], [], []
      start = time.time()
      with self._get_shared_session().as_default():
        batch_count = -1.0

        feed_dict = self.construct_feed_dict(X)
        batch_start = time.time()
        batch_count += 1
        data = self._get_shared_session().run(
            self.output + self.labels + self.weights,
            feed_dict=feed_dict)
        batch_output = np.asarray(data[:num_tasks], dtype=float)
        batch_labels = np.asarray(data[num_tasks:num_tasks * 2], dtype=float)
        batch_weights = np.asarray(data[num_tasks * 2:num_tasks * 3],
                                   dtype=float)
        # reshape to batch_size x num_tasks x ...
        if batch_output.ndim == 3 and batch_labels.ndim == 3:
          batch_output = batch_output.transpose((1, 0, 2))
          batch_labels = batch_labels.transpose((1, 0, 2))
        elif batch_output.ndim == 2 and batch_labels.ndim == 2:
          batch_output = batch_output.transpose((1, 0))
          batch_labels = batch_labels.transpose((1, 0))
        else:
          raise ValueError(
              'Unrecognized rank combination for output and labels: %s %s' %
              (batch_output.shape, batch_labels.shape))
        batch_weights = batch_weights.transpose((1, 0))
        valid = feed_dict[self.valid.name]
        # only take valid outputs
        if np.count_nonzero(~valid):
          batch_output = batch_output[valid]
          batch_labels = batch_labels[valid]
          batch_weights = batch_weights[valid]
        output.append(batch_output)
        labels.append(batch_labels)
        weights.append(batch_weights)

        logging.info('Eval batch took %g seconds', time.time() - start)

        #labels = np.array(from_one_hot(
        #    np.squeeze(np.concatenate(labels)), axis=-1))
        labels = np.squeeze(np.concatenate(labels)) 
        labels = np.array(labels)[:, 1]

    return np.copy(labels)

  def add_output_ops(self):
    """Replace logits with softmax outputs."""
    with self.graph.as_default():
      softmax = []
      with tf.name_scope('inference'):
        for i, logits in enumerate(self.output):
          softmax.append(tf.nn.softmax(logits, name='softmax_%d' % i))
      self.output = softmax

  def build(self):
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


  def add_label_placeholders(self):
    """Add Placeholders for labels for each task.

    This method creates the following Placeholders for each task:
      labels_%d: Float label tensor. For classification tasks, this tensor will
        have shape batch_size x num_classes. For regression tasks, this tensor
        will have shape batch_size.

    Raises:
      NotImplementedError: if not overridden by concrete subclass.
    """
    raise NotImplementedError('Must be overridden by concrete subclass')

  def add_weight_placeholders(self):
    """Add Placeholders for example weights for each task.

    This method creates the following Placeholders for each task:
      weights_%d: Label tensor with shape batch_size.

    Placeholders are wrapped in identity ops to avoid the error caused by
    feeding and fetching the same tensor.
    """
    weights = []
    for task in xrange(self.num_tasks):
      with tf.name_scope(self.placeholder_scope):
        weights.append(tf.identity(
            tf.placeholder(tf.float32, shape=[self.model_params["batch_size"]],
                           name='weights_%d' % task)))
    self.weights = weights

  def add_labels_and_weights(self):
    """Add Placeholders for labels and weights.

    This method results in the creation of the following Placeholders for each
    task:
      labels_%d: Float label tensor. For classification tasks, this tensor will
        have shape batch_size x num_classes. For regression tasks, this tensor
        will have shape batch_size.
      weights_%d: Label tensor with shape batch_size.

    This method calls self.add_label_placeholders and self.add_weight_placeholders; the
    former method must be implemented by a concrete subclass.
    """
    self.add_label_placeholders()
    self.add_weight_placeholders()

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
    # TODO(user): for mixed classification/regression models, pass in a task
    # index to control the cost calculation
    raise NotImplementedError('Must be overridden by concrete subclass')

  def get_training_op(self):
    """Get training op for applying gradients to variables.

    Subclasses that need to do anything fancy with gradients should override
    this method.

    Returns:
    A training op.
    """
    opt = model_ops.Optimizer(self.model_params)
    return opt.minimize(self.loss, global_step=self.global_step, name='train')

  def add_output_ops(self):
    """Add ops for inference.

    Default implementation is pass, derived classes can override as needed.
    """
    pass

  def _get_shared_session(self):
    if not self._shared_session:
      # allow_soft_placement=True allows ops without a GPU implementation
      # to run on the CPU instead.
      config = tf.ConfigProto(allow_soft_placement=True)
      self._shared_session = tf.Session(config=config)
    return self._shared_session

  def close_shared_session(self):
    if self._shared_session:
      self._shared_session.close()

  def _get_feed_dict(self, named_values):
    feed_dict = {}
    for name, value in named_values.iteritems():
      feed_dict['{}/{}:0'.format(self.placeholder_root, name)] = value
    return feed_dict

  def restore(self):
    """Restores the model from the provided training checkpoint.

    Args:
      checkpoint: string. Path to checkpoint file.
    """
    if self._restored_model:
      return
    with self.graph.as_default():
      assert not model_ops.is_training()
      last_checkpoint = self._find_last_checkpoint()

      saver = tf.train.Saver()
      saver.restore(self._get_shared_session(),
                    tf_utils.ParseCheckpoint(last_checkpoint))
      self.global_step_number = int(self._get_shared_session().run(self.global_step))
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
          
  def require_attributes(self, attrs):
    """Require class attributes to be defined.

    Args:
      attrs: A list of attribute names that must be defined.

    Raises:
      AssertionError: if a required attribute is not defined.
    """
    for attr in attrs:
      if getattr(self, attr, None) is None:
        raise AssertionError(
            'self.%s must be defined by a concrete subclass' % attr)

class TensorflowClassifier(TensorflowGraph):
  """Classification model.

  Subclasses must set the following attributes:
    output: Logits op(s) used for computing classification loss and predicted
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
      logits: Tensor with shape batch_size x num_classes containing logits.
      labels: Tensor with shape batch_size x num_classes containing true labels
        in a one-hot encoding.
      weights: Tensor with shape batch_size containing example weights.

    Returns:
      A tensor with shape batch_size containing the weighted cost for each
      example.
    """
    return tf.mul(tf.nn.softmax_cross_entropy_with_logits(logits, labels),
                  weights)

  def add_training_cost(self):
    """Calculate additional classifier-specific costs.

    Returns:
      A list of tensors with shape batch_size containing costs for each task.
    """
    with self.graph.as_default():
      # calculate losses
      weighted_costs = super(TensorflowClassifier, self).add_training_cost()
      epsilon = 1e-3  # small float to avoid dividing by zero
      model_params = self.model_params
      num_tasks = model_params["num_classification_tasks"]
      cond_costs = collections.defaultdict(list)

      with self._shared_name_scope('costs'):
        for task in xrange(num_tasks):
          task_str = str(task).zfill(len(str(num_tasks)))
          with self._shared_name_scope('cost_{}'.format(task_str)):
            with tf.name_scope('conditional'):
              # pos/neg costs: mean over pos/neg examples
              for name, label in [('neg', 0), ('pos', 1)]:
                cond_weights = self.labels[task][:model_params["batch_size"], label]
                cond_cost = tf.div(
                    tf.reduce_sum(tf.mul(weighted_costs[task], cond_weights)),
                    tf.reduce_sum(cond_weights) + epsilon)
                cond_costs[name].append(cond_cost)

        # aggregated costs
        with self._shared_name_scope('aggregated'):
          with tf.name_scope('pos_cost'):
            pos_cost = tf.add_n(cond_costs['pos'])
          with tf.name_scope('neg_cost'):
            neg_cost = tf.add_n(cond_costs['neg'])

      # keep track of the number of positive examples seen by each task
      with tf.name_scope('counts'):
        for task in xrange(num_tasks):
          num_pos = tf.Variable(0.0, name='num_pos_%d' % task, trainable=False)
          # the assignment must occur on the same device as the variable
          with tf.device(num_pos.device):
            tf.get_default_graph().add_to_collection(
                'updates', num_pos.assign_add(
                    tf.reduce_sum(self.labels[task][:model_params["batch_size"], 1])))

      return weighted_costs

  def example_counts(self, y_true):
    """Get counts of examples in each class.

    Args:
      y_true: List of numpy arrays containing true values, one for each task.

    Returns:
      A dict mapping class names to counts.
    """
    classes = np.unique(np.concatenate(y_true))
    counts = {klass: np.zeros(self.num_tasks, dtype=int)
              for klass in classes}
    for task in xrange(self.num_tasks):
      for klass in classes:
        counts[klass][task] = np.count_nonzero(y_true[task] == klass)
    return counts

  def add_label_placeholders(self):
    """Add Placeholders for labels for each task.

    This method creates the following Placeholders for each task:
      labels_%d: Label tensor with shape batch_size x num_classes.

    Placeholders are wrapped in identity ops to avoid the error caused by
    feeding and fetching the same tensor.
    """
    with self.graph.as_default():
      model_params = self.model_params
      batch_size = model_params["batch_size"]
      num_classes = model_params["num_classes"]
      labels = []
      for task in xrange(self.num_tasks):
        with tf.name_scope(self.placeholder_scope):
          labels.append(tf.identity(
              tf.placeholder(tf.float32, shape=[batch_size, num_classes],
                             name='labels_%d' % task)))
      self.labels = labels


class TensorflowRegressor(TensorflowGraph):
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

  def example_counts(self, y_true):
    """Get counts of examples in each class.

    Args:
      y_true: List of numpy arrays containing true values, one for each task.

    Returns:
      A dict mapping class names to counts.
    """
    return {'all': np.asarray([len(y_true[task])
                               for task in xrange(self.num_tasks)])}

  def add_label_placeholders(self):
    """Add Placeholders for labels for each task.

    This method creates the following Placeholders for each task:
      labels_%d: Label tensor with shape batch_size.

    Placeholders are wrapped in identity ops to avoid the error caused by
    feeding and fetching the same tensor.
    """
    with self.graph.as_default():
      batch_size = self.model_params["batch_size"]
      labels = []
      for task in xrange(self.num_tasks):
        with tf.name_scope(self.placeholder_scope):
          labels.append(tf.identity(
              tf.placeholder(tf.float32, shape=[batch_size],
                             name='labels_%d' % task)))
      self.labels = labels

class TensorflowModel(Model):
  """
  Abstract base class shared across all Tensorflow models.
  """

  def __init__(self,
               tasks,
               task_types,
               model_params,
               logdir,
               tf_class=None,
               verbosity=None):
    """
    Args:
      tf_class: Class that inherits from TensorflowGraph
    """ 
    
    assert verbosity in [None, "low", "high"]
    self.verbosity = verbosity
    if tf_class is None:
      tf_class = TensorflowGraph
    self.model_params = model_params
    self.tasks = tasks
    self.task_types = task_types
    self.train_model = tf_class(model_params, logdir, tasks, task_types,
                                train=True, verbosity=verbosity)
    self.eval_model = tf_class(model_params, logdir, tasks, task_types,
                                train=False, verbosity=verbosity)
    self.num_tasks = len(self.task_types)
    self.fit_transformers = None

  def fit(self, dataset):
    """
    Fits TensorflowGraph to data.
    """
    self.train_model.fit(dataset)

  def predict_on_batch(self, X):
    """
    Makes predictions on batch of data.
    """
    return self.eval_model.predict_on_batch(X)

  def save(self):
    """
    No-op since tf models save themselves during fit()
    """
    pass

  def load(self, model_dir):
    """
    Loads model from disk. Thin wrapper around restore() for consistency.
    """
    if model_dir != self.eval_model.logdir:
      raise ValueError("Cannot load from directory that is not logdir.")
    self.eval_model.restore()
