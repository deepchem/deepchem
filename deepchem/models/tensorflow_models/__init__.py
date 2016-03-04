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

These methods are generally dependent on ModelConfig.
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
from deepchem.utils import metrics
from deepchem.models.tensorflow_models import model_ops
from deepchem.models.tensorflow_models import utils as tf_utils


class TensorflowModel(Model):
  """Abstract base class shared across all Tensorflow models.

  Generic base class for defining, training, and evaluating models.

  Subclasses must implement the following methods:
    add_output_ops
    build
    Eval
    training_cost 

  Subclasses must set the following attributes:
    loss: Op to calculate training cost used for gradient calculation.
    output: Op(s) for model output for each task.
    labels: Op(s) for true labels for each task.
    weights: Op(s) for example weights for each task.
    global_step: Scalar variable tracking total training/eval steps.
    updates: Op(s) for running updates of e.g. moving averages for batch
      normalization. Should be set to tf.no_op() if no updates are required.

  This base class provides the following attributes:
    model_params: ModelConfig containing model configuration parameters.
    graph: TensorFlow graph object.
    logdir: Path to the file output directory to store checkpoints etc.
    master: TensorFlow session master specification string.
    num_tasks: Integer number of tasks this model trains/evals on.
    placeholder_root: String placeholder prefix, used to create
      placeholder_scope.
    placeholder_scope: name scope where tf.placeholders are defined.
    summary_writer: SummaryWriter for writing summaries.
    valid: Placeholder for a boolean tensor with shape batch_size to use as a
      mask when calculating gradient costs.

  Args:
    model_params: ModelConfig.
    train: If True, model is in training mode.
    logdir: Directory for output files.
    graph: Default graph.
    summary_writer: SummaryWriter to use for writing summaries. If None, a new
      SummaryWriter will be created.
  """

  # TODO(rbharath): config and model_params overlap significantly. Maybe just
  # get rid of config? Protos are much better for
  # documentation than dictionaries. They're also more overhead though.
  def __init__(self,
               task_types,
               model_params,
               train=False,
               logdir=None,
               graph=None,
               summary_writer=None):
    self.model_params = model_params 
    self.graph = graph if graph is not None else tf.Graph()
    self.logdir = logdir
    
    # Path to save checkpoint files, which matches the
    # replicated supervisor's default path.
    self._save_path = os.path.join(logdir, 'model.ckpt')

    # batches. Lazily created by _get_shared_session().
    self._shared_session = None

    # Guard variable to make sure we don't Restore() this model
    # from a disk checkpoint more than once.
    self._restored_model = False

    # Cache of TensorFlow scopes, to prevent '_1' appended scope names
    # when subclass-overridden methods use the same scopes.
    self._name_scopes = {}

    with self.graph.as_default():
      model_ops.set_training(train)
      self.placeholder_root = 'placeholders'
      with tf.name_scope(self.placeholder_root) as scope:
        self.placeholder_scope = scope
        self.valid = tf.placeholder(tf.bool,
                                    shape=[model_params["batch_size"]],
                                    name='valid')
      print("TensorflowModel.__init__")
      print("self.valid")
      print(self.valid)

    if "num_classification_tasks" in model_params:
      num_classification_tasks = model_params["num_classification_tasks"]
    else:
      num_classification_tasks = 0
    if "num_regression_tasks" in model_params:
      num_regression_tasks = model_params["num_regression_tasks"]
    else:
      num_regression_tasks = 0
    if num_classification_tasks and num_regression_tasks:
      raise AssertionError(
          'Dual classification/regression models are not supported.')
    self.num_tasks = num_classification_tasks + num_regression_tasks
    if self.num_tasks == 0:
      raise AssertionError('Must specify one of '
                           'num_classification_tasks or num_regression_tasks.')

    if summary_writer is None:
      summary_writer = tf.train.SummaryWriter(logdir)
    self.summary_writer = summary_writer

  def build(self):
    """Define the core model.

    NOTE(user): Operations defined here should be in their own name scope to
    avoid any ambiguity when restoring checkpoints.

    Raises:
      NotImplementedError: if not overridden by concrete subclass.
    """
    raise NotImplementedError('Must be overridden by concrete subclass')

  def construct_feed_dict(self, X_b, y_b, w_b, ids_b):
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
    print("TensorflowClassiifer.add_weight_placeholders")
    print("self.weights")
    print(self.weights)

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

  def training_cost(self):
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
              tf.scalar_summary('cost' + task_str,
                                model_ops.MovingAverage(gradient_cost,
                                                        self.global_step))
              gradient_costs.append(gradient_cost)

            with tf.name_scope('old_cost'):
              old_cost = tf.div(
                  tf.reduce_sum(weighted_cost),
                  tf.reduce_sum(self.weights[task]) + epsilon)
              tf.scalar_summary('old-cost' + task_str,
                                model_ops.MovingAverage(old_cost,
                                                        self.global_step))
              old_costs.append(old_cost)

        # aggregated costs
        with self._shared_name_scope('aggregated'):
          with tf.name_scope('gradient'):
            loss = tf.add_n(gradient_costs)
          with tf.name_scope('old_cost'):
            old_loss = tf.add_n(old_costs)

          # weight decay
          if model_params["penalty"] != 0.0:
            penalty = WeightDecay(model_params)
            loss += penalty
            old_loss += penalty

        # loss used for gradient calculation
        self.loss = loss

        # (smoothed) summaries
        tf.scalar_summary('Total Cost',
                          model_ops.MovingAverage(loss, self.global_step))
        tf.scalar_summary('Total Old-Style Cost',
                          model_ops.MovingAverage(old_loss, self.global_step))

      return weighted_costs

  def setup(self):
    """Add ops common to training/eval to the graph."""
    with self.graph.as_default():
      with tf.name_scope('core_model'):
        self.build()
      self.add_labels_and_weights()
      self.global_step = tf.Variable(0, name='global_step', trainable=False)

  def merge_updates(self):
    """Group updates into a single op."""
    with self.graph.as_default():
      updates = tf.get_default_graph().get_collection('updates')
      if updates:
        self.updates = tf.group(*updates, name='updates')
      else:
        self.updates = tf.no_op(name='updates')

  def get_training_op(self):
    """Get training op for applying gradients to variables.

    Subclasses that need to do anything fancy with gradients should override
    this method.

    Returns:
    A training op.
    """
    opt = Optimizer(self.model_params)
    return opt.minimize(self.loss, global_step=self.global_step, name='train')

  def get_summary_op(self):
    """Get summary op for computing all summaries during training.

    Returns:
    A summary op.
    """
    with self.graph.as_default():
      return tf.merge_all_summaries()


  def fit(self,
          dataset,
          max_steps=None,
          summaries=False,
          save_model_secs=60,
          save_summary_secs=30,
          max_checkpoints_to_keep=5):
    """Fit the model.

    Args:
      dataset: Dataset object that represents data on disk.
      max_steps: Maximum number of training steps. If not provided, will
        train indefinitely.
      summaries: If True, add summaries for model parameters.
      save_model_secs: Integer. Saves a checkpoint at this interval in seconds.
      save_summary_secs: Integer. Saves a summary event file at this interval in
        seconds.
      max_checkpoints_to_keep: Integer. Maximum number of checkpoints to keep;
        older checkpoints will be deleted.

    Raises:
      AssertionError: If model is not in training mode.
    """
    with self.graph.as_default():
      assert model_ops.is_training()
      self.setup()
      self.training_cost()
      self.merge_updates()
      self.require_attributes(['loss', 'global_step', 'updates'])
      if summaries:
        self.add_summaries()
      train_op = self.get_training_op()
      summary_op = self.get_summary_op()
      no_op = tf.no_op()
      # TODO(rbharath): This should probably be uncommented!
      tf.train.write_graph(
          tf.get_default_graph().as_graph_def(), self.logdir, 'train.pbtxt')
      self.summary_writer.add_graph(tf.get_default_graph().as_graph_def())
      last_checkpoint_time = time.time()
      last_summary_time = time.time()
      with self._get_shared_session() as sess:
        sess.run(tf.initialize_all_variables())
        saver = tf.train.Saver(max_to_keep=max_checkpoints_to_keep)
        # Save an initial checkpoint.
        saver.save(sess, self._save_path, global_step=self.global_step)
        for (X_b, y_b, w_b, ids_b) in dataset.iterbatches(self.model_params["batch_size"]):
          # Run training op and compute summaries.
          feed_dict = self.construct_feed_dict(X_b, y_b, w_b, ids_b)
          secs_since_summary = time.time() - last_summary_time
          if secs_since_summary > save_summary_secs:
            this_summary_op = summary_op
          else:
            this_summary_op = no_op
          step, loss, _, summary = sess.run(
              [train_op.values()[0], self.loss, self.updates, this_summary_op],
              feed_dict=feed_dict)
          if summary is not None:
            self.summary_writer.add_summary(summary, global_step=step)
            last_summary_time = time.time()
          # Save model checkpoints.
          secs_since_checkpoint = time.time() - last_checkpoint_time
          if secs_since_checkpoint > save_model_secs:
            logging.info('step %d: %g', step, loss)
            saver.save(sess, self._save_path, global_step=self.global_step)
            last_checkpoint_time = time.time()
          # Quit when we reach max_steps.
          if max_steps is not None and step >= max_steps:
            break
        # Always save a final checkpoint when complete.
        saver.save(sess, self._save_path, global_step=self.global_step)

  def save(self, out_dir):
    """
    No-op since tf models save themselves during fit()
    """
    pass
  

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

  def CloseSharedSession(self):
    if self._shared_session:
      self._shared_session.close()

  def restore(self, checkpoint):
    """Restores the model from the provided training checkpoint.

    Args:
      checkpoint: string. Path to checkpoint file.
    """
    with self.graph.as_default():
      if self._restored_model:
        return

      self.setup()
      self.add_output_ops()  # add softmax heads
      #saver = tf.train.Saver(tf.variables.all_variables())
      saver = tf.train.Saver()
      saver.restore(self._get_shared_session(),
                    tf_utils.ParseCheckpoint(checkpoint))
      self.global_step_number = int(self._get_shared_session().run(self.global_step))

      self._restored_model = True

  def load(self, model_dir):
    """
    Loads model from disk. Thin wrapper around restore() for consistency.
    """
    with self.graph.as_default():
      assert not model_ops.is_training()
      if model_dir != self.logdir:
        raise ValueError("Cannot load from directory that is not logdir.")
      last_checkpoint = self._find_last_checkpoint()
      self.restore(last_checkpoint)

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
          

  def Eval(self, input_generator, checkpoint, metrics=None):
    """Evaluate the model.

    Args:
      input_generator: Generator that returns a feed_dict for feeding
        Placeholders in the model graph.
      checkpoint: Checkpoint filename.
      metrics: List of metrics to compute. Defaults to self.default_metrics,
        which is set in subclasses.

    Returns:
      step: Global step for this eval.
      results: A dict mapping metric names to numpy arrays containing metric
        values for each task.
    """
    self.Restore(checkpoint)
    output, labels, weights = self.get_model_output(input_generator)
    y_true, y_pred = self.ParseModelOutput(output, labels, weights)

    # keep counts for each class as a sanity check
    counts = self.ExampleCounts(y_true)

    # compute metrics
    if metrics is None:
      metrics = self.default_metrics
    metric_values = {}
    for metric in metrics:
      metric_values[metric] = self.ComputeMetric(y_true, y_pred, metric)
    self.ReportEval(metric_values, counts=counts,
                    global_step=self.global_step_number)
    return self.global_step_number, metric_values

  def ComputeMetric(self, y_true, y_pred, metric_str, threshold=0.5):
    """Compute a performance metric for each task.

    Args:
      y_true: A list of arrays containing true values for each task.
      y_pred: A list of arrays containing predicted values for each task.
      metric_str: String description of the metric to compute. Must be in
        metrics.METRICS.
      threshold: Float threshold to apply to probabilities for positive/negative
        class assignment.

    Returns:
      A numpy array containing metric values for each task.
    """
    computed_metrics = []
    for task in xrange(self.num_tasks):
      yt = y_true[task]
      yp = y_pred[task]
      try:
        metric_value = metrics.compute_metric(yt, yp, metric_str,
                                                      threshold=threshold)
      except (AssertionError, ValueError) as e:
        warnings.warn('Error calculating metric %s for task %d: %s'
                      % (metric_str, task, e))
        metric_value = np.nan
      computed_metrics.append(metric_value)
    return computed_metrics

  def predict(self, dataset, transformers):
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
    with self.graph.as_default():
      assert not model_ops.is_training()
      assert self._restored_model
      self.require_attributes(['output', 'labels', 'weights'])

      # run eval data through the model
      num_tasks = self.num_tasks
      output, labels, weights = [], [], []
      start = time.time()
      with self._get_shared_session().as_default():
        batches_per_summary = 1000
        seconds_per_summary = 0
        batch_count = -1.0
        #for feed_dict in input_generator:
        for (X_b, y_b, w_b, ids_b) in dataset.iterbatches(self.model_params["batch_size"]):
          feed_dict = self.construct_feed_dict(X_b, y_b, w_b, ids_b)
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
          print("valid")
          print(valid)
          # only take valid outputs
          if np.count_nonzero(~valid):
            batch_output = batch_output[valid]
            batch_labels = batch_labels[valid]
            batch_weights = batch_weights[valid]
          output.append(batch_output)
          labels.append(batch_labels)
          weights.append(batch_weights)

          # Writes summary for tracking eval progress.
          seconds_per_summary += (time.time() - batch_start)
          self.require_attributes(['summary_writer'])
          if batch_count % batches_per_summary == 0:
            mean_seconds_per_batch = seconds_per_summary / batches_per_summary
            seconds_per_summary = 0
            summaries = [
                tf.scalar_summary('secs/batch', mean_seconds_per_batch),
                tf.scalar_summary('batches_evaluated', batch_count)
            ]
            self.summary_writer.add_summary(tf.merge_summary(summaries).eval(),
                                            global_step=self.global_step_number)
            self.summary_writer.flush()

        logging.info('Eval took %g seconds', time.time() - start)

        output = np.concatenate(output)
        labels = np.concatenate(labels)
        weights = np.concatenate(weights)

      return output, labels, weights

  def ReportEval(self, metrics, global_step, counts=None, name=None):
    """Write Eval summaries.

    Args:
      metrics: Dict mapping metric names to numpy arrays containing metric
        values for each task.
      global_step: Integer. Global step number inference was run on.
      counts: Dict mapping class names to counts.
      name: String name for this group of metrics. Useful for organizing
        metrics calculated using different subsets of the data.
    """
    with self.graph.as_default():
      # create a DataFrame to hold results
      data = dict()
      if counts is not None:
        data.update({'count_%s' % group: values
                     for group, values in counts.iteritems()})
      data.update(metrics)
      df = pd.DataFrame(data)
      print('Eval at step: %d' % global_step)
      print(df)
      # add global step to df
      df['step'] = global_step

      # save an update to disk
      filename = os.path.join(self.logdir, 'eval-%d.pkl' % global_step)
      with open(filename, 'w') as f:
        pickle.dump(df, f, pickle.HIGHEST_PROTOCOL)

      # write a summary for each metric
      self.require_attributes(['summary_writer'])
      with tf.Session(self.master):
        summaries = []
        prefix = '' if name is None else '%s - ' % name
        for metric_name, results in metrics.iteritems():
          for task in xrange(self.num_tasks):
            task_str = str(task).zfill(len(str(self.num_tasks)))
            summaries.append(
                tf.scalar_summary('%s%s_%s' % (prefix, metric_name, task_str),
                                  results[task]))
          summaries.append(tf.scalar_summary(
              '%sMean %s' % (prefix, metric_name), np.mean(results)))
          summaries.append(tf.scalar_summary(
              '%sMedian %s' % (prefix, metric_name), np.median(results)))
        self.summary_writer.add_summary(tf.merge_summary(summaries).eval(),
                                        global_step=global_step)
        self.summary_writer.flush()

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

  def add_summaries(self):
    """Add summaries for model parameters."""
    with self.graph.as_default():
      for var in tf.trainable_variables():
        if 'BatchNormalize' in var.name:
          continue
        tf.histogram_summary(var.name, var)


class TensorflowClassifier(TensorflowModel):
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

  def training_cost(self):
    """Calculate additional classifier-specific costs.

    Returns:
      A list of tensors with shape batch_size containing costs for each task.
    """
    with self.graph.as_default():
      weighted_costs = super(TensorflowClassifier, self).training_cost()  # calculate loss
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
                tf.scalar_summary('%s_%s' % (name, task_str),
                                  model_ops.MovingAverage(cond_cost,
                                                          self.global_step))
                cond_costs[name].append(cond_cost)

        # aggregated costs
        with self._shared_name_scope('aggregated'):
          with tf.name_scope('pos_cost'):
            pos_cost = tf.add_n(cond_costs['pos'])
          with tf.name_scope('neg_cost'):
            neg_cost = tf.add_n(cond_costs['neg'])

        # (smoothed) summaries
        tf.scalar_summary('Total Neg Cost',
                          model_ops.MovingAverage(neg_cost, self.global_step))
        tf.scalar_summary('Total Pos Cost',
                          model_ops.MovingAverage(pos_cost, self.global_step))

      # keep track of the number of positive examples seen by each task
      with tf.name_scope('counts'):
        for task in xrange(num_tasks):
          num_pos = tf.Variable(0.0, name='num_pos_%d' % task, trainable=False)
          # the assignment must occur on the same device as the variable
          with tf.device(num_pos.device):
            tf.get_default_graph().add_to_collection(
                'updates', num_pos.assign_add(
                    tf.reduce_sum(self.labels[task][:model_params["batch_size"], 1])))
          tf.scalar_summary(num_pos.name, num_pos)

      return weighted_costs


  def add_output_ops(self):
    """Replace logits with softmax outputs."""
    with self.graph.as_default():
      softmax = []
      with tf.name_scope('inference'):
        for i, logits in enumerate(self.output):
          softmax.append(tf.nn.softmax(logits, name='softmax_%d' % i))
      self.output = softmax

  def ExampleCounts(self, y_true):
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
      print("TensorflowClassiifer.add_label_placeholders")
      print("self.labels")
      print(self.labels)

  def ParseModelOutput(self, output, labels, weights):
    """Parse model output to get true and predicted values for each task.

    Args:
      output: Numpy array containing model output with shape
        batch_size x num_tasks x num_classes.
      labels: Numpy array containing one-hot example labels with shape
        batch_size x num_tasks x num_classes.
      weights: Numpy array containing example weights with shape
        batch_size x num_tasks.

    Returns:
      y_true: List of numpy arrays containing true labels, one for each task.
      y_pred: List of numpy arrays containing predicted labels, one for each
        task.
    """
    y_true, y_pred = [], []
    for task in xrange(self.model_params["num_classification_tasks"]):
      # mask examples with zero weight
      mask = weights[:, task] > 0
      # get true class labels
      y_true.append(labels[mask, task, 1])
      # get positive class probabilities for predictions
      y_pred.append(output[mask, task, 1])
    return y_true, y_pred


class TensorflowRegressor(TensorflowModel):
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
    return tf.mul(tf.nn.l2_loss(output - labels), weights)

  def ExampleCounts(self, y_true):
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

  def ParseModelOutput(self, output, labels, weights):
    """Parse model output to get true and predicted values for each task.

    Args:
      output: Numpy array containing model output with shape
        batch_size x num_tasks x num_classes.
      labels: Numpy array containing one-hot example labels with shape
        batch_size x num_tasks x num_classes.
      weights: Numpy array containing example weights with shape
        batch_size x num_tasks.

    Returns:
      y_true: List of numpy arrays containing true labels, one for each task.
      y_pred: List of numpy arrays containing predicted labels, one for each
        task.
    """
    # build arrays of true and predicted values for R-squared calculation
    y_true, y_pred = [], []
    for task in xrange(self.model_params["num_regression_tasks"]):
      mask = weights[:, task] > 0  # ignore examples with zero weight
      y_true.append(labels[mask, task])
      y_pred.append(output[mask, task])
    return y_true, y_pred


def Optimizer(model_params):
  """Create model optimizer.

  Args:
    model_params: ModelConfig.

  Returns:
    A training Optimizer.

  Raises:
    NotImplementedError: If an unsupported optimizer is requested.
  """
  # TODO(user): gradient clipping (see Minimize)
  if model_params["optimizer"] == 'adagrad':
    train_op = tf.train.AdagradOptimizer(model_params["learning_rate"])
  elif model_params["optimizer"] == 'adam':
    train_op = tf.train.AdamOptimizer(model_params["learning_rate"])
  elif model_params["optimizer"] == 'momentum':
    train_op = tf.train.MomentumOptimizer(model_params["learning_rate"],
                                          model_params["memory"])
  elif model_params["optimizer"] == 'rmsprop':
    train_op = tf.train.RMSPropOptimizer(model_params["learning_rate"],
                                         model_params["memory"])
  elif model_params["optimizer"] == 'sgd':
    train_op = tf.train.GradientDescentOptimizer(model_params["learning_rate"])
  else:
    raise NotImplementedError('Unsupported optimizer %s' % model_params["optimizer"])
  return train_op


def WeightDecay(model_params):
  """Add weight decay.

  Args:
    model_params: ModelConfig.

  Returns:
    A scalar tensor containing the weight decay cost.

  Raises:
    NotImplementedError: If an unsupported penalty type is requested.
  """
  variables = []
  # exclude bias variables
  for v in tf.trainable_variables():
    if v.get_shape().ndims == 2:
      variables.append(v)

  with tf.name_scope('weight_decay'):
    if model_params["penalty_type"] == 'l1':
      cost = tf.add_n([tf.reduce_sum(tf.Abs(v)) for v in variables])
    elif model_params["penalty_type"] == 'l2':
      cost = tf.add_n([tf.nn.l2_loss(v) for v in variables])
    else:
      raise NotImplementedError('Unsupported penalty_type %s' %
                                model_params["penalty_type"])
    cost *= model_params["penalty"]
    tf.scalar_summary('Weight Decay Cost', cost)
  return cost
