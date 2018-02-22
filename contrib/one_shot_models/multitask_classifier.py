"""
Implements a multitask graph convolutional classifier.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Han Altae-Tran and Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "MIT"

import warnings
import os
import sys
import numpy as np
import tensorflow as tf
import sklearn.metrics
import tempfile
from deepchem.data import pad_features
from deepchem.utils.save import log
from deepchem.models import Model
from deepchem.nn.copy import Input
from deepchem.nn.copy import Dense
from deepchem.nn import model_ops
# TODO(rbharath): Find a way to get rid of this import?
from deepchem.models.tf_new_models.graph_topology import merge_dicts


def get_loss_fn(final_loss):
  # Obtain appropriate loss function
  if final_loss == 'L2':

    def loss_fn(x, t):
      diff = tf.subtract(x, t)
      return tf.reduce_sum(tf.square(diff), 0)
  elif final_loss == 'weighted_L2':

    def loss_fn(x, t, w):
      diff = tf.subtract(x, t)
      weighted_diff = tf.multiply(diff, w)
      return tf.reduce_sum(tf.square(weighted_diff), 0)
  elif final_loss == 'L1':

    def loss_fn(x, t):
      diff = tf.subtract(x, t)
      return tf.reduce_sum(tf.abs(diff), 0)
  elif final_loss == 'huber':

    def loss_fn(x, t):
      diff = tf.subtract(x, t)
      return tf.reduce_sum(
          tf.minimum(0.5 * tf.square(diff),
                     huber_d * (tf.abs(diff) - 0.5 * huber_d)), 0)
  elif final_loss == 'cross_entropy':

    def loss_fn(x, t, w):
      costs = tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=t)
      weighted_costs = tf.multiply(costs, w)
      return tf.reduce_sum(weighted_costs)
  elif final_loss == 'hinge':

    def loss_fn(x, t, w):
      t = tf.multiply(2.0, t) - 1
      costs = tf.maximum(0.0, 1.0 - tf.multiply(t, x))
      weighted_costs = tf.multiply(costs, w)
      return tf.reduce_sum(weighted_costs)

  return loss_fn


class MultitaskGraphClassifier(Model):

  def __init__(self,
               model,
               n_tasks,
               n_feat,
               logdir=None,
               batch_size=50,
               final_loss='cross_entropy',
               learning_rate=.001,
               optimizer_type="adam",
               learning_rate_decay_time=1000,
               beta1=.9,
               beta2=.999,
               pad_batches=True,
               verbose=True):

    warnings.warn("MultitaskGraphClassifier is deprecated. "
                  "Will be removed in DeepChem 1.4.", DeprecationWarning)
    super(MultitaskGraphClassifier, self).__init__(
        model_dir=logdir, verbose=verbose)
    self.n_tasks = n_tasks
    self.final_loss = final_loss
    self.model = model
    self.sess = tf.Session(graph=self.model.graph)

    with self.model.graph.as_default():
      # Extract model info
      self.batch_size = batch_size
      self.pad_batches = pad_batches
      # Get graph topology for x
      self.graph_topology = self.model.get_graph_topology()
      self.feat_dim = n_feat

      # Raw logit outputs
      self.logits = self.build()
      self.loss_op = self.add_training_loss(self.final_loss, self.logits)
      self.outputs = self.add_softmax(self.logits)

      self.learning_rate = learning_rate
      self.T = learning_rate_decay_time
      self.optimizer_type = optimizer_type

      self.optimizer_beta1 = beta1
      self.optimizer_beta2 = beta2

      # Set epsilon
      self.epsilon = 1e-7
      self.add_optimizer()

      # Initialize
      self.init_fn = tf.global_variables_initializer()
      self.sess.run(self.init_fn)

      # Path to save checkpoint files, which matches the
      # replicated supervisor's default path.
      self._save_path = os.path.join(self.model_dir, 'model.ckpt')

  def build(self):
    # Create target inputs
    self.label_placeholder = tf.placeholder(
        dtype='bool', shape=(None, self.n_tasks), name="label_placeholder")
    self.weight_placeholder = tf.placeholder(
        dtype='float32', shape=(None, self.n_tasks), name="weight_placholder")

    feat = self.model.return_outputs()
    ################################################################ DEBUG
    #print("multitask classifier")
    #print("feat")
    #print(feat)
    ################################################################ DEBUG
    output = model_ops.multitask_logits(feat, self.n_tasks)
    return output

  def add_optimizer(self):
    if self.optimizer_type == "adam":
      self.optimizer = tf.train.AdamOptimizer(
          self.learning_rate,
          beta1=self.optimizer_beta1,
          beta2=self.optimizer_beta2,
          epsilon=self.epsilon)
    else:
      raise ValueError("Optimizer type not recognized.")

    # Get train function
    self.train_op = self.optimizer.minimize(self.loss_op)

  def construct_feed_dict(self, X_b, y_b=None, w_b=None, training=True):
    """Get initial information about task normalization"""
    # TODO(rbharath): I believe this is total amount of data
    n_samples = len(X_b)
    if y_b is None:
      y_b = np.zeros((n_samples, self.n_tasks))
    if w_b is None:
      w_b = np.zeros((n_samples, self.n_tasks))
    targets_dict = {self.label_placeholder: y_b, self.weight_placeholder: w_b}

    # Get graph information
    atoms_dict = self.graph_topology.batch_to_feed_dict(X_b)

    # TODO (hraut->rhbarath): num_datapoints should be a vector, with ith element being
    # the number of labeled data points in target_i. This is to normalize each task
    # num_dat_dict = {self.num_datapoints_placeholder : self.}

    # Get other optimizer information
    # TODO(rbharath): Figure out how to handle phase appropriately
    feed_dict = merge_dicts([targets_dict, atoms_dict])
    return feed_dict

  def add_training_loss(self, final_loss, logits):
    """Computes loss using logits."""
    loss_fn = get_loss_fn(final_loss)  # Get loss function
    task_losses = []
    # label_placeholder of shape (batch_size, n_tasks). Split into n_tasks
    # tensors of shape (batch_size,)
    task_labels = tf.split(
        axis=1, num_or_size_splits=self.n_tasks, value=self.label_placeholder)
    task_weights = tf.split(
        axis=1, num_or_size_splits=self.n_tasks, value=self.weight_placeholder)
    for task in range(self.n_tasks):
      task_label_vector = task_labels[task]
      task_weight_vector = task_weights[task]
      # Convert the labels into one-hot vector encodings.
      one_hot_labels = tf.to_float(
          tf.one_hot(tf.to_int32(tf.squeeze(task_label_vector)), 2))
      # Since we use tf.nn.softmax_cross_entropy_with_logits note that we pass in
      # un-softmaxed logits rather than softmax outputs.
      task_loss = loss_fn(logits[task], one_hot_labels, task_weight_vector)
      task_losses.append(task_loss)
    # It's ok to divide by just the batch_size rather than the number of nonzero
    # examples (effect averages out)
    total_loss = tf.add_n(task_losses)
    total_loss = tf.div(total_loss, self.batch_size)
    return total_loss

  def add_softmax(self, outputs):
    """Replace logits with softmax outputs."""
    softmax = []
    with tf.name_scope('inference'):
      for i, logits in enumerate(outputs):
        softmax.append(tf.nn.softmax(logits, name='softmax_%d' % i))
    return softmax

  def fit(self,
          dataset,
          nb_epoch=10,
          max_checkpoints_to_keep=5,
          log_every_N_batches=50,
          checkpoint_interval=10,
          **kwargs):
    # Perform the optimization
    log("Training for %d epochs" % nb_epoch, self.verbose)

    # TODO(rbharath): Disabling saving for now to try to debug.
    for epoch in range(nb_epoch):
      log("Starting epoch %d" % epoch, self.verbose)
      for batch_num, (X_b, y_b, w_b, ids_b) in enumerate(
          dataset.iterbatches(self.batch_size, pad_batches=self.pad_batches)):
        if batch_num % log_every_N_batches == 0:
          log("On batch %d" % batch_num, self.verbose)
        self.sess.run(
            self.train_op, feed_dict=self.construct_feed_dict(X_b, y_b, w_b))

  def save(self):
    """
    No-op since this model doesn't currently support saving...
    """
    pass

  def predict(self, dataset, transformers=[], **kwargs):
    """Wraps predict to set batch_size/padding."""
    return super(MultitaskGraphClassifier, self).predict(
        dataset, transformers, batch_size=self.batch_size)

  def predict_proba(self, dataset, transformers=[], n_classes=2, **kwargs):
    """Wraps predict_proba to set batch_size/padding."""
    return super(MultitaskGraphClassifier, self).predict_proba(
        dataset, transformers, n_classes=n_classes, batch_size=self.batch_size)

  def predict_on_batch(self, X):
    """Return model output for the provided input.
    """
    if self.pad_batches:
      X = pad_features(self.batch_size, X)
    # run eval data through the model
    n_tasks = self.n_tasks
    with self.sess.as_default():
      feed_dict = self.construct_feed_dict(X)
      # Shape (n_samples, n_tasks)
      batch_outputs = self.sess.run(self.outputs, feed_dict=feed_dict)

    n_samples = len(X)
    outputs = np.zeros((n_samples, self.n_tasks))
    for task, output in enumerate(batch_outputs):
      outputs[:, task] = np.argmax(output, axis=1)
    return outputs

  def predict_proba_on_batch(self, X, n_classes=2):
    """Returns class probabilities on batch"""
    # run eval data through the model
    if self.pad_batches:
      X = pad_features(self.batch_size, X)
    n_tasks = self.n_tasks
    with self.sess.as_default():
      feed_dict = self.construct_feed_dict(X)
      batch_outputs = self.sess.run(self.outputs, feed_dict=feed_dict)

    n_samples = len(X)
    outputs = np.zeros((n_samples, self.n_tasks, n_classes))
    for task, output in enumerate(batch_outputs):
      outputs[:, task, :] = output
    return outputs

  def get_num_tasks(self):
    """Needed to use Model.predict() from superclass."""
    return self.n_tasks
