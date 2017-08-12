# -*- coding: utf-8 -*-
"""
Created on Tue Nov 08 14:10:02 2016

@author: Zhenqin Wu
"""
import warnings
import tensorflow as tf
import numpy as np
import os
import time

from deepchem.metrics import from_one_hot
from deepchem.models.tensorflow_models import TensorflowGraph
from deepchem.models.tensorflow_models import TensorflowGraphModel
from deepchem.nn import model_ops
from deepchem.utils.save import log
from deepchem.data import pad_features
from deepchem.metrics import to_one_hot


def weight_decay(penalty_type, penalty):
  # due to the different shape of weight(ndims=2) and bias(ndims=1),
  # will using this version for logreg
  variables = []
  # exclude bias variables
  for v in tf.trainable_variables():
    if v.get_shape().as_list()[0] > 1:
      variables.append(v)

  with tf.name_scope('weight_decay'):
    if penalty_type == 'l1':
      cost = tf.add_n([tf.reduce_sum(tf.abs(v)) for v in variables])
    elif penalty_type == 'l2':
      cost = tf.add_n([tf.nn.l2_loss(v) for v in variables])
    else:
      raise NotImplementedError('Unsupported penalty_type %s' % penalty_type)
    cost *= penalty
    tf.summary.scalar('Weight Decay Cost', cost)
  return cost


class TensorflowLogisticRegression(TensorflowGraphModel):
  """ A simple tensorflow based logistic regression model. """

  def build(self, graph, name_scopes, training):
    """Constructs the graph architecture of model: n_tasks * sigmoid nodes.

    This method creates the following Placeholders:
      mol_features: Molecule descriptor (e.g. fingerprint) tensor with shape
        batch_size x n_features.
    """
    warnings.warn("TensorflowLogisticRegression is deprecated. "
                  "Will be removed in DeepChem 1.4.", DeprecationWarning)
    placeholder_scope = TensorflowGraph.get_placeholder_scope(
        graph, name_scopes)
    n_features = self.n_features
    with graph.as_default():
      with placeholder_scope:
        mol_features = tf.placeholder(
            tf.float32, shape=[None, n_features], name='mol_features')

      weight_init_stddevs = self.weight_init_stddevs
      bias_init_consts = self.bias_init_consts
      lg_list = []

      label_placeholders = self.add_label_placeholders(graph, name_scopes)
      weight_placeholders = self.add_example_weight_placeholders(
          graph, name_scopes)
      if training:
        graph.queue = tf.FIFOQueue(
            capacity=5,
            dtypes=[tf.float32] *
            (len(label_placeholders) + len(weight_placeholders) + 1))
        graph.enqueue = graph.queue.enqueue([mol_features] + label_placeholders
                                            + weight_placeholders)
        queue_outputs = graph.queue.dequeue()
        labels = queue_outputs[1:len(label_placeholders) + 1]
        weights = queue_outputs[len(label_placeholders) + 1:]
        prev_layer = queue_outputs[0]
      else:
        labels = label_placeholders
        weights = weight_placeholders
        prev_layer = mol_features

      for task in range(self.n_tasks):
        #setting up n_tasks nodes(output nodes)
        lg = model_ops.fully_connected_layer(
            tensor=prev_layer,
            size=1,
            weight_init=tf.truncated_normal(
                shape=[self.n_features, 1], stddev=weight_init_stddevs[0]),
            bias_init=tf.constant(value=bias_init_consts[0], shape=[1]))
        lg_list.append(lg)
    return (lg_list, labels, weights)

  def add_label_placeholders(self, graph, name_scopes):
    #label placeholders with size batch_size * 1
    labels = []
    placeholder_scope = TensorflowGraph.get_placeholder_scope(
        graph, name_scopes)
    with placeholder_scope:
      for task in range(self.n_tasks):
        labels.append(
            tf.identity(
                tf.placeholder(
                    tf.float32, shape=[None, 1], name='labels_%d' % task)))
    return labels

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
            # using self-defined regularization
            penalty = weight_decay(self.penalty_type, self.penalty)
            loss += penalty

      return loss

  def cost(self, logits, labels, weights):
    return tf.multiply(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels),
        weights)

  def add_output_ops(self, graph, output):
    # adding output nodes of sigmoid function
    with graph.as_default():
      sigmoid = []
      with tf.name_scope('inference'):
        for i, logits in enumerate(output):
          sigmoid.append(tf.nn.sigmoid(logits, name='sigmoid_%d' % i))
      output = sigmoid
    return output

  def construct_feed_dict(self, X_b, y_b=None, w_b=None, ids_b=None):

    orig_dict = {}
    orig_dict["mol_features"] = X_b
    for task in range(self.n_tasks):
      if y_b is not None:
        y_2column = to_one_hot(y_b[:, task])
        # fix the size to be [?,1]
        orig_dict["labels_%d" % task] = y_2column[:, 1:2]
      else:
        # Dummy placeholders
        orig_dict["labels_%d" % task] = np.zeros((self.batch_size, 1))
      if w_b is not None:
        orig_dict["weights_%d" % task] = w_b[:, task]
      else:
        # Dummy placeholders
        orig_dict["weights_%d" % task] = np.ones((self.batch_size,))
    return TensorflowGraph.get_feed_dict(orig_dict)

  def predict_proba_on_batch(self, X):
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
        # transfer 2D prediction tensor to 2D x n_classes(=2)
        complimentary = np.ones(np.shape(batch_outputs))
        complimentary = complimentary - batch_outputs
        batch_outputs = np.concatenate(
            [complimentary, batch_outputs], axis=batch_outputs.ndim - 1)
        # reshape to batch_size x n_tasks x ...
        if batch_outputs.ndim == 3:
          batch_outputs = batch_outputs.transpose((1, 0, 2))
        elif batch_outputs.ndim == 2:
          batch_outputs = batch_outputs.transpose((1, 0))
        else:
          raise ValueError('Unrecognized rank combination for output: %s ' %
                           (batch_outputs.shape,))

      outputs = batch_outputs

    return np.copy(outputs)

  def predict_on_batch(self, X):

    if self.pad_batches:
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
        # transfer 2D prediction tensor to 2D x n_classes(=2)
        complimentary = np.ones(np.shape(batch_output))
        complimentary = complimentary - batch_output
        batch_output = np.concatenate(
            [complimentary, batch_output], axis=batch_output.ndim - 1)
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

    return np.copy(outputs)
