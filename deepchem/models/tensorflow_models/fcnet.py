"""TensorFlow implementation of fully connected networks. 
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import time
import numpy as np
import tensorflow as tf

from deepchem.nn import model_ops
from deepchem.metrics import from_one_hot
from deepchem.models.tensorflow_models import TensorflowGraph
from deepchem.models.tensorflow_models import TensorflowClassifier
from deepchem.models.tensorflow_models import TensorflowRegressor
from deepchem.metrics import to_one_hot

class TensorflowMultiTaskClassifier(TensorflowClassifier):
  """Implements an icml model as configured in a model_config.proto."""

  def build(self, graph, name_scopes, training):
    """Constructs the graph architecture as specified in its config.

    This method creates the following Placeholders:
      mol_features: Molecule descriptor (e.g. fingerprint) tensor with shape
        batch_size x n_features.
    """
    placeholder_scope = TensorflowGraph.get_placeholder_scope(
        graph, name_scopes)
    n_features = self.n_features
    with graph.as_default():
      with placeholder_scope:
        self.mol_features = tf.placeholder(
            tf.float32,
            shape=[None, n_features],
            name='mol_features')

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
        layer = tf.nn.relu(model_ops.fully_connected_layer(
            tensor=prev_layer,
            size=layer_sizes[i],
            weight_init=tf.truncated_normal(
                shape=[prev_layer_size, layer_sizes[i]],
                stddev=weight_init_stddevs[i]),
            bias_init=tf.constant(value=bias_init_consts[i],
                                  shape=[layer_sizes[i]])))
        layer = model_ops.dropout(layer, dropouts[i], training)
        prev_layer = layer
        prev_layer_size = layer_sizes[i]

      output = model_ops.multitask_logits(
          layer, self.n_tasks)
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
        orig_dict["labels_%d" % task] = to_one_hot(y_b[:, task])
      else:
        # Dummy placeholders
        orig_dict["labels_%d" % task] = np.squeeze(to_one_hot(
            np.zeros((self.batch_size,))))
      if w_b is not None:
        orig_dict["weights_%d" % task] = w_b[:, task]
      else:
        # Dummy placeholders
        orig_dict["weights_%d" % task] = np.ones(
            (self.batch_size,)) 
    return TensorflowGraph.get_feed_dict(orig_dict)


class TensorflowMultiTaskRegressor(TensorflowRegressor):
  """Implements an icml model as configured in a model_config.proto."""

  def build(self, graph, name_scopes, training):
    """Constructs the graph architecture as specified in its config.

    This method creates the following Placeholders:
      mol_features: Molecule descriptor (e.g. fingerprint) tensor with shape
        batch_size x n_features.
    """
    n_features = self.n_features
    placeholder_scope = TensorflowGraph.get_placeholder_scope(
        graph, name_scopes)
    with graph.as_default():
      with placeholder_scope:
        self.mol_features = tf.placeholder(
            tf.float32,
            shape=[None, n_features],
            name='mol_features')

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
        layer = tf.nn.relu(model_ops.fully_connected_layer(
            tensor=prev_layer,
            size=layer_sizes[i],
            weight_init=tf.truncated_normal(
                shape=[prev_layer_size, layer_sizes[i]],
                stddev=weight_init_stddevs[i]),
            bias_init=tf.constant(value=bias_init_consts[i],
                                  shape=[layer_sizes[i]])))
        layer = model_ops.dropout(layer, dropouts[i], training)
        prev_layer = layer
        prev_layer_size = layer_sizes[i]

      output = []
      for task in range(self.n_tasks):
        output.append(tf.squeeze(
            model_ops.fully_connected_layer(
                tensor=prev_layer,
                size=layer_sizes[i],
                weight_init=tf.truncated_normal(
                    shape=[prev_layer_size, 1],
                    stddev=weight_init_stddevs[i]),
                bias_init=tf.constant(value=bias_init_consts[i],
                                      shape=[1]))))
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
        orig_dict["labels_%d" % task] = np.squeeze(
            np.zeros((self.batch_size,)))
      if w_b is not None:
        orig_dict["weights_%d" % task] = w_b[:, task]
      else:
        # Dummy placeholders
        orig_dict["weights_%d" % task] = np.ones(
            (self.batch_size,)) 
    return TensorflowGraph.get_feed_dict(orig_dict)

class TensorflowMultiTaskFitTransformRegressor(TensorflowRegressor):
  """Implements a TensorflowMultiTaskRegressor that performs on-the-fly transformation during fit/predict"""

  def __init__(self, n_tasks, n_features, logdir=None, layer_sizes=[1000],
               weight_init_stddevs=[.02], bias_init_consts=[1.], penalty=0.0,
               penalty_type="l2", dropouts=[0.5], learning_rate=.001,
               momentum=.9, optimizer="adam", batch_size=50, n_classes=2,
               fit_transformers=[], verbose=True, seed=None, **kwargs):

    self.fit_transformers = fit_transformers
    TensorflowGraphModel.__init__(self, n_tasks, n_features, logdir=logdir, 
	       layer_sizes=layer_sizes, weight_init_stddevs=weight_init_stddevs, 
	       bias_init_consts=bias_init_consts, penalty=penalty, 
	       penalty_type=penalty_type, dropouts=dropouts, 
	       learning_rate=learning_rate, momentum=momentum, optimizer=optimizer, 
	       batch_size=batch_size, n_classes=n_classes, verbose=verbose, seed=seed, 
	       **kwargs)

  def fit(self, dataset, nb_epoch=10, pad_batches=False, 
          max_checkpoints_to_keep=5, log_every_N_batches=50, **kwargs):
    """Fit the model.

    Parameters
    ---------- 
    dataset: dc.data.Dataset
      Dataset object holding training data 
    nb_epoch: 10
      Number of training epochs.
    pad_batches: bool
      Whether or not to pad each batch to exactly be of size batch_size.
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
      train_op = self.get_training_op(
          self.train_graph.graph, self.train_graph.loss)
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
              dataset.iterbatches(self.batch_size, pad_batches=pad_batches)):
            if ind % log_every_N_batches == 0:
              log("On batch %d" % ind, self.verbose)
            # Run training op.
            feed_dict = self.construct_feed_dict(X_b, y_b, w_b, ids_b)
            fetches = self.train_graph.output + [
                train_op, self.train_graph.loss]
            fetched_values = sess.run(fetches, feed_dict=feed_dict)
            output = fetched_values[:len(self.train_graph.output)]
            loss = fetched_values[-1]
            avg_loss += loss
            y_pred = np.squeeze(np.array(output))
            y_b = y_b.flatten()
            n_batches += 1
          saver.save(sess, self._save_path, global_step=epoch)
          avg_loss = float(avg_loss)/n_batches
          log('Ending epoch %d: Average loss %g' % (epoch, avg_loss), self.verbose)
        # Always save a final checkpoint when complete.
        saver.save(sess, self._save_path, global_step=epoch+1)
    ############################################################## TIMING
    time2 = time.time()
    print("TIMING: model fitting took %0.3f s" % (time2-time1),
          self.verbose)
    ############################################################## TIMING

  def predict_on_batch(self, X, pad_batch=False):
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
    if pad_batch:
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
          raise ValueError(
              'Unrecognized rank combination for output: %s' %
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
