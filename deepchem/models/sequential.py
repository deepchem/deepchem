"""
Contains Sequential model adapted from keras/keras/models.py.

This class is adapted from Keras directly. Have cut out functionality
and changed API to match DeepChem style.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2017, Stanford University"
__license__ = "MIT"

import time
import os
import tempfile
import numpy as np
import tensorflow as tf
from deepchem.models.models import Model
from deepchem.nn import model_ops
from deepchem.nn.copy import Layer
from deepchem.nn.copy import InputLayer


class Sequential(Model):
  """Linear stack of layers.

  Parameters
  ----------
  layers: list of layers to add to the model.

  Note
  ----
  The first layer passed to a Sequential model
  should have a defined input shape. What that
  means is that it should have received an `input_shape`
  or `batch_input_shape` argument,
  or for some type of layers (recurrent, Dense...)
  an `input_dim` argument.

  Example
  -------
  >>> import deepchem as dc
  >>> model = dc.models.Sequential()
  >>> # Add features
  >>> model.add_features(dc.nn.Input(shape=(50,)))
  >>> # Add labels
  >>> model.add_labels(dc.nn.Input(shape=(1,)))
  >>> model.add(dc.nn.Dense(32, 50))
  >>> model.add(dc.nn.Dense(64, 32))
  """

  def __init__(self, name=None, logdir=None):
    super(Sequential, self).__init__(self, model_dir=logdir)
    self.layers = []  # stack of layers
    self.outputs = None  # tensors (length 1)

    if not name:
      prefix = 'sequential_'
      name = prefix + str(model_ops.get_uid(prefix))
    self.name = name
    self.graph = tf.Graph()

    config = tf.ConfigProto(allow_soft_placement=True)
    self.session = tf.Session(graph=self.graph, config=config)
    # Path to save checkpoint files
    self._save_path = os.path.join(self.model_dir, 'model.ckpt')

  def add(self, layer):
    """Adds a layer instance on top of the layer stack.

    Parameters
    ----------
    layer: layer instance.
    """
    if not isinstance(layer, Layer):
      raise TypeError("The added layer must be an instance of class Layer. "
                      "Found: " + str(layer))
    with self.graph.as_default():
      if not self.layers:
        raise ValueError("Call add_features() before calling add()")
        # first layer in model: check that it is an input layer

      else:
        self.outputs = layer(self.outputs)

      self.layers.append(layer)

  def add_features(self, layer):
    """Adds an input layer."""
    if self.layers:
      raise ValueError(
          "add_features() has to be called before layers are added.")
    if not isinstance(layer, InputLayer):
      raise ValueError("First layer in sequential model must be InputLayer")
    with self.graph.as_default():
      self.features = layer()[0]
      self.outputs = self.features
      self.layers = [layer]

  def add_labels(self, layer):
    """Adds a layer for labels"""
    with self.graph.as_default():
      self.labels = layer()[0]

  def add_loss(self, loss, inputs=None):
    """Adds a loss to model.

    Parameters
    ----------
    losses: list
    """
    # Add losses to graph
    with self.graph.as_default():
      # Loss for each batch element
      batch_loss = loss(self.outputs, self.labels)
      # Loss should be a float
      self.loss = tf.reduce_sum(batch_loss)

  @property
  def uses_learning_phase(self):
    return self.uses_learning_phase

  def fit(self,
          dataset,
          nb_epoch=10,
          max_checkpoints_to_keep=5,
          log_every_N_batches=50,
          learning_rate=.001,
          batch_size=50,
          checkpoint_interval=10):
    """Trains the model for a fixed number of epochs.

    TODO(rbharath0: This is mostly copied from TensorflowGraphModel. Should
    eventually refactor both together.

    Parameters
    ----------
    dataset: dc.data.Dataset
    nb_epoch: 10
      Number of training epochs.
      Dataset object holding training data
        batch_size: integer. Number of samples per gradient update.
        nb_epoch: integer, the number of epochs to train the model.
        verbose: 0 for no logging to stdout,
            1 for progress bar logging, 2 for one log line per epoch.
        initial_epoch: epoch at which to start training
            (useful for resuming a previous training run)
    checkpoint_interval: int
      Frequency at which to write checkpoints, measured in epochs
    """
    ############################################################## TIMING
    time1 = time.time()
    ############################################################## TIMING
    print("Training for %d epochs" % nb_epoch)
    with self.graph.as_default():
      opt = model_ops.optimizer("adam", learning_rate)
      train_op = opt.minimize(self.loss, name='train')
      with self.session as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=max_checkpoints_to_keep)
        # Save an initial checkpoint.
        saver.save(sess, self._save_path, global_step=0)
        for epoch in range(nb_epoch):
          avg_loss, n_batches = 0., 0
          # TODO(rbharath): Don't support example weighting yet.
          for ind, (X_b, y_b, w_b,
                    ids_b) in enumerate(dataset.iterbatches(batch_size)):
            if ind % log_every_N_batches == 0:
              print("On batch %d" % ind)
            feed_dict = {self.features: X_b, self.labels: y_b}
            fetches = [self.outputs] + [train_op, self.loss]
            fetched_values = sess.run(fetches, feed_dict=feed_dict)
            output = fetched_values[:1]
            loss = fetched_values[-1]
            avg_loss += loss
            y_pred = np.squeeze(np.array(output))
            y_b = y_b.flatten()
            n_batches += 1
          if epoch % checkpoint_interval == checkpoint_interval - 1:
            saver.save(sess, self._save_path, global_step=epoch)
          avg_loss = float(avg_loss) / n_batches
          print('Ending epoch %d: Average loss %g' % (epoch, avg_loss))
        # Always save a final checkpoint when complete.
        saver.save(sess, self._save_path, global_step=epoch + 1)
    ############################################################## TIMING
    time2 = time.time()
    print("TIMING: model fitting took %0.3f s" % (time2 - time1))
    ############################################################## TIMING

  def evaluate(self,
               x,
               y,
               batch_size=32,
               verbose=1,
               sample_weight=None,
               **kwargs):
    """Computes the loss on some input data, batch by batch.

    Parameters
    ----------
    x: input data, as a Numpy array or list of Numpy arrays
        (if the model has multiple inputs).
    y: labels, as a Numpy array.
    batch_size: integer. Number of samples per gradient update.
    verbose: verbosity mode, 0 or 1.
    sample_weight: sample weights, as a Numpy array.

    Returns
    -------
    Scalar test loss (if the model has no metrics)
    or list of scalars (if the model computes other metrics).
    The attribute `model.metrics_names` will give you
    the display labels for the scalar outputs.
    """
    if self.model is None:
      raise RuntimeError('The model needs to be compiled ' 'before being used.')
    if 'show_accuracy' in kwargs:
      kwargs.pop('show_accuracy')
      warnings.warn('The "show_accuracy" argument is deprecated, '
                    'instead you should pass the "accuracy" metric to '
                    'the model at compile time:\n'
                    '`model.compile(optimizer, loss, '
                    'metrics=["accuracy"])`')
    if kwargs:
      raise TypeError('Received unknown keyword arguments: ' + str(kwargs))
    return self.model.evaluate(
        x,
        y,
        batch_size=batch_size,
        verbose=verbose,
        sample_weight=sample_weight)

  def predict(self, x, batch_size=32, verbose=0):
    """Generates output predictions for the input samples,
      processing the samples in a batched way.

      # Arguments
          x: the input data, as a Numpy array.
          batch_size: integer.
          verbose: verbosity mode, 0 or 1.

      # Returns
          A Numpy array of predictions.
      """
    if self.model is None:
      self.build()
    return self.model.predict(x, batch_size=batch_size, verbose=verbose)

  def predict_on_batch(self, x):
    """Returns predictions for a single batch of samples.
      """
    if self.model is None:
      self.build()
    return self.model.predict_on_batch(x)

  def train_on_batch(self,
                     x,
                     y,
                     class_weight=None,
                     sample_weight=None,
                     **kwargs):
    """Single gradient update over one batch of samples.

      # Arguments
          x: input data, as a Numpy array or list of Numpy arrays
              (if the model has multiple inputs).
          y: labels, as a Numpy array.
          class_weight: dictionary mapping classes to a weight value,
              used for scaling the loss function (during training only).
          sample_weight: sample weights, as a Numpy array.

      # Returns
          Scalar training loss (if the model has no metrics)
          or list of scalars (if the model computes other metrics).
          The attribute `model.metrics_names` will give you
          the display labels for the scalar outputs.
      """
    if self.model is None:
      raise RuntimeError('The model needs to be compiled ' 'before being used.')
    if 'accuracy' in kwargs:
      kwargs.pop('accuracy')
      warnings.warn('The "accuracy" argument is deprecated, '
                    'instead you should pass the "accuracy" metric to '
                    'the model at compile time:\n'
                    '`model.compile(optimizer, loss, '
                    'metrics=["accuracy"])`')
    if kwargs:
      raise TypeError('Received unknown keyword arguments: ' + str(kwargs))
    return self.model.train_on_batch(
        x, y, sample_weight=sample_weight, class_weight=class_weight)

  def test_on_batch(self, x, y, sample_weight=None, **kwargs):
    """Evaluates the model over a single batch of samples.

      # Arguments
          x: input data, as a Numpy array or list of Numpy arrays
              (if the model has multiple inputs).
          y: labels, as a Numpy array.
          sample_weight: sample weights, as a Numpy array.

      # Returns
          Scalar test loss (if the model has no metrics)
          or list of scalars (if the model computes other metrics).
          The attribute `model.metrics_names` will give you
          the display labels for the scalar outputs.
      """
    if self.model is None:
      raise RuntimeError('The model needs to be compiled ' 'before being used.')
    if 'accuracy' in kwargs:
      kwargs.pop('accuracy')
      warnings.warn('The "accuracy" argument is deprecated, '
                    'instead you should pass the "accuracy" metric to '
                    'the model at compile time:\n'
                    '`model.compile(optimizer, loss, '
                    'metrics=["accuracy"])`')
    if kwargs:
      raise TypeError('Received unknown keyword arguments: ' + str(kwargs))
    return self.model.test_on_batch(x, y, sample_weight=sample_weight)

  def predict_proba(self, x, batch_size=32, verbose=1):
    """Generates class probability predictions for the input samples
      batch by batch.

      # Arguments
          x: input data, as a Numpy array or list of Numpy arrays
              (if the model has multiple inputs).
          batch_size: integer.
          verbose: verbosity mode, 0 or 1.

      # Returns
          A Numpy array of probability predictions.
      """
    preds = self.predict(x, batch_size, verbose)
    if preds.min() < 0. or preds.max() > 1.:
      warnings.warn('Network returning invalid probability values. '
                    'The last layer might not normalize predictions '
                    'into probabilities '
                    '(like softmax or sigmoid would).')
    return preds
