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
__license__ = "GPL"

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
  >>> model = dc.models.Sequential()
  >>> # first layer must have a defined input shape
  >>> model.add(dc.nn.Dense(32, input_dim=500))
  >>> # afterwards, Keras does automatic shape inference
  >>> model.add(dc.nn.Dense(32))

  >>> # also possible (equivalent to the above):
  >>> model = dc.models.Sequential()
  >>> model.add(dc.nn.Dense(32, input_shape=(500,)))
  >>> model.add(dc.nn.Dense(32))

  >>> # also possible (equivalent to the above):
  >>> model = dc.models.Sequential()
  >>> # here the batch dimension is None,
  >>> # which means any batch size will be accepted by the model.
  >>> model.add(dc.nn.Dense(32, batch_input_shape=(None, 500)))
  >>> model.add(dc.nn.Dense(32))
  """

  def __init__(self, name=None):
    self.layers = []  # stack of layers
    self.outputs = []  # tensors (length 1)

    if not name:
      prefix = 'sequential_'
      name = prefix + str(model_ops.get_uid(prefix))
    self.name = name
    self.graph = tf.Graph()

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
        # first layer in model: check that it is an input layer
        if not isinstance(layer, InputLayer):
          raise ValueError("First layer in sequential model must be InputLayer")
        self.outputs = layer()

      else:
        self.outputs = layer(self.outputs[0])

      self.layers.append(layer)

  def add_loss(self, loss, inputs=None):
    """Adds a loss to model.
    
    Parameters
    ----------
    losses: list
    """
    # Add losses to graph
    with self.graph.as_default():
      self.loss = loss()

  @property
  def uses_learning_phase(self):
    return self.uses_learning_phase

  def fit(self, dataset, batch_size=32, nb_epoch=10, verbose=1,
          initial_epoch=0, **kwargs):
    """Trains the model for a fixed number of epochs.

    # Arguments
        x: input data, as a Numpy array or list of Numpy arrays
            (if the model has multiple inputs).
        y: labels, as a Numpy array.
        batch_size: integer. Number of samples per gradient update.
        nb_epoch: integer, the number of epochs to train the model.
        verbose: 0 for no logging to stdout,
            1 for progress bar logging, 2 for one log line per epoch.
        initial_epoch: epoch at which to start training
            (useful for resuming a previous training run)
    """
    pass
    #return self.model.fit(x, y,
    #                      batch_size=batch_size,
    #                      nb_epoch=nb_epoch,
    #                      verbose=verbose,
    #                      initial_epoch=initial_epoch)

  def evaluate(self, x, y, batch_size=32, verbose=1,
               sample_weight=None, **kwargs):
      """Computes the loss on some input data, batch by batch.

      # Arguments
          x: input data, as a Numpy array or list of Numpy arrays
              (if the model has multiple inputs).
          y: labels, as a Numpy array.
          batch_size: integer. Number of samples per gradient update.
          verbose: verbosity mode, 0 or 1.
          sample_weight: sample weights, as a Numpy array.

      # Returns
          Scalar test loss (if the model has no metrics)
          or list of scalars (if the model computes other metrics).
          The attribute `model.metrics_names` will give you
          the display labels for the scalar outputs.
      """
      if self.model is None:
          raise RuntimeError('The model needs to be compiled '
                             'before being used.')
      if 'show_accuracy' in kwargs:
          kwargs.pop('show_accuracy')
          warnings.warn('The "show_accuracy" argument is deprecated, '
                        'instead you should pass the "accuracy" metric to '
                        'the model at compile time:\n'
                        '`model.compile(optimizer, loss, '
                        'metrics=["accuracy"])`')
      if kwargs:
          raise TypeError('Received unknown keyword arguments: ' +
                          str(kwargs))
      return self.model.evaluate(x, y,
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

  def train_on_batch(self, x, y, class_weight=None,
                     sample_weight=None, **kwargs):
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
          raise RuntimeError('The model needs to be compiled '
                             'before being used.')
      if 'accuracy' in kwargs:
          kwargs.pop('accuracy')
          warnings.warn('The "accuracy" argument is deprecated, '
                        'instead you should pass the "accuracy" metric to '
                        'the model at compile time:\n'
                        '`model.compile(optimizer, loss, '
                        'metrics=["accuracy"])`')
      if kwargs:
          raise TypeError('Received unknown keyword arguments: ' +
                          str(kwargs))
      return self.model.train_on_batch(x, y,
                                       sample_weight=sample_weight,
                                       class_weight=class_weight)

  def test_on_batch(self, x, y,
                    sample_weight=None, **kwargs):
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
          raise RuntimeError('The model needs to be compiled '
                             'before being used.')
      if 'accuracy' in kwargs:
          kwargs.pop('accuracy')
          warnings.warn('The "accuracy" argument is deprecated, '
                        'instead you should pass the "accuracy" metric to '
                        'the model at compile time:\n'
                        '`model.compile(optimizer, loss, '
                        'metrics=["accuracy"])`')
      if kwargs:
          raise TypeError('Received unknown keyword arguments: ' +
                          str(kwargs))
      return self.model.test_on_batch(x, y,
                                      sample_weight=sample_weight)

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

  def predict_classes(self, x, batch_size=32, verbose=1):
      """Generate class predictions for the input samples
      batch by batch.

      # Arguments
          x: input data, as a Numpy array or list of Numpy arrays
              (if the model has multiple inputs).
          batch_size: integer.
          verbose: verbosity mode, 0 or 1.

      # Returns
          A numpy array of class predictions.
      """
      proba = self.predict(x, batch_size=batch_size, verbose=verbose)
      if proba.shape[-1] > 1:
          return proba.argmax(axis=-1)
      else:
          return (proba > 0.5).astype('int32')
