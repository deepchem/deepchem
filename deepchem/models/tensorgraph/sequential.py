"""
Convenience class for building sequential deep networks.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import warnings
import tensorflow as tf
from deepchem.models.tensorgraph.tensor_graph import TensorGraph
from deepchem.models.tensorgraph.layers import Feature
from deepchem.models.tensorgraph.layers import Label
from deepchem.models.tensorgraph.layers import SoftMaxCrossEntropy
from deepchem.models.tensorgraph.layers import ReduceMean
from deepchem.models.tensorgraph.layers import ReduceSquareDifference


class Sequential(TensorGraph):
  """Sequential models are linear stacks of layers.

  Analogous to the Sequential model from Keras and allows for less
  verbose construction of simple deep learning model.

  Example
  -------

  >>> import deepchem as dc
  >>> import numpy as np
  >>> from deepchem.models.tensorgraph import layers
  >>> # Define Data
  >>> X = np.random.rand(20, 2)                     
  >>> y = [[0, 1] for x in range(20)]
  >>> dataset = dc.data.NumpyDataset(X, y)                              
  >>> model = dc.models.Sequential(learning_rate=0.01)                  
  >>> model.add(layers.Dense(out_channels=2))                                  
  >>> model.add(layers.SoftMax())
  """

  def __init__(self, **kwargs):
    """Initializes a sequential model
    """
    self.num_layers = 0
    self._prev_layer = None
    if "use_queue" in kwargs:
      if kwargs["use_queue"]:
        raise ValueError("Sequential doesn't support queues.")
    kwargs["use_queue"] = False
    self._layer_list = []
    self._built = False
    super(Sequential, self).__init__(**kwargs)

  def add(self, layer):
    """Adds a new layer to model.

    Parameter
    ---------
    layer: Layer
      Adds layer to this graph.
    """
    self._layer_list.append(layer)

  def fit(self, dataset, loss, **kwargs):
    """Fits on the specified dataset.

    If called for the first time, constructs the TensorFlow graph for this
    model. Fits this graph on the specified dataset according to the specified
    loss.

    Parameters
    ----------
    dataset: dc.data.Dataset
      Dataset with data
    loss: string
      Only "binary_crossentropy" or "mse" for now.
    """
    X_shape, y_shape, _, _ = dataset.get_shape()
    # Calling fit() for first time
    if not self.built:
      feature_shape = X_shape[1:]
      label_shape = y_shape[1:]
      # Add in features
      features = Feature(shape=(None,) + feature_shape)
      # Add in labels
      labels = Label(shape=(None,) + label_shape)

      # Add in all layers
      prev_layer = features
      if len(self._layer_list) == 0:
        raise ValueError("No layers have been added to model.")
      for ind, layer in enumerate(self._layer_list):
        if len(layer.in_layers) > 1:
          raise ValueError("Cannot specify more than one "
                           "in_layer for Sequential.")
        layer.in_layers += [prev_layer]
        prev_layer = layer
      # The last layer is the output of the model
      self.outputs.append(prev_layer)

      if loss == "binary_crossentropy":
        smce = SoftMaxCrossEntropy(in_layers=[labels, prev_layer])
        self.set_loss(ReduceMean(in_layers=[smce]))
      elif loss == "mse":
        mse = ReduceSquareDifference(in_layers=[prev_layer, labels])
        self.set_loss(mse)
      else:
        # TODO(rbharath): Add in support for additional
        # losses.
        raise ValueError("Unsupported loss.")

    super(Sequential, self).fit(dataset, **kwargs)

  def restore(self, checkpoint=None):
    """Not currently supported.
    """
    # TODO(rbharath): The TensorGraph can't be built until
    # fit is called since the shapes of features/labels
    # not specified. Need to figure out a good restoration
    # method for this use case.
    raise ValueError("Restore is not yet supported " "for sequential models.")
