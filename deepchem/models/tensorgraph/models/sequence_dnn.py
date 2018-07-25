"""Implements SequenceDNNs for use in DRAGONN models.

Code adapated from github.com/kundajelab/dragonn
repository. The SequenceDNN class is useful for prediction
tasks working with genomic data.
"""
import tensorflow as tf
import collections
from deepchem.models import Sequential
from deepchem.models.tensorgraph import layers
from deepchem.data import NumpyDataset


class SequenceDNN(Sequential):
  """
  Sequence DNN models.

  Parameters
  ----------
  seq_length : int
      length of input sequence.
  loss: string
    the loss function to use.  Supported values are 'binary_crossentropy' and 'mse'.
  num_tasks : int, optional
      number of tasks. Default: 1.
  num_filters : list[int] | tuple[int]
      number of convolutional filters in each layer. Default: (15,).
  kernel_size: list[int] | tuple[int]
      width of each layer's convolutional filters. Default: (15,).
  pool_width : int
      width of max pooling after the last layer. Default: 35.
  L1 : float
      strength of L1 penalty.
  dropout : float
      dropout probability in every convolutional layer. Default: 0.
  verbose: bool
      Verbose print statements activated if true.
  """

  def __init__(self,
               seq_length,
               loss,
               use_RNN=False,
               num_tasks=1,
               num_filters=(15,),
               kernel_size=(15,),
               pool_width=35,
               L1=0,
               dropout=0.0,
               verbose=True,
               **kwargs):
    super(SequenceDNN, self).__init__(loss, **kwargs)
    if not isinstance(num_filters, collections.Sequence) or not isinstance(
        kernel_size, collections.Sequence):
      raise ValueError("num_filters and kernel_size must be lists")
    if not len(num_filters) == len(kernel_size):
      raise ValueError("num_filters and kernel_size must be of same length")
    n_layers = len(num_filters)
    self.num_tasks = num_tasks
    self.verbose = verbose
    for layer in range(n_layers):
      self.add(
          layers.Conv2D(num_filters[layer], kernel_size=kernel_size[layer]))
      self.add(layers.Dropout(dropout))
    self.add(layers.Flatten())
    self.add(layers.Dense(self.num_tasks, activation_fn=tf.nn.relu))
