"""Implements SequenceDNNs for use in DRAGONN models.

Code adapated from github.com/kundajelab/dragonn
repository. The SequenceDNN class is useful for prediction
tasks working with genomic data.
"""
import tensorflow as tf
from deepchem.nn.regularizers import l1
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
  num_tasks : int, optional
      number of tasks. Default: 1.
  num_filters : list[int] | tuple[int]
      number of convolutional filters in each layer. Default: (15,).
  conv_width : list[int] | tuple[int]
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
               use_RNN=False,
               num_tasks=1,
               num_filters=15,
               kernel_size=15,
               pool_width=35,
               GRU_size=35,
               TDD_size=15,
               L1=0,
               dropout=0.0,
               verbose=True,
               **kwargs):
    super(SequenceDNN, self).__init__(**kwargs)
    self.num_tasks = num_tasks
    self.verbose = verbose
    assert len(num_filters) == len(conv_width)
    self.add(layers.Conv2D(num_filters, kernel_size=kernel_size))
    self.add(layers.Dropout(dropout))
    self.add(layers.MaxPool2D())
    #if use_RNN:
    #  num_max_pool_outputs = self.model.layers[-1].output_shape[-1]
    #  self.add(Reshape((num_filters[-1], num_max_pool_outputs)))
    #  self.add(Permute((2, 1)))
    #  self.add(GRU(GRU_size, return_sequences=True))
    #  self.add(TimeDistributedDense(TDD_size, activation='relu'))
    self.add(layers.Flatten())
    self.add(layers.Dense(self.num_tasks, activation_fn=tf.nn.relu))
    #self.add(Activation('sigmoid'))
