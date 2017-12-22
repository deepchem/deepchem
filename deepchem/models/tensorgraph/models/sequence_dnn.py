"""Implements SequenceDNNs for use in DRAGONN models.

Code adapated from github.com/kundajelab/dragonn
repository. The SequenceDNN class is useful for prediction
tasks working with genomic data.
"""
from deepchem.models import Sequential
from deepchem.models.tensorgraph import layers

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
               num_filters=(15, 15, 15),
               conv_width=(15, 15, 15),
               pool_width=35,
               GRU_size=35,
               TDD_size=15,
               L1=0,
               dropout=0.0,
               verbose=True):
    self.num_tasks = num_tasks
    self.verbose = verbose
    assert len(num_filters) == len(conv_width)
    for i, (nb_filter, nb_col) in enumerate(zip(num_filters, conv_width)):
      conv_height = 4 if i == 0 else 1
      self.add(
          layers.Conv2D(
            nb_filter=nb_filter,
            nb_row=conv_height,
            nb_col=nb_col,
            activation='linear',
            init='he_normal',
            input_shape=(1, 4, seq_length),
            W_regularizer=l1(L1),
            b_regularizer=l1(L1)))
      self.add(Activation('relu'))
      self.add(Dropout(dropout))
      self.add(MaxPooling2D(pool_size=(1, pool_width)))
      if use_RNN:
        num_max_pool_outputs = self.model.layers[-1].output_shape[-1]
        self.add(Reshape((num_filters[-1], num_max_pool_outputs)))
        self.add(Permute((2, 1)))
        self.add(GRU(GRU_size, return_sequences=True))
        self.add(TimeDistributedDense(TDD_size, activation='relu'))
      self.add(Flatten())
      self.add(Dense(output_dim=self.num_tasks))
      self.add(Activation('sigmoid'))
      self.compile(optimizer='adam', loss='binary_crossentropy')
    else:
      raise ValueError(
          "Exactly one of seq_length or keras_model must be specified!")

  def train(self,
            X,
            y,
            validation_data,
            early_stopping_metric='Loss',
            early_stopping_patience=5,
            save_best_model_to_prefix=None):
    if y.dtype != bool:
      assert set(np.unique(y)) == {0, 1}
      y = y.astype(bool)
    multitask = y.shape[1] > 1
    if not multitask:
      num_positives = y.sum()
      num_sequences = len(y)
      num_negatives = num_sequences - num_positives
    if self.verbose:
      print('Training model (* indicates new best result)...')
    X_valid, y_valid = validation_data
    early_stopping_wait = 0
    best_metric = np.inf if early_stopping_metric == 'Loss' else -np.inf
    for epoch in range(1, self.num_epochs + 1):
      self.model.fit(
          X,
          y,
          batch_size=128,
          nb_epoch=1,
          class_weight={
              True: num_sequences / num_positives,
              False: num_sequences / num_negatives
          } if not multitask else None,
          verbose=self.verbose)
      if self.verbose:
        print('Epoch {}:'.format(epoch))
      if self.verbose:
        print()
    if self.verbose:
      print('Finished training after {} epochs.'.format(epoch))
      if save_best_model_to_prefix is not None:
        print("The best model's architecture and weights (from epoch {0}) "
              'were saved to {1}.arch.json and {1}.weights.h5'.format(
                  best_epoch, save_best_model_to_prefix))
