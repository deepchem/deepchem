from __future__ import absolute_import, division, print_function
import matplotlib
import numpy as np
import os
import subprocess
import sys
import tempfile
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from dragonn.metrics import ClassificationResult
from keras.layers.core import (Activation, Dense, Dropout, Flatten, Permute,
                                Reshape, TimeDistributedDense)
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.recurrent import GRU
from keras.regularizers import l1
from keras.layers.core import (Activation, Dense, Flatten,
                                TimeDistributedDense)
from keras.layers.recurrent import GRU
from keras.callbacks import EarlyStopping


#class SequenceDNN(Model):
#  """
#  Sequence DNN models.
#
#  Parameters
#  ----------
#  seq_length : int, optional
#      length of input sequence.
#  keras_model : instance of keras.models.Sequential, optional
#      seq_length or keras_model must be specified.
#  num_tasks : int, optional
#      number of tasks. Default: 1.
#  num_filters : list[int] | tuple[int]
#      number of convolutional filters in each layer. Default: (15,).
#  conv_width : list[int] | tuple[int]
#      width of each layer's convolutional filters. Default: (15,).
#  pool_width : int
#      width of max pooling after the last layer. Default: 35.
#  L1 : float
#      strength of L1 penalty.
#  dropout : float
#      dropout probability in every convolutional layer. Default: 0.
#  verbose: int
#      Verbosity level during training. Valida values: 0, 1, 2.
#
#  Returns
#  -------
#  Compiled DNN model.
#  """
#
#  def __init__(self,
#               seq_length=None,
#               keras_model=None,
#               use_RNN=False,
#               num_tasks=1,
#               num_filters=(15, 15, 15),
#               conv_width=(15, 15, 15),
#               pool_width=35,
#               GRU_size=35,
#               TDD_size=15,
#               L1=0,
#               dropout=0.0,
#               num_epochs=100,
#               verbose=1):
#    self.num_tasks = num_tasks
#    self.num_epochs = num_epochs
#    self.verbose = verbose
#    self.train_metrics = []
#    self.valid_metrics = []
#    if keras_model is not None and seq_length is None:
#      self.model = keras_model
#      self.num_tasks = keras_model.layers[-1].output_shape[-1]
#    elif seq_length is not None and keras_model is None:
#      self.model = Sequential()
#      assert len(num_filters) == len(conv_width)
#      for i, (nb_filter, nb_col) in enumerate(zip(num_filters, conv_width)):
#        conv_height = 4 if i == 0 else 1
#        self.model.add(
#            Convolution2D(
#                nb_filter=nb_filter,
#                nb_row=conv_height,
#                nb_col=nb_col,
#                activation='linear',
#                init='he_normal',
#                input_shape=(1, 4, seq_length),
#                W_regularizer=l1(L1),
#                b_regularizer=l1(L1)))
#        self.model.add(Activation('relu'))
#        self.model.add(Dropout(dropout))
#      self.model.add(MaxPooling2D(pool_size=(1, pool_width)))
#      if use_RNN:
#        num_max_pool_outputs = self.model.layers[-1].output_shape[-1]
#        self.model.add(Reshape((num_filters[-1], num_max_pool_outputs)))
#        self.model.add(Permute((2, 1)))
#        self.model.add(GRU(GRU_size, return_sequences=True))
#        self.model.add(TimeDistributedDense(TDD_size, activation='relu'))
#      self.model.add(Flatten())
#      self.model.add(Dense(output_dim=self.num_tasks))
#      self.model.add(Activation('sigmoid'))
#      self.model.compile(optimizer='adam', loss='binary_crossentropy')
#    else:
#      raise ValueError(
#          "Exactly one of seq_length or keras_model must be specified!")
#
#  def train(self,
#            X,
#            y,
#            validation_data,
#            early_stopping_metric='Loss',
#            early_stopping_patience=5,
#            save_best_model_to_prefix=None):
#    if y.dtype != bool:
#      assert set(np.unique(y)) == {0, 1}
#      y = y.astype(bool)
#    multitask = y.shape[1] > 1
#    if not multitask:
#      num_positives = y.sum()
#      num_sequences = len(y)
#      num_negatives = num_sequences - num_positives
#    if self.verbose >= 1:
#      print('Training model (* indicates new best result)...')
#    X_valid, y_valid = validation_data
#    early_stopping_wait = 0
#    best_metric = np.inf if early_stopping_metric == 'Loss' else -np.inf
#    for epoch in range(1, self.num_epochs + 1):
#      self.model.fit(
#          X,
#          y,
#          batch_size=128,
#          nb_epoch=1,
#          class_weight={
#              True: num_sequences / num_positives,
#              False: num_sequences / num_negatives
#          } if not multitask else None,
#          verbose=self.verbose >= 2)
#      epoch_train_metrics = self.test(X, y)
#      epoch_valid_metrics = self.test(X_valid, y_valid)
#      self.train_metrics.append(epoch_train_metrics)
#      self.valid_metrics.append(epoch_valid_metrics)
#      if self.verbose >= 1:
#        print('Epoch {}:'.format(epoch))
#        print('Train {}'.format(epoch_train_metrics))
#        print('Valid {}'.format(epoch_valid_metrics), end='')
#      current_metric = epoch_valid_metrics[early_stopping_metric].mean()
#      if (early_stopping_metric == 'Loss') == (current_metric <= best_metric):
#        if self.verbose >= 1:
#          print(' *')
#        best_metric = current_metric
#        best_epoch = epoch
#        early_stopping_wait = 0
#        if save_best_model_to_prefix is not None:
#          self.save(save_best_model_to_prefix)
#      else:
#        if self.verbose >= 1:
#          print()
#        if early_stopping_wait >= early_stopping_patience:
#          break
#        early_stopping_wait += 1
#    if self.verbose >= 1:
#      print('Finished training after {} epochs.'.format(epoch))
#      if save_best_model_to_prefix is not None:
#        print("The best model's architecture and weights (from epoch {0}) "
#              'were saved to {1}.arch.json and {1}.weights.h5'.format(
#                  best_epoch, save_best_model_to_prefix))
#
#  def predict(self, X):
#    return self.model.predict(X, batch_size=128, verbose=False)
#
#  def get_sequence_filters(self):
#    """
#    Returns 3D array of 2D sequence filters.
#    """
#    return self.model.layers[0].get_weights()[0].squeeze(axis=1)
#
#  def deeplift(self, X, batch_size=200):
#    """
#    Returns (num_task, num_samples, 1, num_bases, sequence_length) deeplift score array.
#    """
#    assert len(np.shape(X)) == 4 and np.shape(X)[1] == 1
#    from deeplift.conversion import keras_conversion as kc
#
#    # convert to deeplift model and get scoring function
#    deeplift_model = kc.convert_sequential_model(self.model, verbose=False)
#    score_func = deeplift_model.get_target_contribs_func(
#        find_scores_layer_idx=0)
#    # use a 40% GC reference
#    input_references = [np.array([0.3, 0.2, 0.2, 0.3])[None, None, :, None]]
#    # get deeplift scores
#    deeplift_scores = np.zeros((self.num_tasks,) + X.shape)
#    for i in range(self.num_tasks):
#      deeplift_scores[i] = score_func(
#          task_idx=i,
#          input_data_list=[X],
#          batch_size=batch_size,
#          progress_update=None,
#          input_references_list=input_references)
#    return deeplift_scores
#
#  def in_silico_mutagenesis(self, X):
#    """
#    Returns (num_task, num_samples, 1, num_bases, sequence_length) ISM score array.
#    """
#    mutagenesis_scores = np.empty(X.shape + (self.num_tasks,), dtype=np.float32)
#    wild_type_predictions = self.predict(X)
#    wild_type_predictions = wild_type_predictions[:, np.newaxis, np.newaxis,
#                                                  np.newaxis]
#    for sequence_index, (sequence, wild_type_prediction) in enumerate(
#        zip(X, wild_type_predictions)):
#      mutated_sequences = np.repeat(
#          sequence[np.newaxis], np.prod(sequence.shape), axis=0)
#      # remove wild-type
#      arange = np.arange(len(mutated_sequences))
#      horizontal_cycle = np.tile(
#          np.arange(sequence.shape[-1]), sequence.shape[-2])
#      mutated_sequences[arange, :, :, horizontal_cycle] = 0
#      # add mutant
#      vertical_repeat = np.repeat(
#          np.arange(sequence.shape[-2]), sequence.shape[-1])
#      mutated_sequences[arange, :, vertical_repeat, horizontal_cycle] = 1
#      # make mutant predictions
#      mutated_predictions = self.predict(mutated_sequences)
#      mutated_predictions = mutated_predictions.reshape(sequence.shape +
#                                                        (self.num_tasks,))
#      mutagenesis_scores[
#          sequence_index] = wild_type_prediction - mutated_predictions
#    return np.rollaxis(mutagenesis_scores, -1)
#
#  @staticmethod
#  def _plot_scores(X, output_directory, peak_width, score_func, score_name):
#    from dragonn.plot import plot_bases_on_ax
#    scores = score_func(X).squeeze(
#        axis=2)  # (num_task, num_samples, num_bases, sequence_length)
#    try:
#      os.makedirs(output_directory)
#    except OSError:
#      pass
#    num_tasks = len(scores)
#    for task_index, task_scores in enumerate(scores):
#      for sequence_index, sequence_scores in enumerate(task_scores):
#        # sequence_scores is num_bases x sequence_length
#        basewise_max_sequence_scores = sequence_scores.max(axis=0)
#        plt.clf()
#        figure, (top_axis, bottom_axis) = plt.subplots(2)
#        top_axis.plot(
#            range(1,
#                  len(basewise_max_sequence_scores) + 1),
#            basewise_max_sequence_scores)
#        top_axis.set_title('{} scores (motif highlighted)'.format(score_name))
#        peak_position = basewise_max_sequence_scores.argmax()
#        top_axis.axvspan(
#            peak_position - peak_width,
#            peak_position + peak_width,
#            color='grey',
#            alpha=0.1)
#        peak_sequence_scores = sequence_scores[:, peak_position - peak_width:
#                                               peak_position + peak_width].T
#        # Set non-max letter_heights to zero
#        letter_heights = np.zeros_like(peak_sequence_scores)
#        letter_heights[np.arange(len(letter_heights)),
#                       peak_sequence_scores.argmax(axis=1)] = \
#            basewise_max_sequence_scores[peak_position - peak_width :
#                                         peak_position + peak_width]
#        plot_bases_on_ax(letter_heights, bottom_axis)
#        bottom_axis.set_xticklabels(
#            tuple(
#                map(str,
#                    np.arange(peak_position - peak_width,
#                              peak_position + peak_width + 1))))
#        bottom_axis.tick_params(axis='x', labelsize='small')
#        plt.xlabel('Position')
#        plt.ylabel('Score')
#        plt.savefig(
#            os.path.join(output_directory, 'sequence_{}{}'.format(
#                sequence_index, '_task_{}'.format(task_index)
#                if num_tasks > 1 else '')))
#        plt.close()
#
#  def plot_deeplift(self, X, output_directory, peak_width=10):
#    self._plot_scores(
#        X,
#        output_directory,
#        peak_width,
#        score_func=self.deeplift,
#        score_name='DeepLift')
#
#  def plot_in_silico_mutagenesis(self, X, output_directory, peak_width=10):
#    self._plot_scores(
#        X,
#        output_directory,
#        peak_width,
#        score_func=self.in_silico_mutagenesis,
#        score_name='ISM')
#
#  def plot_architecture(self, output_file):
#    from dragonn.visualize_util import plot as plot_keras_model
#    plot_keras_model(self.model, output_file, show_shape=True)
#
#  def save(self, save_best_model_to_prefix):
#    arch_fname = save_best_model_to_prefix + '.arch.json'
#    weights_fname = save_best_model_to_prefix + '.weights.h5'
#    open(arch_fname, 'w').write(self.model.to_json())
#    self.model.save_weights(weights_fname, overwrite=True)
#
#  @staticmethod
#  def load(arch_fname, weights_fname=None):
#    model_json_string = open(arch_fname).read()
#    sequence_dnn = SequenceDNN(keras_model=model_from_json(model_json_string))
#    if weights_fname is not None:
#      sequence_dnn.model.load_weights(weights_fname)
#    return sequence_dnn


class MotifScoreRNN(Model):

  def __init__(self, input_shape, gru_size=10, tdd_size=4):
    self.model = Sequential()
    self.model.add(
        GRU(gru_size, return_sequences=True, input_shape=input_shape))
    if tdd_size is not None:
      self.model.add(TimeDistributedDense(tdd_size))
    self.model.add(Flatten())
    self.model.add(Dense(1))
    self.model.add(Activation('sigmoid'))
    print('Compiling model...')
    self.model.compile(optimizer='adam', loss='binary_crossentropy')

  def train(self, X, y, validation_data):
    print('Training model...')
    multitask = y.shape[1] > 1
    if not multitask:
      num_positives = y.sum()
      num_sequences = len(y)
      num_negatives = num_sequences - num_positives
    self.model.fit(
        X,
        y,
        batch_size=128,
        nb_epoch=100,
        validation_data=validation_data,
        class_weight={
            True: num_sequences / num_positives,
            False: num_sequences / num_negatives
        } if not multitask else None,
        callbacks=[EarlyStopping(monitor='val_loss', patience=10)],
        verbose=True)

  def predict(self, X):
    return self.model.predict(X, batch_size=128, verbose=False)


class gkmSVM(Model):

  def __init__(self,
               prefix='./gkmSVM',
               word_length=11,
               mismatches=3,
               C=1,
               threads=1,
               cache_memory=100,
               verbosity=4):
    self.word_length = word_length
    self.mismatches = mismatches
    self.C = C
    self.threads = threads
    self.prefix = '_'.join(map(str, (prefix, word_length, mismatches, C)))
    options_list = zip(
        ['-l', '-d', '-c', '-T', '-m', '-v'],
        map(str,
            (word_length, mismatches, C, threads, cache_memory, verbosity)))
    self.options = ' '.join([' '.join(option) for option in options_list])

  @property
  def model_file(self):
    model_fname = '{}.model.txt'.format(self.prefix)
    return model_fname if os.path.isfile(model_fname) else None

  @staticmethod
  def encode_sequence_into_fasta_file(sequence_iterator, ofname):
    """writes sequences into fasta file
        """
    with open(ofname, "w") as wf:
      for i, seq in enumerate(sequence_iterator):
        print('>{}'.format(i), file=wf)
        print(seq, file=wf)

  def train(self, X, y, validation_data=None):
    """
        Trains gkm-svm, saves model file.
        """
    y = y.squeeze()
    pos_sequence = X[y]
    neg_sequence = X[~y]
    pos_fname = "%s.pos_seq.fa" % self.prefix
    neg_fname = "%s.neg_seq.fa" % self.prefix
    # create temporary fasta files
    self.encode_sequence_into_fasta_file(pos_sequence, pos_fname)
    self.encode_sequence_into_fasta_file(neg_sequence, neg_fname)
    # run command
    command = ' '.join(('gkmtrain', self.options, pos_fname, neg_fname,
                        self.prefix))
    process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    process.wait()  # wait for it to finish
    # remove fasta files
    os.system("rm %s" % pos_fname)
    os.system("rm %s" % neg_fname)

  def predict(self, X):
    if self.model_file is None:
      raise RuntimeError("GkmSvm hasn't been trained!")
    # write test fasta file
    test_fname = "%s.test.fa" % self.prefix
    self.encode_sequence_into_fasta_file(X, test_fname)
    # test gkmsvm
    temp_ofp = tempfile.NamedTemporaryFile()
    threads_option = '-T %s' % (str(self.threads))
    command = ' '.join([
        'gkmpredict', test_fname, self.model_file, temp_ofp.name, threads_option
    ])
    process = subprocess.Popen(command, shell=True)
    process.wait()  # wait for it to finish
    os.system("rm %s" % test_fname)  # remove fasta file
    # get classification results
    temp_ofp.seek(0)
    y = np.array([line.split()[-1] for line in temp_ofp], dtype=float)
    temp_ofp.close()
    return np.expand_dims(y, 1)
