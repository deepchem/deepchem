"""Sequence to sequence translation models."""

from deepchem.models import KerasModel, layers
from heapq import heappush, heappushpop
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Layer, Dense, Dropout, GRU, Lambda, Conv1D, Flatten, BatchNormalization


class VariationalRandomizer(Layer):
  """Add random noise to the embedding and include a corresponding loss."""

  def __init__(self, embedding_dimension, annealing_start_step,
               annealing_final_step, **kwargs):
    super(VariationalRandomizer, self).__init__(**kwargs)
    self._embedding_dimension = embedding_dimension
    self._annealing_final_step = annealing_final_step
    self._annealing_start_step = annealing_start_step
    self.dense_mean = Dense(embedding_dimension)
    self.dense_stddev = Dense(embedding_dimension)
    self.combine = layers.CombineMeanStd(training_only=True)

  def call(self, inputs, training=True):
    input, global_step = inputs
    embedding_mean = self.dense_mean(input)
    embedding_stddev = self.dense_stddev(input)
    embedding = self.combine(
        [embedding_mean, embedding_stddev], training=training)
    mean_sq = embedding_mean * embedding_mean
    stddev_sq = embedding_stddev * embedding_stddev
    kl = mean_sq + stddev_sq - tf.math.log(stddev_sq + 1e-20) - 1
    anneal_steps = self._annealing_final_step - self._annealing_start_step
    if anneal_steps > 0:
      current_step = tf.cast(global_step,
                             tf.float32) - self._annealing_start_step
      anneal_frac = tf.maximum(0.0, current_step) / anneal_steps
      kl_scale = tf.minimum(1.0, anneal_frac * anneal_frac)
    else:
      kl_scale = 1.0
    self.add_loss(0.5 * kl_scale * tf.reduce_mean(kl))
    return embedding


class SeqToSeq(KerasModel):
  """Implements sequence to sequence translation models.

  The model is based on the description in Sutskever et al., "Sequence to
  Sequence Learning with Neural Networks" (https://arxiv.org/abs/1409.3215),
  although this implementation uses GRUs instead of LSTMs.  The goal is to
  take sequences of tokens as input, and translate each one into a different
  output sequence.  The input and output sequences can both be of variable
  length, and an output sequence need not have the same length as the input
  sequence it was generated from.  For example, these models were originally
  developed for use in natural language processing.  In that context, the
  input might be a sequence of English words, and the output might be a
  sequence of French words.  The goal would be to train the model to translate
  sentences from English to French.

  The model consists of two parts called the "encoder" and "decoder".  Each one
  consists of a stack of recurrent layers.  The job of the encoder is to
  transform the input sequence into a single, fixed length vector called the
  "embedding".  That vector contains all relevant information from the input
  sequence.  The decoder then transforms the embedding vector into the output
  sequence.

  These models can be used for various purposes.  First and most obviously,
  they can be used for sequence to sequence translation.  In any case where you
  have sequences of tokens, and you want to translate each one into a different
  sequence, a SeqToSeq model can be trained to perform the translation.

  Another possible use case is transforming variable length sequences into
  fixed length vectors.  Many types of models require their inputs to have a
  fixed shape, which makes it difficult to use them with variable sized inputs
  (for example, when the input is a molecule, and different molecules have
  different numbers of atoms).  In that case, you can train a SeqToSeq model as
  an autoencoder, so that it tries to make the output sequence identical to the
  input one.  That forces the embedding vector to contain all information from
  the original sequence.  You can then use the encoder for transforming
  sequences into fixed length embedding vectors, suitable to use as inputs to
  other types of models.

  Another use case is to train the decoder for use as a generative model.  Here
  again you begin by training the SeqToSeq model as an autoencoder.  Once
  training is complete, you can supply arbitrary embedding vectors, and
  transform each one into an output sequence.  When used in this way, you
  typically train it as a variational autoencoder.  This adds random noise to
  the encoder, and also adds a constraint term to the loss that forces the
  embedding vector to have a unit Gaussian distribution.  You can then pick
  random vectors from a Gaussian distribution, and the output sequences should
  follow the same distribution as the training data.

  When training as a variational autoencoder, it is best to use KL cost
  annealing, as described in https://arxiv.org/abs/1511.06349.  The constraint
  term in the loss is initially set to 0, so the optimizer just tries to
  minimize the reconstruction loss.  Once it has made reasonable progress
  toward that, the constraint term can be gradually turned back on.  The range
  of steps over which this happens is configurable.
  """

  sequence_end = object()

  def __init__(self,
               input_tokens,
               output_tokens,
               max_output_length,
               encoder_layers=4,
               decoder_layers=4,
               embedding_dimension=512,
               dropout=0.0,
               reverse_input=True,
               variational=False,
               annealing_start_step=5000,
               annealing_final_step=10000,
               **kwargs):
    """Construct a SeqToSeq model.

    In addition to the following arguments, this class also accepts all the keyword arguments
    from TensorGraph.

    Parameters
    ----------
    input_tokens: list
      a list of all tokens that may appear in input sequences
    output_tokens: list
      a list of all tokens that may appear in output sequences
    max_output_length: int
      the maximum length of output sequence that may be generated
    encoder_layers: int
      the number of recurrent layers in the encoder
    decoder_layers: int
      the number of recurrent layers in the decoder
    embedding_dimension: int
      the width of the embedding vector.  This also is the width of all
      recurrent layers.
    dropout: float
      the dropout probability to use during training
    reverse_input: bool
      if True, reverse the order of input sequences before sending them into
      the encoder.  This can improve performance when working with long sequences.
    variational: bool
      if True, train the model as a variational autoencoder.  This adds random
      noise to the encoder, and also constrains the embedding to follow a unit
      Gaussian distribution.
    annealing_start_step: int
      the step (that is, batch) at which to begin turning on the constraint term
      for KL cost annealing
    annealing_final_step: int
      the step (that is, batch) at which to finish turning on the constraint term
      for KL cost annealing
    """
    if SeqToSeq.sequence_end not in input_tokens:
      input_tokens = input_tokens + [SeqToSeq.sequence_end]
    if SeqToSeq.sequence_end not in output_tokens:
      output_tokens = output_tokens + [SeqToSeq.sequence_end]
    self._input_tokens = input_tokens
    self._output_tokens = output_tokens
    self._input_dict = dict((x, i) for i, x in enumerate(input_tokens))
    self._output_dict = dict((x, i) for i, x in enumerate(output_tokens))
    self._max_output_length = max_output_length
    self._embedding_dimension = embedding_dimension
    self._reverse_input = reverse_input
    self.encoder = self._create_encoder(encoder_layers, dropout)
    self.decoder = self._create_decoder(decoder_layers, dropout)
    features = self._create_features()
    gather_indices = Input(shape=(2,), dtype=tf.int32)
    global_step = Input(shape=tuple(), dtype=tf.int32)
    embedding = self.encoder([features, gather_indices])
    self._embedding = self.encoder([features, gather_indices], training=False)
    if variational:
      randomizer = VariationalRandomizer(
          self._embedding_dimension, annealing_start_step, annealing_final_step)
      embedding = randomizer([self._embedding, global_step])
      self._embedding = randomizer(
          [self._embedding, global_step], training=False)
    output = self.decoder(embedding)
    model = tf.keras.Model(
        inputs=[features, gather_indices, global_step], outputs=output)
    super(SeqToSeq, self).__init__(model, self._create_loss(), **kwargs)

  def _create_features(self):
    return Input(shape=(None, len(self._input_tokens)))

  def _create_encoder(self, n_layers, dropout):
    """Create the encoder as a tf.keras.Model."""
    input = self._create_features()
    gather_indices = Input(shape=(2,), dtype=tf.int32)
    prev_layer = input
    for i in range(n_layers):
      if dropout > 0.0:
        prev_layer = Dropout(rate=dropout)(prev_layer)
      prev_layer = GRU(
          self._embedding_dimension, return_sequences=True)(prev_layer)
    prev_layer = Lambda(lambda x: tf.gather_nd(x[0], x[1]))(
        [prev_layer, gather_indices])
    return tf.keras.Model(inputs=[input, gather_indices], outputs=prev_layer)

  def _create_decoder(self, n_layers, dropout):
    """Create the decoder as a tf.keras.Model."""
    input = Input(shape=(self._embedding_dimension,))
    prev_layer = layers.Stack()(self._max_output_length * [input])
    for i in range(n_layers):
      if dropout > 0.0:
        prev_layer = Dropout(dropout)(prev_layer)
      prev_layer = GRU(
          self._embedding_dimension, return_sequences=True)(prev_layer)
    output = Dense(
        len(self._output_tokens), activation=tf.nn.softmax)(prev_layer)
    return tf.keras.Model(inputs=input, outputs=output)

  def _create_loss(self):
    """Create the loss function."""

    def loss_fn(outputs, labels, weights):
      prob = tf.reduce_sum(outputs[0] * labels[0], axis=2)
      mask = tf.reduce_sum(labels[0], axis=2)
      log_prob = tf.math.log(prob + 1e-20) * mask
      loss = -tf.reduce_mean(tf.reduce_sum(log_prob, axis=1))
      return loss + sum(self.model.losses)

    return loss_fn

  def fit_sequences(self,
                    sequences,
                    max_checkpoints_to_keep=5,
                    checkpoint_interval=1000,
                    restore=False):
    """Train this model on a set of sequences

    Parameters
    ----------
    sequences: iterable
      the training samples to fit to.  Each sample should be
      represented as a tuple of the form (input_sequence, output_sequence).
    max_checkpoints_to_keep: int
      the maximum number of checkpoints to keep.  Older checkpoints are discarded.
    checkpoint_interval: int
      the frequency at which to write checkpoints, measured in training steps.
    restore: bool
      if True, restore the model from the most recent checkpoint and continue training
      from there.  If False, retrain the model from scratch.
    """
    self.fit_generator(
        self._generate_batches(sequences),
        max_checkpoints_to_keep=max_checkpoints_to_keep,
        checkpoint_interval=checkpoint_interval,
        restore=restore)

  def predict_from_sequences(self, sequences, beam_width=5):
    """Given a set of input sequences, predict the output sequences.

    The prediction is done using a beam search with length normalization.

    Parameters
    ----------
    sequences: iterable
      the input sequences to generate a prediction for
    beam_width: int
      the beam width to use for searching.  Set to 1 to use a simple greedy search.
    """
    result = []
    for batch in self._batch_elements(sequences):
      features = self._create_input_array(batch)
      indices = np.array([(i, len(batch[i]) if i < len(batch) else 0)
                          for i in range(self.batch_size)])
      probs = self.predict_on_generator([[(features, indices,
                                           np.array(self.get_global_step())),
                                          None, None]])
      for i in range(len(batch)):
        result.append(self._beam_search(probs[i], beam_width))
    return result

  def predict_from_embeddings(self, embeddings, beam_width=5):
    """Given a set of embedding vectors, predict the output sequences.

    The prediction is done using a beam search with length normalization.

    Parameters
    ----------
    embeddings: iterable
      the embedding vectors to generate predictions for
    beam_width: int
      the beam width to use for searching.  Set to 1 to use a simple greedy search.
    """
    result = []
    for batch in self._batch_elements(embeddings):
      embedding_array = np.zeros(
          (self.batch_size, self._embedding_dimension), dtype=np.float32)
      for i, e in enumerate(batch):
        embedding_array[i] = e
      probs = self.decoder(embedding_array, training=False)
      probs = probs.numpy()
      for i in range(len(batch)):
        result.append(self._beam_search(probs[i], beam_width))
    return result

  def predict_embeddings(self, sequences):
    """Given a set of input sequences, compute the embedding vectors.

    Parameters
    ----------
    sequences: iterable
      the input sequences to generate an embedding vector for
    """
    result = []
    for batch in self._batch_elements(sequences):
      features = self._create_input_array(batch)
      indices = np.array([(i, len(batch[i]) if i < len(batch) else 0)
                          for i in range(self.batch_size)])
      embeddings = self.predict_on_generator(
          [[(features, indices, np.array(self.get_global_step())), None, None]],
          outputs=self._embedding)
      for i in range(len(batch)):
        result.append(embeddings[i])
    return np.array(result, dtype=np.float32)

  def _beam_search(self, probs, beam_width):
    """Perform a beam search for the most likely output sequence."""
    if beam_width == 1:
      # Do a simple greedy search.

      s = []
      for i in range(len(probs)):
        token = self._output_tokens[np.argmax(probs[i])]
        if token == SeqToSeq.sequence_end:
          break
        s.append(token)
      return s

    # Do a beam search with length normalization.

    logprobs = np.log(probs)
    # Represent each candidate as (normalized prob, raw prob, sequence)
    candidates = [(0.0, 0.0, [])]
    for i in range(len(logprobs)):
      new_candidates = []
      for c in candidates:
        if len(c[2]) > 0 and c[2][-1] == SeqToSeq.sequence_end:
          # This candidate sequence has already been terminated
          if len(new_candidates) < beam_width:
            heappush(new_candidates, c)
          else:
            heappushpop(new_candidates, c)
        else:
          # Consider all possible tokens we could add to this candidate sequence.
          for j, logprob in enumerate(logprobs[i]):
            new_logprob = logprob + c[1]
            newc = (new_logprob / (len(c[2]) + 1), new_logprob,
                    c[2] + [self._output_tokens[j]])
            if len(new_candidates) < beam_width:
              heappush(new_candidates, newc)
            else:
              heappushpop(new_candidates, newc)
      candidates = new_candidates
    return sorted(candidates)[-1][2][:-1]

  def _create_input_array(self, sequences):
    """Create the array describing the input sequences for a batch."""
    lengths = [len(x) for x in sequences]
    if self._reverse_input:
      sequences = [reversed(s) for s in sequences]
    features = np.zeros(
        (self.batch_size, max(lengths) + 1, len(self._input_tokens)),
        dtype=np.float32)
    for i, sequence in enumerate(sequences):
      for j, token in enumerate(sequence):
        features[i, j, self._input_dict[token]] = 1
    features[np.arange(len(sequences)), lengths, self._input_dict[
        SeqToSeq.sequence_end]] = 1
    return features

  def _create_output_array(self, sequences):
    """Create the array describing the target sequences for a batch."""
    lengths = [len(x) for x in sequences]
    labels = np.zeros(
        (self.batch_size, self._max_output_length, len(self._output_tokens)),
        dtype=np.float32)
    end_marker_index = self._output_dict[SeqToSeq.sequence_end]
    for i, sequence in enumerate(sequences):
      for j, token in enumerate(sequence):
        labels[i, j, self._output_dict[token]] = 1
      for j in range(lengths[i], self._max_output_length):
        labels[i, j, end_marker_index] = 1
    return labels

  def _batch_elements(self, elements):
    """Combine elements into batches."""
    batch = []
    for s in elements:
      batch.append(s)
      if len(batch) == self.batch_size:
        yield batch
        batch = []
    if len(batch) > 0:
      yield batch

  def _generate_batches(self, sequences):
    """Create feed_dicts for fitting."""
    for batch in self._batch_elements(sequences):
      inputs = []
      outputs = []
      for input, output in batch:
        inputs.append(input)
        outputs.append(output)
      for i in range(len(inputs), self.batch_size):
        inputs.append([])
        outputs.append([])
      features = self._create_input_array(inputs)
      labels = self._create_output_array(outputs)
      gather_indices = np.array([(i, len(x)) for i, x in enumerate(inputs)])
      yield ([features, gather_indices,
              np.array(self.get_global_step())], [labels], [])


class AspuruGuzikAutoEncoder(SeqToSeq):
  """
  This is an implementation of Automatic Chemical Design Using a Continuous Representation of Molecules
  http://pubs.acs.org/doi/full/10.1021/acscentsci.7b00572

  Abstract
  --------
  We report a method to convert discrete representations of molecules to and
  from a multidimensional continuous representation. This model allows us to
  generate new molecules for efficient exploration and optimization through
  open-ended spaces of chemical compounds. A deep neural network was trained on
  hundreds of thousands of existing chemical structures to construct three
  coupled functions: an encoder, a decoder, and a predictor. The encoder
  converts the discrete representation of a molecule into a real-valued
  continuous vector, and the decoder converts these continuous vectors back to
  discrete molecular representations. The predictor estimates chemical
  properties from the latent continuous vector representation of the molecule.
  Continuous representations of molecules allow us to automatically generate
  novel chemical structures by performing simple operations in the latent space,
  such as decoding random vectors, perturbing known chemical structures, or
  interpolating between molecules. Continuous representations also allow the use
  of powerful gradient-based optimization to efficiently guide the search for
  optimized functional compounds. We demonstrate our method in the domain of
  drug-like molecules and also in a set of molecules with fewer that nine heavy
  atoms.

  Notes
  -------
  This is currently an imperfect reproduction of the paper.  One difference is
  that teacher forcing in the decoder is not implemented.  The paper also
  discusses co-learning molecular properties at the same time as training the
  encoder/decoder.  This is not done here.  The hyperparameters chosen are from
  ZINC dataset.

  This network also currently suffers from exploding gradients.  Care has to be taken when training.

  NOTE(LESWING): Will need to play around with annealing schedule to not have exploding gradients
  TODO(LESWING): Teacher Forcing
  TODO(LESWING): Sigmoid variational loss annealing schedule
  The output GRU layer had one
  additional input, corresponding to the character sampled from the softmax output of the
  previous time step and was trained using teacher forcing. 48 This increased the accuracy
  of generated SMILES strings, which resulted in higher fractions of valid SMILES strings
  for latent points outside the training data, but also made training more difficult, since the
  decoder showed a tendency to ignore the (variational) encoding and rely solely on the input
  sequence. The variational loss was annealed according to sigmoid schedule after 29 epochs,
  running for a total 120 epochs

  I also added a BatchNorm before the mean and std embedding layers.  This has empiracally
  made training more stable, and is discussed in Ladder Variational Autoencoders.
  https://arxiv.org/pdf/1602.02282.pdf
  Maybe if Teacher Forcing and Sigmoid variational loss annealing schedule are used the
  BatchNorm will no longer be neccessary.
  """

  def __init__(self,
               num_tokens,
               max_output_length,
               embedding_dimension=196,
               filter_sizes=[9, 9, 10],
               kernel_sizes=[9, 9, 11],
               decoder_dimension=488,
               **kwargs):
    """
    Parameters
    ----------
    filter_sizes: list of int
      Number of filters for each 1D convolution in the encoder
    kernel_sizes: list of int
      Kernel size for each 1D convolution in the encoder
    decoder_dimension: int
      Number of channels for the GRU Decoder
    """
    if len(filter_sizes) != len(kernel_sizes):
      raise ValueError("Must have same number of layers and kernels")
    self._filter_sizes = filter_sizes
    self._kernel_sizes = kernel_sizes
    self._decoder_dimension = decoder_dimension
    super(AspuruGuzikAutoEncoder, self).__init__(
        input_tokens=num_tokens,
        output_tokens=num_tokens,
        max_output_length=max_output_length,
        embedding_dimension=embedding_dimension,
        variational=True,
        reverse_input=False,
        **kwargs)

  def _create_features(self):
    return Input(shape=(self._max_output_length, len(self._input_tokens)))

  def _create_encoder(self, n_layers, dropout):
    """Create the encoder as a tf.keras.Model."""
    input = self._create_features()
    gather_indices = Input(shape=(2,), dtype=tf.int32)
    prev_layer = input
    for i in range(len(self._filter_sizes)):
      filter_size = self._filter_sizes[i]
      kernel_size = self._kernel_sizes[i]
      if dropout > 0.0:
        prev_layer = Dropout(rate=dropout)(prev_layer)
      prev_layer = Conv1D(
          filters=filter_size, kernel_size=kernel_size,
          activation=tf.nn.relu)(prev_layer)
    prev_layer = Flatten()(prev_layer)
    prev_layer = Dense(
        self._decoder_dimension, activation=tf.nn.relu)(prev_layer)
    prev_layer = BatchNormalization()(prev_layer)
    return tf.keras.Model(inputs=[input, gather_indices], outputs=prev_layer)

  def _create_decoder(self, n_layers, dropout):
    """Create the decoder as a tf.keras.Model."""
    input = Input(shape=(self._embedding_dimension,))
    prev_layer = Dense(self._embedding_dimension, activation=tf.nn.relu)(input)
    prev_layer = layers.Stack()(self._max_output_length * [prev_layer])
    for i in range(3):
      if dropout > 0.0:
        prev_layer = Dropout(dropout)(prev_layer)
      prev_layer = GRU(
          self._decoder_dimension, return_sequences=True)(prev_layer)
    output = Dense(
        len(self._output_tokens), activation=tf.nn.softmax)(prev_layer)
    return tf.keras.Model(inputs=input, outputs=output)

  def _create_input_array(self, sequences):
    return self._create_output_array(sequences)
