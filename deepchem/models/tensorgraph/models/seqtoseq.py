"""Sequence to sequence translation models."""

from deepchem.models import TensorGraph
from deepchem.models.tensorgraph import layers
from heapq import heappush, heappushpop
import numpy as np
import tensorflow as tf


class SeqToSeq(TensorGraph):
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
    """
    super(SeqToSeq, self).__init__(
        use_queue=False, **kwargs)  # TODO can we make it work with the queue?
    if SeqToSeq.sequence_end not in input_tokens:
      input_tokens = input_tokens + [SeqToSeq.sequence_end]
    if SeqToSeq.sequence_end not in output_tokens:
      output_tokens = output_tokens + [SeqToSeq.sequence_end]
    self._input_tokens = input_tokens
    self._output_tokens = output_tokens
    self._input_dict = dict((x, i) for i, x in enumerate(input_tokens))
    self._output_dict = dict((x, i) for i, x in enumerate(output_tokens))
    self._max_output_length = max_output_length
    self._features = layers.Feature(shape=(None, None, len(input_tokens)))
    self._labels = layers.Label(shape=(None, None, len(output_tokens)))
    self._gather_indices = layers.Feature(shape=(None, 2), dtype=tf.int32)
    self._reverse_input = reverse_input
    prev_layer = self._features
    for i in range(encoder_layers):
      if dropout > 0.0:
        prev_layer = layers.Dropout(dropout, in_layers=prev_layer)
      prev_layer = layers.GRU(
          embedding_dimension, self.batch_size, in_layers=prev_layer)
    prev_layer = layers.Gather(in_layers=[prev_layer, self._gather_indices])
    self.embedding = prev_layer
    prev_layer = layers.Repeat(max_output_length, in_layers=prev_layer)
    for i in range(decoder_layers):
      if dropout > 0.0:
        prev_layer = layers.Dropout(dropout, in_layers=prev_layer)
      prev_layer = layers.GRU(
          embedding_dimension, self.batch_size, in_layers=prev_layer)
    output_layer = layers.Dense(
        len(output_tokens), in_layers=prev_layer, activation_fn=tf.nn.softmax)
    self.add_output(output_layer)
    prob = layers.ReduceSum(output_layer * self._labels, axis=2)
    log_prob = layers.Log(prob + np.finfo(np.float32).eps)
    loss = -layers.ReduceMean(layers.ReduceSum(log_prob, axis=1))
    self.set_loss(loss)
    self.output = output_layer

  def fit_sequences(self,
                    generator,
                    max_checkpoints_to_keep=5,
                    checkpoint_interval=1000,
                    restore=False):
    """Train this model on a set of sequences

    Parameters
    ----------
    generator: generator
      this should generate a series of training samples.  Each sample should be
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
        self._generate_batches(generator),
        max_checkpoints_to_keep=max_checkpoints_to_keep,
        checkpoint_interval=checkpoint_interval,
        restore=restore)

  def predict_from_sequence(self, sequence, beam_width=5):
    """Given an input sequence, predict the output sequence.

    The prediction is done using a beam search with length normalization.

    Parameters
    ----------
    sequence: sequence
      the input sequence to generate a prediction for
    beam_width: int
      the beam width to use for searching.  Set to 1 to use a simple greedy search.
    """
    feed_dict = {}
    feed_dict[self._features] = self._create_input_array([sequence])
    feed_dict[self._gather_indices] = [(i, len(sequence))
                                       for i in range(self.batch_size)]
    feed_dict[self._training_placeholder] = 0.0
    for initial, zero in zip(self.rnn_initial_states, self.rnn_zero_states):
      feed_dict[initial] = zero
    with self._get_tf("Graph").as_default():
      probs = self.session.run(self.output, feed_dict=feed_dict)[0]
    return self._beam_search(probs, beam_width)

  def predict_from_embedding(self, embedding, beam_width=5):
    """Given an embedding vector, predict the output sequence.

    The prediction is done using a beam search with length normalization.

    Parameters
    ----------
    embedding: array
      the embedding vector to generate a prediction for
    beam_width: int
      the beam width to use for searching.  Set to 1 to use a simple greedy search.
    """
    feed_dict = {}
    feed_dict[self.embedding] = embedding
    feed_dict[self._training_placeholder] = 0.0
    for initial, zero in zip(self.rnn_initial_states, self.rnn_zero_states):
      feed_dict[initial] = zero
    with self._get_tf("Graph").as_default():
      probs = self.session.run(self.output, feed_dict=feed_dict)[0]
    return self._beam_search(probs, beam_width)

  def predict_embedding(self, sequence):
    """Given an input sequence, compute the embedding vector.

    Parameters
    ----------
    sequence: sequence
      the input sequence to generate an embedding vector for
    """
    feed_dict = {}
    feed_dict[self._features] = self._create_input_array([sequence])
    feed_dict[self._gather_indices] = [(i, len(sequence))
                                       for i in range(self.batch_size)]
    feed_dict[self._training_placeholder] = 0.0
    for initial, zero in zip(self.rnn_initial_states, self.rnn_zero_states):
      feed_dict[initial] = zero
    with self._get_tf("Graph").as_default():
      return self.session.run(self.embedding, feed_dict=feed_dict)

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
      if lengths[i] < self._max_output_length:
        labels[i, lengths[i], end_marker_index] = 1
    return labels

  def _generate_batches(self, generator):
    """Combine sequences into batches."""
    while True:
      inputs = []
      outputs = []
      weights = [1] * self.batch_size
      try:
        while len(inputs) < self.batch_size:
          input, output = next(generator)
          inputs.append(input)
          outputs.append(output)
      except StopIteration:
        if len(inputs) == 0:
          return
        for i in range(len(inputs), self.batch_size):
          inputs.append([])
          outputs.append([])
          weights[i] = 0
      feed_dict = {}
      feed_dict[self._features] = self._create_input_array(inputs)
      feed_dict[self._labels] = self._create_output_array(outputs)
      feed_dict[self._gather_indices] = [(i, len(x))
                                         for i, x in enumerate(inputs)]
      for initial, zero in zip(self.rnn_initial_states, self.rnn_zero_states):
        feed_dict[initial] = zero
      yield feed_dict
