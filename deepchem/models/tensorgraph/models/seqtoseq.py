"""Sequence to sequence translation models."""

from deepchem.models import TensorGraph
from deepchem.models.tensorgraph import layers
from heapq import heappush, heappushpop
import numpy as np
import tensorflow as tf

class SeqToSeq(TensorGraph):
  """Implements sequence to sequence translation models."""
  
  sequence_end = object()
  
  def __init__(self, input_tokens, output_tokens, max_output_length, encoder_layers=4, decoder_layers=4, embedding_dimension=1024, dropout=0.0, **kwargs):
    super(SeqToSeq, self).__init__(use_queue=False, **kwargs) # TODO can we make it work with the queue?
    if SeqToSeq.sequence_end not in input_tokens:
      input_tokens = input_tokens+[SeqToSeq.sequence_end]
    if SeqToSeq.sequence_end not in output_tokens:
      output_tokens = output_tokens+[SeqToSeq.sequence_end]
    self._input_tokens = input_tokens
    self._output_tokens = output_tokens
    self._input_dict = dict((x,i) for i,x in enumerate(input_tokens))
    self._output_dict = dict((x,i) for i,x in enumerate(output_tokens))
    self._max_output_length = max_output_length
    self._features = layers.Feature(shape=(None, None, len(input_tokens)))
    self._labels = layers.Label(shape=(None, None, len(output_tokens)))
    self._gather_indices = layers.Feature(shape=(None, 2), dtype=tf.int32)
    prev_layer = self._features
    for i in range(encoder_layers):
      if dropout > 0.0:
        prev_layer = layers.Dropout(dropout, in_layers=prev_layer)
      prev_layer = layers.GRU(embedding_dimension, self.batch_size, in_layers=prev_layer)
    prev_layer = layers.Gather(in_layers=[prev_layer, self._gather_indices])
    self.embedding = prev_layer
    prev_layer = layers.Repeat(max_output_length, in_layers=prev_layer)
    for i in range(decoder_layers):
      if dropout > 0.0:
        prev_layer = layers.Dropout(dropout, in_layers=prev_layer)
      prev_layer = layers.GRU(embedding_dimension, self.batch_size, in_layers=prev_layer)
    output_layer = layers.Dense(len(output_tokens), in_layers=prev_layer, activation_fn=tf.nn.softmax)
    self.add_output(output_layer)
    prob = layers.ReduceSum(layers.Multiply([output_layer, self._labels]), axis=2)
    log_prob = layers.Log(layers.Add([prob, layers.Constant(np.finfo(np.float32).eps)]))
    objective = layers.ReduceMean(layers.ReduceSum(log_prob, axis=1))
    loss = layers.Multiply([objective, layers.Constant(-1)])
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
    self.fit_generator(self._generate_batches(generator), max_checkpoints_to_keep=max_checkpoints_to_keep,
                       checkpoint_interval=checkpoint_interval, restore=restore)

  def predict_from_sequence(self, sequence, beam_width=10):
    with self._get_tf("Graph").as_default():
      feed_dict = {}
      feed_dict[self._features] = self._create_input_array([sequence])
      feed_dict[self._gather_indices] = [(i, len(sequence)) for i in range(self.batch_size)]
      for initial, zero in zip(self.rnn_initial_states, self.rnn_zero_states):
        feed_dict[initial] = zero
      probs = self.session.run(self.output, feed_dict=feed_dict)[0]
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
              new_logprob = logprob+c[1]
              newc = (new_logprob/(len(c[2])+1), new_logprob, c[2]+[self._output_tokens[j]])
              if len(new_candidates) < beam_width:
                heappush(new_candidates, newc)
              else:
                heappushpop(new_candidates, newc)
        candidates = new_candidates
      return sorted(candidates)[-1][2][:-1]

  def _create_input_array(self, sequences):
    lengths = [len(x) for x in sequences]
    features = np.zeros((self.batch_size, max(lengths)+1, len(self._input_tokens)), dtype=np.float32)
    for i, sequence in enumerate(sequences):
      for j, token in enumerate(sequence):
        features[i, j, self._input_dict[token]] = 1
    features[np.arange(len(sequences)), lengths, self._input_dict[SeqToSeq.sequence_end]] = 1
    return features

  def _create_output_array(self, sequences):
    lengths = [len(x) for x in sequences]
    labels = np.zeros((self.batch_size, self._max_output_length, len(self._output_tokens)), dtype=np.float32)
    for i, sequence in enumerate(sequences):
      for j, token in enumerate(sequence):
        labels[i, j, self._output_dict[token]] = 1
    labels[np.arange(len(sequences)), lengths, self._output_dict[SeqToSeq.sequence_end]] = 1
    return labels

  def _generate_batches(self, generator):
    while True:
      inputs = []
      outputs = []
      weights = [1]*self.batch_size
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
      feed_dict[self._gather_indices] = [(i, len(x)) for i, x in enumerate(inputs)]
      for initial, zero in zip(self.rnn_initial_states, self.rnn_zero_states):
        feed_dict[initial] = zero
      yield feed_dict
