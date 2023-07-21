"""Sequence to sequence translation models."""

from heapq import heappush, heappushpop
from deepchem.models.torch_models import layers
import torch
import torch.nn as nn

import numpy as np
from torch_geometric.utils import scatter

from deepchem.models import TorchModel


class VariationalRandomizer(nn.Module):
    """Add random noise to the embedding and include a corresponding loss."""

    def __init__(self, embedding_dimension, annealing_start_step,
                 annealing_final_step, **kwargs):
        super(VariationalRandomizer, self).__init__(**kwargs)
        self._embedding_dimension = embedding_dimension
        self._annealing_final_step = annealing_final_step
        self._annealing_start_step = annealing_start_step
        self.dense_mean = nn.LazyLinear(embedding_dimension)
        self.dense_stddev = nn.LazyLinear(embedding_dimension)
        self.combine = layers.CombineMeanStd(training_only=True)

    def forward(self, inputs, training=True):
        input, global_step = inputs
        embedding_mean = self.dense_mean(input)
        embedding_stddev = self.dense_stddev(input)
        embedding = self.combine([embedding_mean, embedding_stddev],
                                 training=training)
        mean_sq = embedding_mean * embedding_mean
        stddev_sq = embedding_stddev * embedding_stddev
        kl = mean_sq + stddev_sq - torch.log(stddev_sq + 1e-20) - 1
        anneal_steps = self._annealing_final_step - self._annealing_start_step
        if anneal_steps > 0:
            current_step = global_step.to(
                torch.float64) - self._annealing_start_step
            anneal_frac = torch.maximum(0.0, current_step) / anneal_steps
            kl_scale = torch.minimum(1.0, anneal_frac * anneal_frac)
        else:
            kl_scale = 1.0
        loss = 0.5 * kl_scale * torch.mean(kl)
        self.kl_loss = loss.item()
        return embedding

    def get_loss(self):
        return self.kl_loss


class SeqToSeqModel(TorchModel):
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
        if SeqToSeqModel.sequence_end not in input_tokens:
            input_tokens = input_tokens + [SeqToSeqModel.sequence_end]
        if SeqToSeqModel.sequence_end not in output_tokens:
            output_tokens = output_tokens + [SeqToSeqModel.sequence_end]
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens
        self._input_dict = dict((x, i) for i, x in enumerate(input_tokens))
        self._output_dict = dict((x, i) for i, x in enumerate(output_tokens))
        self._max_output_length = max_output_length
        self._embedding_dimension = embedding_dimension
        self._reverse_input = reverse_input
        self.encoder = self._create_encoder(encoder_layers, dropout)
        self.decoder = self._create_decoder(decoder_layers, dropout)

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
        self.fit_generator(self._generate_batches(sequences),
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
            probs = self.predict_on_generator([[
                (features, indices, np.array(self.get_global_step())), None,
                None
            ]])
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
                [[(features, indices, np.array(self.get_global_step())), None,
                  None]],
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
                if token == SeqToSeqModel.sequence_end:
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
                if len(c[2]) > 0 and c[2][-1] == SeqToSeqModel.sequence_end:
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
        features[np.arange(len(sequences)), lengths,
                 self._input_dict[SeqToSeqModel.sequence_end]] = 1
        return features

    def _create_output_array(self, sequences):
        """Create the array describing the target sequences for a batch."""
        lengths = [len(x) for x in sequences]
        labels = np.zeros(
            (self.batch_size, self._max_output_length, len(
                self._output_tokens)),
            dtype=np.float32)
        end_marker_index = self._output_dict[SeqToSeqModel.sequence_end]
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
            gather_indices = np.array([(i, len(x)) for i, x in enumerate(inputs)
                                      ])
            yield ([features, gather_indices,
                    np.array(self.get_global_step())], [labels], [])


class _create_encoder(nn.Module):
    """Encoder as a nn.Module."""
    def __init__(self, input_size, n_layers, _embedding_dimension, dropout):
        super(_create_encoder, self).__init__()
        self.n_layers = n_layers
        self._embedding_dimension = _embedding_dimension
        self.dropout = dropout
        self.GRU = nn.GRU(input_size, self._embedding_dimension, self.n_layers, dropout=self.dropout)
    def forward(self, input, gather_indices):
        output, hn = self.GRU(input)
        output = output[tuple(gather_indices)]
        return output
        

class _create_decoder(nn.Module):
    """Decoder as a nn.Module."""
    def __init__(self, n_layers, dropout):
        self.GRU = nn.GRU()