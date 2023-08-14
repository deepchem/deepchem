"""Sequence to sequence translation models."""

from heapq import heappush, heappushpop
from deepchem.models.torch_models import layers
import torch
import torch.nn as nn
from typing import  List, Tuple

import numpy as np
from deepchem.utils.pytorch_utils import get_activation

from deepchem.models.torch_models import TorchModel

import torch
from torch import nn

def param_count(matrix):
    """Count the number of weights in a matrix or TT matrix"""
    assert isinstance(matrix, torch.nn.Module)
    total = 0
    for param in matrix.parameters():
        num = param.shape
        total += num.numel()
    
    return total

class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias, device):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.device = device
        self.input_weights = self._create_input_hidden_weights()
        self.hidden_weights = self._create_hidden_hidden_weights()

    def _create_input_hidden_weights(self):
        return nn.Linear(self.input_size, 3 * self.hidden_size, self.bias).to(self.device)

    def _create_hidden_hidden_weights(self):
        return nn.Linear(self.hidden_size, 3 * self.hidden_size, self.bias).to(self.device)

    def forward(self, input, hx):
        # Updates hidden state using the following rules:
        # r_t = \sigma(W_{ir} x_t + b_{ir} +        W_{hr} h_{(t-1)} + b_{hr})  RESET
        # z_t = \sigma(W_{iz} x_t + b_{iz} +        W_{hz} h_{(t-1)} + b_{hz})  UPDATE
        # n_t = \tanh( W_{in} x_t + b_{in} + r_t * (W_{hn} h_{(t-1)} + b_{hn})) NEW
        # h_t = (1 - z_t) * n_t + z_t * h_{(t-1)}

        # Apply the stacked weights
        input_part = self.input_weights(input)
        hidden_part = self.hidden_weights(hx)

        # Update hidden state using the above rules
        hsize = self.hidden_size
        reset_gate = torch.sigmoid(input_part[:, :hsize] +
                                   hidden_part[:, :hsize])
        update_gate = torch.sigmoid(input_part[:, hsize:2 * hsize] +
                                    hidden_part[:, hsize:2 * hsize])
        new_gate = torch.tanh(input_part[:, 2 * hsize:] +
                              reset_gate * hidden_part[:, 2 * hsize:])
        hy = (1 - update_gate) * new_gate + update_gate * hx

        return hy


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, device,
                 bias=True, log_grads=False):
        super(GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.device = device
        self.log_grads = log_grads

        # instantiate gru cell for each layer.
        self._all_layers = []
        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            if i == 0:
                cell = self._create_first_layer_cell()
            else:
                cell = self._create_other_layer_cell()
            setattr(self, name, cell)
            self._all_layers.append(cell)

    def _create_first_layer_cell(self):
        return GRUCell(self.input_size, self.hidden_size, self.bias, self.device)

    def _create_other_layer_cell(self):
        return GRUCell(self.hidden_size, self.hidden_size, self.bias, self.device)

    def init_hidden(self, batch_size):
        h = torch.zeros(batch_size, self.hidden_size).to(self.device)
        return h

    def param_count(self):
        total = 0
        for cell in self._all_layers:
            for attr in ('input_weights', 'hidden_weights'):
                total += param_count(getattr(cell, attr))
        return total

    def forward(self, input, init_states=None):
        """
        :param input:       Tensor of input data of shape (batch_size, seq_len, input_size).
        :param init_states: Initial hidden states of GRU. If None, is initialized to zeros.
                            Shape is (batch_size, hidden_size).

        :return:    outputs, h
                    outputs:  Torch tensor of shape (seq_len, batch_size, hidden_size)
                              containing output features from last layer of GRU.
                    h:        Output features (ie hiddens state) from last time step of
                              the last layer. Shape is (batch_size, hidden_size)
        """

        batch_size, seq_len, input_size = input.size()
        outputs = torch.zeros(batch_size, seq_len, self.hidden_size).to(input.device)

        # initialise hidden and cell states.
        h = self.init_hidden(batch_size) if init_states is None else init_states
        internal_state = [h] * self.num_layers

        for step in range(seq_len):
            x = input[:, step, :]
            for i in range(self.num_layers):
                # name = 'cell{}'.format(i)
                # gru_cell = getattr(self, name)
                gru_cell = self._all_layers[i]

                h = internal_state[i]
                x = gru_cell(x, h)
                internal_state[i] = x
            outputs[:, step, :] = x

        return outputs, x

class VariationalRandomizer(nn.Module):
    """Add random noise to the embedding and include a corresponding loss."""

    def __init__(self,
                 embedding_dimension: int,
                 annealing_start_step: int,
                 annealing_final_step: int,
                 **kwargs):
        super(VariationalRandomizer, self).__init__(**kwargs)
        self._embedding_dimension = embedding_dimension
        self._annealing_final_step = annealing_final_step
        self._annealing_start_step = annealing_start_step
        self.dense_mean = nn.LazyLinear(embedding_dimension)
        self.dense_stddev = nn.LazyLinear(embedding_dimension)
        self.combine = layers.CombineMeanStd(training_only=True)
        self.kl_loss = list()

    def forward(self, inputs: List[torch.Tensor], training=True):
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
            anneal_frac = torch.maximum(torch.tensor(0.0), current_step) / anneal_steps
            kl_scale = torch.minimum(torch.tensor(1.0), anneal_frac * anneal_frac)
        else:
            kl_scale = 1.0
        loss = 0.5 * kl_scale * torch.mean(kl)
        self.kl_loss.append(loss)
        return embedding

    def get_loss(self):
        return self.kl_loss

class _create_encoder(nn.Module):
    """Encoder as a nn.Module."""
    def __init__(self, input_size, n_layers, _embedding_dimension, dropout=0.1):
        super(_create_encoder, self).__init__()
        self.input_size = input_size
        self.n_layers = n_layers
        self._embedding_dimension = _embedding_dimension
        self.dropout = dropout
        self.GRU = GRU(self.input_size, self._embedding_dimension, self.n_layers, 'cuda', True)
        self.l = torch.Tensor([])
    def forward(self, inputs):
        input_ = inputs[0]
        gather_indices = inputs[1]
        input_, hidden = self.GRU(input_)
        def mapper(data: torch.Tensor, indices: torch.Tensor):
            l = list()
            for i in range(len(data)):
                l.append(data[indices[i][0]][indices[i][1]])
            return torch.stack(l, 0)
        a = lambda x: mapper(x[0], x[1])
        output = a([input_, gather_indices])
        return output


class _create_decoder(nn.Module):
    """Decoder as a nn.Module."""
    def __init__(self, n_layers, embedding_dimension, max_output_length, output_tokens, dropout):
        super(_create_decoder, self).__init__()
        #self.input_size = input_size
        self.n_layers = n_layers
        self._embedding_dimension = embedding_dimension
        self._max_output_length = max_output_length
        self._output_tokens = output_tokens
        self.dropout = dropout
        self.GRU = GRU(self._embedding_dimension, self._embedding_dimension, self.n_layers, 'cuda', True)
        self.final_linear = nn.LazyLinear(len(self._output_tokens))
        self.act = get_activation("softmax")
    def forward(self, inputs: torch.Tensor):
        inputs = torch.stack(self._max_output_length * [inputs], 1)
        inputs, hidden = self.GRU(inputs)
        output = self.final_linear(inputs)
        output = self.act(output)
        return output


class SeqToSeq(nn.Module):
    def __init__(self,
                 input_tokens: List,
                 output_tokens: List,
                 max_output_length: int,
                 encoder_layers: int=4,
                 decoder_layers: int=4,
                 embedding_dimension: int=512,
                 dropout: float=0.0,
                 reverse_input: bool=True,
                 variational: bool=False,
                 annealing_start_step: int=5000,
                 annealing_final_step: int=10000):
        super(SeqToSeq, self).__init__()
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens
        self._max_output_length = max_output_length
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self._embedding_dimension = embedding_dimension
        self.dropout = dropout
        self._reverse_input = reverse_input
        self.variational = variational
        self.annealing_start_step = annealing_start_step
        self.annealing_final_step = annealing_final_step
        self.encoder = _create_encoder(len(self._input_tokens), self.encoder_layers, self._embedding_dimension, self.dropout)
        self.decoder = _create_decoder(self.decoder_layers, self._embedding_dimension, self._max_output_length, self._output_tokens, self.dropout)
        if self.variational:
            self.randomizer = VariationalRandomizer(self._embedding_dimension,
                                               self.annealing_start_step,
                                               self.annealing_final_step)
    def forward(self, inputs):
        """
        - features
        - gather_indices
        - global_step
        """
        features = inputs[0]
        gather_indices = inputs[1]
        global_step = inputs[2]
        embedding = self.encoder([features, gather_indices])
        self.encoder.training = False
        self._embedding = self.encoder([features, gather_indices])
        self.encoder.training = True

        if self.variational:
            embedding = self.randomizer([self._embedding, global_step])
            self.randomizer.training = False
            self._embedding = self.randomizer([self._embedding, global_step])
            self.randomizer.training = True
        output = self.decoder(embedding)
        return output


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
        self.variational=False
        self.annealing_start_step=5000
        self.annealing_final_step=10000

        model = SeqToSeq(input_tokens,
                 output_tokens,
                 max_output_length,
                 encoder_layers,
                 decoder_layers,
                 embedding_dimension,
                 dropout,
                 reverse_input,
                 variational,
                 annealing_start_step,
                 annealing_final_step)
        super(SeqToSeqModel, self).__init__(model, self._create_loss(), **kwargs)

    def _create_loss(self):
        """Create the loss function."""
        def loss_fn(outputs, labels, weights):
            prob = torch.sum(outputs[0] * labels[0], dim=2)
            mask = torch.sum(labels[0], dim=2)
            log_prob = torch.log(prob + 1e-20) * mask
            loss = -torch.mean(torch.sum(log_prob, dim=1))
            return loss
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
            probs = self.model.decoder(embedding_array, training=False)
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
        #for i in sequences:
        #    print(i)
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
