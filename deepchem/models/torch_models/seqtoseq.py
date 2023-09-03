from typing import List
import torch.nn as nn
from deepchem.models.torch_models.layers import EncoderRNN, DecoderRNN, VariationalRandomizer


class SeqToSeq(nn.Module):

    def __init__(self,
                 n_input_tokens,
                 n_output_tokens,
                 max_output_length,
                 embedding_dimension=512,
                 dropout=0.0,
                 variational=False,
                 annealing_start_step=5000,
                 annealing_final_step=10000):
        super(SeqToSeq, self).__init__()
        self._variational = variational
        self.encoder = EncoderRNN(n_input_tokens, embedding_dimension, dropout)
        self.decoder = DecoderRNN(embedding_dimension, n_output_tokens,
                                  max_output_length)
        if variational:
            self.randomizer = VariationalRandomizer(self._embedding_dimension,
                                                    annealing_start_step,
                                                    annealing_final_step)

    def forward(self, inputs: List):
        input, global_step = inputs
        embedding, _ = self.encoder(input)
        self.encoder.training = False
        self._embedding, _ = self.encoder(input)
        self.encoder.training = True
        if self._variational:
            embedding = self.randomizer([self._embedding, global_step])
            self._embedding = self.randomizer([self._embedding, global_step],
                                              training=False)
        output, _ = self.decoder(
            [embedding, embedding[:, -1].unsqueeze(0), None])
        return output
