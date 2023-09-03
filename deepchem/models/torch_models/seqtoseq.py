from typing import List
import torch.nn as nn
from deepchem.models.torch_models.layers import EncoderRNN, DecoderRNN, VariationalRandomizer


class SeqToSeq(nn.Module):

    def __init__(self,
                 n_input_tokens: int,
                 n_output_tokens: int,
                 max_output_length: int,
                 embedding_dimension: int=512,
                 dropout: float=0.0,
                 variational: bool=False,
                 annealing_start_step: int=5000,
                 annealing_final_step: int=10000):
        """Initialize SeqToSeq model.
        
        Parameters
        ----------
        n_input_tokens: int
            Number of input tokens.
        n_output_tokens: int
            Number of output tokens.
        max_output_length: int
            Maximum length of output sequence.
        embedding_dimension: int (default 512)
            Width of the embedding vector. This also is the width of all recurrent
            layers.
        dropout: float (default 0.0)
            Dropout probability to use during training.
        variational: bool (default False)
            If True, train the model as a variational autoencoder. This adds random
            noise to the encoder, and also constrains the embedding to follow a unit
            Gaussian distribution.
        annealing_start_step: int
            the step (that is, batch) at which to begin turning on the constraint
            term for KL cost annealing.
        annealing_final_step: int
            the step (that is, batch) at which to finish turning on the constraint
            term for KL cost annealing.

        """
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
        """Generates Embeddings using Encoder then passes it to Decoder to
        predict output sequences.
        
        Parameters
        ----------
        inputs: List
            List of two tensors. 
            First tensor is batch of input sequence.
            Second tensor is the current global_step.
        
        Returns
        -------
        output: torch.Tensor
            Predicted output sequence.

        """
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
