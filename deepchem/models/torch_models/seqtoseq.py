from typing import List

import torch.nn as nn
import torch

from deepchem.models.torch_models.layers import EncoderRNN, DecoderRNN, VariationalRandomizer


class SeqToSeq(nn.Module):
    """Implements sequence to sequence translation models.

    The model is based on the description in Sutskever et al., "Sequence to
    Sequence Learning with Neural Networks" (https://arxiv.org/abs/1409.3215),
    although this implementation uses GRUs instead of LSTMs. The goal is to
    take sequences of tokens as input, and translate each one into a different
    output sequence. The input and output sequences can both be of variable
    length, and an output sequence need not have the same length as the input
    sequence it was generated from. For example, these models were originally
    developed for use in natural language processing. In that context, the
    input might be a sequence of English words, and the output might be a
    sequence of French words. The goal would be to train the model to translate
    sentences from English to French.

    The model consists of two parts called the "encoder" and "decoder". Each one
    consists of a stack of recurrent layers. The job of the encoder is to
    transform the input sequence into a single, fixed length vector called the
    "embedding". That vector contains all relevant information from the input
    sequence. The decoder then transforms the embedding vector into the output
    sequence.

    These models can be used for various purposes. First and most obviously,
    they can be used for sequence to sequence translation. In any case where you
    have sequences of tokens, and you want to translate each one into a different
    sequence, a SeqToSeq model can be trained to perform the translation.

    Another possible use case is transforming variable length sequences into
    fixed length vectors. Many types of models require their inputs to have a
    fixed shape, which makes it difficult to use them with variable sized inputs
    (for example, when the input is a molecule, and different molecules have
    different numbers of atoms). In that case, you can train a SeqToSeq model as
    an autoencoder, so that it tries to make the output sequence identical to the
    input one. That forces the embedding vector to contain all information from
    the original sequence. You can then use the encoder for transforming
    sequences into fixed length embedding vectors, suitable to use as inputs to
    other types of models.

    Another use case is to train the decoder for use as a generative model. Here
    again you begin by training the SeqToSeq model as an autoencoder. Once
    training is complete, you can supply arbitrary embedding vectors, and
    transform each one into an output sequence. When used in this way, you
    typically train it as a variational autoencoder. This adds random noise to
    the encoder, and also adds a constraint term to the loss that forces the
    embedding vector to have a unit Gaussian distribution. You can then pick
    random vectors from a Gaussian distribution, and the output sequences should
    follow the same distribution as the training data.

    When training as a variational autoencoder, it is best to use KL cost
    annealing, as described in https://arxiv.org/abs/1511.06349. The constraint
    term in the loss is initially set to 0, so the optimizer just tries to
    minimize the reconstruction loss. Once it has made reasonable progress
    toward that, the constraint term can be gradually turned back on. The range
    of steps over which this happens is configurable.

    In this class, we establish a sequential model for the Sequence to Sequence (DTNN) [1]_.

    Examples
    --------
    >>> import torch
    >>> from deepchem.models.torch_models.seqtoseq import SeqToSeq
    >>> from deepchem.utils.batch_utils import create_input_array
    >>> # Dataset of SMILES strings for testing SeqToSeq models.
    >>> train_smiles = [
    ...     'Cc1cccc(N2CCN(C(=O)C34CC5CC(CC(C5)C3)C4)CC2)c1C',
    ...     'Cn1ccnc1SCC(=O)Nc1ccc(Oc2ccccc2)cc1',
    ...     'COc1cc2c(cc1NC(=O)CN1C(=O)NC3(CCc4ccccc43)C1=O)oc1ccccc12',
    ...     'CCCc1cc(=O)nc(SCC(=O)N(CC(C)C)C2CCS(=O)(=O)C2)[nH]1',
    ... ]
    >>> tokens = set()
    >>> for s in train_smiles:
    ...     tokens = tokens.union(set(c for c in s))
    >>> token_list = sorted(list(tokens))
    >>> batch_size = len(train_smiles)
    >>> MAX_LENGTH = max(len(s) for s in train_smiles)
    >>> token_list = token_list + [" "]
    >>> input_dict = dict((x, i) for i, x in enumerate(token_list))
    >>> n_tokens = len(token_list)
    >>> embedding_dimension = 16
    >>> model = SeqToSeq(n_tokens, n_tokens, MAX_LENGTH, batch_size,
    ...                  embedding_dimension)
    >>> inputs = create_input_array(train_smiles, MAX_LENGTH, False, batch_size,
    ...                             input_dict, " ")
    >>> output, embeddings = model([torch.tensor(inputs), torch.tensor([1])])
    >>> output.shape
    torch.Size([4, 57, 19])
    >>> embeddings.shape
    torch.Size([1, 4, 16])

    References
    ----------
    .. [1] Sutskever et al., "Sequence to Sequence Learning with Neural Networks"

    """

    def __init__(self,
                 n_input_tokens: int,
                 n_output_tokens: int,
                 max_output_length: int,
                 batch_size: int = 100,
                 embedding_dimension: int = 512,
                 dropout: float = 0.0,
                 variational: bool = False,
                 annealing_start_step: int = 5000,
                 annealing_final_step: int = 10000):
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
        annealing_start_step: int (default 5000)
            the step (that is, batch) at which to begin turning on the constraint
            term for KL cost annealing.
        annealing_final_step: int (default 10000)
            the step (that is, batch) at which to finish turning on the constraint
            term for KL cost annealing.

        """
        super(SeqToSeq, self).__init__()
        self._variational = variational
        self.encoder = EncoderRNN(n_input_tokens, embedding_dimension, dropout)
        self.decoder = DecoderRNN(embedding_dimension, n_output_tokens,
                                  max_output_length, batch_size)
        if variational:
            self.randomizer = VariationalRandomizer(embedding_dimension,
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
        _embedding: torch.Tensor
            Embeddings generated by the Encoder.

        """
        input, global_step = inputs
        _, embedding = self.encoder(input.to(torch.long))
        self.encoder.training = False
        _, self._embedding = self.encoder(input.to(torch.long))
        self.encoder.training = True
        if self._variational:
            embedding = self.randomizer([self._embedding, global_step])
            self._embedding = self.randomizer([self._embedding, global_step],
                                              training=False)
        output, _ = self.decoder([embedding, None])
        return output, self._embedding
