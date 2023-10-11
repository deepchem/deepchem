from heapq import heappush, heappushpop
from typing import List

import numpy as np

import torch.nn as nn
import torch

from deepchem.models.torch_models.layers import EncoderRNN, DecoderRNN, VariationalRandomizer
from deepchem.models.torch_models import TorchModel
from deepchem.utils.batch_utils import batch_elements, create_input_array, create_output_array


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

    In this class, we establish a sequential model for the Sequence to Sequence (SeqToSeq) [1]_.

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
    >>> model = SeqToSeq(n_tokens, n_tokens, MAX_LENGTH, batch_size=batch_size,
    ...                  embedding_dimension=embedding_dimension)
    >>> inputs = create_input_array(train_smiles, MAX_LENGTH, False, batch_size,
    ...                             input_dict, " ")
    >>> output, embeddings = model([torch.tensor(inputs), torch.tensor([1])])
    >>> output.shape
    torch.Size([4, 57, 19])
    >>> embeddings.shape
    torch.Size([4, 16])

    References
    ----------
    .. [1] Sutskever et al., "Sequence to Sequence Learning with Neural Networks"

    """

    def __init__(self,
                 n_input_tokens: int,
                 n_output_tokens: int,
                 max_output_length: int,
                 encoder_layers: int = 4,
                 decoder_layers: int = 4,
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
        encoder_layers: int (default 4)
            Number of recurrent layers in the encoder
        decoder_layers: int (default 4)
            Number of recurrent layers in the decoder
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
            the tep (that is, batch) at which to finish turning on the constraint
            term for KL cost annealing.

        """
        super(SeqToSeq, self).__init__()
        self._variational = variational
        self.encoder = EncoderRNN(n_input_tokens, embedding_dimension,
                                  encoder_layers, dropout)
        self.decoder = DecoderRNN(embedding_dimension, n_output_tokens,
                                  decoder_layers, max_output_length, batch_size)
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


class SeqToSeqModel(TorchModel):
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
    >>> from deepchem.models.torch_models.seqtoseq import SeqToSeqModel
    >>> data = [
    ...     ("Cc1cccc(N2CCN(C(=O)C34CC5CC(CC(C5)C3)C4)CC2)c1C",
    ...      "Cc1cccc(N2CCN(C(=O)C34CC5CC(CC(C5)C3)C4)CC2)c1C"),
    ...     ("Cn1ccnc1SCC(=O)Nc1ccc(Oc2ccccc2)cc1",
    ...      "Cn1ccnc1SCC(=O)Nc1ccc(Oc2ccccc2)cc1"),
    ...     ("COc1cc2c(cc1NC(=O)CN1C(=O)NC3(CCc4ccccc43)C1=O)oc1ccccc12",
    ...      "COc1cc2c(cc1NC(=O)CN1C(=O)NC3(CCc4ccccc43)C1=O)oc1ccccc12"),
    ...     ("O=C1/C(=C/NC2CCS(=O)(=O)C2)c2ccccc2C(=O)N1c1ccccc1",
    ...      "O=C1/C(=C/NC2CCS(=O)(=O)C2)c2ccccc2C(=O)N1c1ccccc1"),
    ...     ("NC(=O)NC(Cc1ccccc1)C(=O)O",
    ...     "NC(=O)NC(Cc1ccccc1)C(=O)O")]
    >>> train_smiles = [s[0] for s in data]
    >>> tokens = set()
    >>> for s in train_smiles:
    ...     tokens = tokens.union(set(c for c in s))
    >>> tokens = sorted(list(tokens))
    >>> from deepchem.models.optimizers import Adam, ExponentialDecay
    >>> max_length = max(len(s) for s in train_smiles)
    >>> batch_size = 100
    >>> batches_per_epoch = len(train_smiles) / batch_size
    >>> model = SeqToSeqModel(
    ...     tokens,
    ...     tokens,
    ...     max_length,
    ...     encoder_layers=2,
    ...     decoder_layers=2,
    ...     embedding_dimension=256,
    ...     model_dir="fingerprint",
    ...     batch_size=batch_size,
    ...     learning_rate=ExponentialDecay(0.001, 0.9, batches_per_epoch))
    >>> for i in range(20):
    ...     loss = model.fit_sequences(data)
    >>> prediction = model.predict_from_sequences(train_smiles, 5)

    References
    ----------
    .. [1] Sutskever et al., "Sequence to Sequence Learning with Neural Networks"

    """

    sequence_end = object()

    def __init__(self,
                 input_tokens: List,
                 output_tokens: List,
                 max_output_length: int,
                 encoder_layers: int = 4,
                 decoder_layers: int = 4,
                 batch_size: int = 100,
                 embedding_dimension: int = 512,
                 dropout: float = 0.0,
                 reverse_input: bool = True,
                 variational: bool = False,
                 annealing_start_step: int = 5000,
                 annealing_final_step: int = 10000,
                 **kwargs):
        """Construct a SeqToSeq model.

        Parameters
        ----------
        input_tokens: list
            List of all tokens that may appear in input sequences.
        output_tokens: list
            List of all tokens that may appear in output sequences
        max_output_length: int
            Maximum length of output sequence that may be generated
        encoder_layers: int (default 4)
            Number of recurrent layers in the encoder
        decoder_layers: int (default 4)
            Number of recurrent layers in the decoder
        embedding_dimension: int (default 512)
            Width of the embedding vector. This also is the width of all
            recurrent layers.
        dropout: float (default 0.0)
            Dropout probability to use during training.
        reverse_input: bool (default True)
            If True, reverse the order of input sequences before sending
            them into the encoder. This can improve performance when
            working with long sequences.
        variational: bool (default False)
            If True, train the model as a variational autoencoder. This
            adds random noise to the encoder, and also constrains the
            embedding to follow a unit Gaussian distribution.
        annealing_start_step: int (default 5000)
            Step (that is, batch) at which to begin turning on the constraint
            term for KL cost annealing.
        annealing_final_step: int (default 10000)
            Step (that is, batch) at which to finish turning on the constraint
            term for KL cost annealing.

        """
        if SeqToSeqModel.sequence_end not in input_tokens:
            input_tokens = input_tokens + [SeqToSeqModel.sequence_end]
        if SeqToSeqModel.sequence_end not in output_tokens:
            output_tokens = output_tokens + [SeqToSeqModel.sequence_end]
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens
        self._input_dict = dict((x, i) for i, x in enumerate(input_tokens))
        self._output_dict = dict((x, i) for i, x in enumerate(output_tokens))
        self._n_input_tokens = len(input_tokens)
        self._n_output_tokens = len(output_tokens)
        self._max_output_length = max_output_length
        self.batch_size = batch_size
        self._embedding_dimension = embedding_dimension
        self._dropout = dropout
        self._reverse_input = reverse_input
        self._variational = variational
        self._annealing_start_step = annealing_start_step
        self._annealing_final_step = annealing_final_step

        self.model: nn.Module = SeqToSeq(
            n_input_tokens=self._n_input_tokens,
            n_output_tokens=self._n_output_tokens,
            max_output_length=self._max_output_length,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
            batch_size=self.batch_size,
            embedding_dimension=self._embedding_dimension,
            dropout=self._dropout,
            variational=self._variational,
            annealing_start_step=self._annealing_start_step,
            annealing_final_step=self._annealing_final_step)

        super(SeqToSeqModel,
              self).__init__(self.model,
                             self._create_loss(),
                             output_types=['prediction', 'embedding'],
                             batch_size=self.batch_size,
                             **kwargs)

    def _create_loss(self):
        """Create loss function for model."""
        if self._variational:
            loss = sum(self.model.randomizer.loss_list)
        else:
            loss = torch.tensor(0.0)

        def loss_fn(outputs, labels, weights):
            output = outputs[0].view(-1, outputs[0].size(-1))
            target = labels[0].view(-1)
            loss_ = nn.NLLLoss()(torch.log(output.to(torch.float32)),
                                 target.to(torch.int64))
            return loss + loss_

        return loss_fn

    def fit_sequences(self,
                      sequences: List[str],
                      max_checkpoints_to_keep: int = 5,
                      checkpoint_interval: int = 1000,
                      restore: bool = False):
        """Train this model on a set of sequences

        Parameters
        ----------
        sequences: List[str]
            Training samples to fit to. Each sample should be represented
            as a tuple of the form (input_sequence, output_sequence).
        max_checkpoints_to_keep: int
            Maximum number of checkpoints to keep. Older checkpoints are
            discarded.
        checkpoint_interval: int
            Frequency at which to write checkpoints, measured in training steps.
        restore: bool
            if True, restore the model from the most recent checkpoint and
            continue training from there. If False, retrain the model from
            scratch.

        """
        loss = self.fit_generator(
            self._generate_batches(sequences),
            max_checkpoints_to_keep=max_checkpoints_to_keep,
            checkpoint_interval=checkpoint_interval,
            restore=restore)
        return loss

    def predict_from_sequences(self, sequences: List[str], beam_width=5):
        """Given a set of input sequences, predict the output sequences.

        The prediction is done using a beam search with length normalization.

        Parameters
        ----------
        sequences: List[str]
            Input sequences to generate a prediction for
        beam_width: int (default 5)
            Beam width to use for searching.
            Set to 1 to use a simple greedy search.

        """
        result = []
        for batch in batch_elements(sequences, self.batch_size):
            features = create_input_array(batch, self._max_output_length,
                                          self._reverse_input, self.batch_size,
                                          self._input_dict,
                                          SeqToSeqModel.sequence_end)
            probs = self.predict_on_generator([
                ((features, np.array(self.get_global_step())), None, None)
            ])
            for i in range(len(batch)):
                result.append(self._beam_search(probs[i], beam_width))
        return result

    def predict_embedding(self, sequences: List[str]):  # type: ignore[override]
        """Given a set of input sequences, compute the embedding vectors.

        Parameters
        ----------
        sequences: List[str]
            Input sequences to generate embeddings for.

        """
        result = []
        for batch in batch_elements(sequences, self.batch_size):
            features = create_input_array(batch, self._max_output_length,
                                          self._reverse_input, self.batch_size,
                                          self._input_dict,
                                          SeqToSeqModel.sequence_end)
            probs = self.predict_on_generator(
                [((features, np.array(self.get_global_step())), None, None)],
                output_types=["embedding"])
            for i in range(len(batch)):
                result.append(probs[i])
        return result

    def predict_from_embedding(self,
                               embeddings: List[np.ndarray],
                               beam_width=5):
        """Given a set of embedding vectors, predict the output sequences.

        The prediction is done using a beam search with length normalization.

        Parameters
        ----------
        embeddings: List[np.ndarray]
            Embedding vectors to generate predictions for
        beam_width: int
            Beam width to use for searching.
            Set to 1 to use a simple greedy search.

        """
        result = []
        for batch in batch_elements(embeddings, self.batch_size):
            embedding_array = np.zeros(
                (self.batch_size, self._embedding_dimension), dtype=np.float32)
            for i, e in enumerate(batch):
                embedding_array[i] = e
            probs, _ = self.model.decoder(  # type: ignore
                [torch.tensor(embedding_array, device=self.device), None])
            probs = probs.cpu().detach().numpy()
            for i in range(len(batch)):
                result.append(self._beam_search(probs[i], beam_width))
        return result

    def _beam_search(self, probs: np.ndarray, beam_width: int):
        """Perform a beam search for the most likely output sequence.

        Beam search is a heuristic search algorithm often used in natural
        language processing.It is primarily used for finding the most likely
        sequence of decisions in probabilistic models. Instead of exhaustively
        exploring all possibilities, beam search maintains a set of "beams" or
        hypotheses at each step and narrows down the search space based on a
        predetermined beam width.

        Parameters
        ----------
        probs: np.ndarray
            Probability distribution over output tokens at each step.
        beam_width: int
            Beam width to use for searching.
            Set to 1 to use a simple greedy search.

        Returns
        -------
        List
            Most likely output sequence.

        """
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
        candidates: List = [(0.0, 0.0, [])]
        for i in range(len(logprobs)):
            new_candidates: List = []
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

    def _generate_batches(self, sequences: List[str]):
        """Create feed_dicts for fitting.

        Parameters
        ----------
        sequences: List[str]
            Training samples to fit to. Each sample should be represented
            as a tuple of the form (input_sequence, output_sequence).

        Returns
        -------
        Iterable
            Feed_dicts for fitting.

        """
        for batch in batch_elements(sequences, self.batch_size):
            inputs = []
            outputs = []
            for input, output in batch:
                inputs.append(input)
                outputs.append(output)
            for i in range(len(inputs), self.batch_size):
                inputs.append([])
                outputs.append([])
            features = create_input_array(inputs, self._max_output_length,
                                          self._reverse_input, self.batch_size,
                                          self._input_dict,
                                          SeqToSeqModel.sequence_end)
            labels = create_output_array(outputs, self._max_output_length,
                                         self.batch_size, self._output_dict,
                                         SeqToSeqModel.sequence_end)
            yield ([features, np.array(self.get_global_step())], [labels], [])
