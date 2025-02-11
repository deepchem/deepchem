import torch.nn as nn
from torch import Tensor
from transformers import PreTrainedTokenizer, BertTokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.nn.modules.loss import _Loss
from deepchem.models.losses import Loss
from deepchem.models.torch_models import TorchModel
from deepchem.models.optimizers import Optimizer, Adam
from deepchem.data import Dataset, NumpyDataset
import torch
from typing import Iterable, Optional, Union, Tuple, List, Any, Dict, Sequence


class LSTMNeuralNet(nn.Module):
    """
    An LSTM-based neural network for token generation or classification tasks.
    This class contains an embedding layer, an LSTM layer, and a fully connected
    layer to translate hidden states into vocabulary tokens.

    Examples
    --------
    >>> import torch
    >>> from deepchem.models.torch_models.lstm_generator_models import LSTMNeuralNet
    >>> model = LSTMNeuralNet(vocab_size=12, embedding_dim=8, hidden_dim=4, num_layers=1)
    >>> x = torch.randint(0, 12, (2, 10))
    >>> output = model(x)
    """

    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,
                 hidden_dim: int,
                 num_layers: int = 1) -> None:
        """
        Initialize the LSTMNeuralNet.

        Parameters
        ----------
        vocab_size: int
            Size of the vocabulary.
        embedding_dim: int
            Dimension of the embedding vector.
        hidden_dim: int
            Number of hidden units in each LSTM layer.
        num_layers: int
            Number of LSTM layers in the network.
        """
        super(LSTMNeuralNet, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers,
                           batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the LSTMNeuralNet.

        Parameters
        ----------
        x: Tensor
            Input tensor of token ids.

        Returns
        -------
        Tensor
            Raw logits of shape [batch_size, sequence_length, vocab_size].
        """
        embedded = self.embedding(x)
        output, _ = self.rnn(embedded)
        output = self.fc(output)
        return output


class LSTMGenerator(TorchModel):
    """LSTM Generator

    This class implements an LSTM-based [1]_ generator for token generation tasks.
    The generator is trained on a list of sequences and can be used to generate
    new sequences of tokens. The model is implemented using PyTorch and can be useful
    to generate SMILES, PSMILES and Weighted Graph strings for generation tasks.
    The generator is used in our research paper "Open-source Polymer Generative Pipeline" [2]_ to generate
    hypothetical polymers using PSMILES and Weighted Directed Graph representations.

    References
    ----------
    .. [1] Staudemeyer, Ralf C., and Eric Rothstein Morris. "Understanding LSTM
        --a tutorial into long short-term memory recurrent neural networks."
        arXiv preprint arXiv:1909.09586 (2019).
    .. [2] Mohanty, Debasish, et al. "Open-source Polymer Generative Pipeline."
        arXiv preprint arXiv:2412.08658 (2024).

    Examples
    --------
    >>> from deepchem.models.torch_models import LSTMGenerator
    >>> from deepchem.data import NumpyDataset
    >>> generator = LSTMGenerator()
    >>> dataset = NumpyDataset(["CCC"])
    >>> loss = generator.fit(dataset, nb_epoch=1)
    >>> new_sequences = generator.sample(num_gen=10, max_len=10, temperature=1.0)
    """

    def __init__(
            self,
            loss: Union[Loss, _Loss] = nn.CrossEntropyLoss(),
            # There is a problem with deepchem's SparseSoftmaxCrossEntropy.
            # The reduction='none' kwarg is not allowing backpropagation gradients
            batch_size: int = 8,
            embedding_dim: int = 128,
            hidden_dim: int = 256,
            num_layers: int = 2,
            tokenizer: Optional[PreTrainedTokenizer] = None,
            learning_rate: Union[float] = 0.001,
            optimizer: Optional[Optimizer] = None,
            model_dir: str = "lstm_generator_model",
            device: Optional[torch.device] = None) -> None:
        """
        Initializes the LSTMGenerator model.

        Parameters
        ----------
        loss: Loss, default nn.CrossEntropyLoss
            Loss function to use for training the model.
        batch_size: int, default 8
            Batch size to use during training.
        embedding_dim: int, default 128
            Dimension of the embedding vector.
        hidden_dim: int, default 256
            Number of hidden units in each LSTM layer.
        num_layers: int, default 2
            Number of LSTM layers in the network.
        tokenizer: PreTrainedTokenizer, default None
            Tokenizer to use for tokenizing input sequences.
        learning_rate: float, default 0.001
            Learning rate to use during training.
        optimizer: Optimizer, default None
            Optimizer to use for training the model. If None, Adam optimizer is used.
        model_dir: str, default "lstm_generator_model"
            Directory to save the model files.
        device: torch.device, default None
            Device to use for training the model.
        """

        self.tokenizer = BertTokenizer.from_pretrained(
            'bert-base-cased') if tokenizer is None else tokenizer
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        model = LSTMNeuralNet(vocab_size=self.tokenizer.vocab_size,
                              embedding_dim=embedding_dim,
                              hidden_dim=hidden_dim,
                              num_layers=num_layers)
        optimizer = optimizer if optimizer else Adam(
            learning_rate=learning_rate)
        if device is None:
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        super(LSTMGenerator, self).__init__(model=model,
                                            loss=loss,
                                            batch_size=batch_size,
                                            model_dir=model_dir,
                                            optimizer=optimizer,
                                            device=device,
                                            learning_rate=learning_rate)

    def default_generator(self,
                          dataset: Dataset,
                          epochs: int = 1,
                          *args,
                          **kwargs) -> Iterable[Tuple[List, List, List]]:
        """
        Generates a default generator for the input sequences.

        Parameters
        ----------
        dataset: Dataset
            Dataset of input sequences to tokenize.
        epochs: int, default 1
            Number of epochs to train the model.

        Returns
        -------
        Iterable
            Generator that yields input, target and zero weight
            as a tuple of tensors for training the model.
        """
        sequences = []
        for (X_, _, _, _) in dataset.iterbatches(batch_size=self.batch_size):
            # Tokenize the sequence and add special tokens
            for sequence in list(X_):
                tokens = self.tokenizer.encode(sequence,
                                               add_special_tokens=True)
                sequences.append(tokens)

        # Convert the list of sequences into tensors and pad them
        padded_sequences = pad_sequence(
            [torch.tensor(seq) for seq in sequences
            ],  # Convert each sequence to a tensor
            batch_first=True,
            padding_value=self.tokenizer.
            pad_token_id  # Use BERT's padding token ID
        )

        for epoch in range(epochs):
            for i in range(0, len(padded_sequences), self.batch_size):
                batch = padded_sequences[
                    i:min(i + self.batch_size, len(padded_sequences))]
                inputs = batch[:, :-1]
                targets = batch[:, 1:]
                weights = torch.ones(inputs.shape[0], device=self.device)
                yield ([inputs], [targets], [weights])

    def fit(self,
            dataset: Dataset,
            nb_epoch: int = 1,
            max_checkpoints_to_keep: int = 5,
            checkpoint_interval: int = 1000,
            *args,
            **kwargs) -> float:
        """
        Fits the model on the input sequences.

        Parameters
        ----------
        dataset: Dataset
            Dataset of input sequences to train the model.
        nb_epoch: int, default 1
            Number of epochs to train the model.
        max_checkpoints_to_keep: int, default 5
            Maximum number of checkpoints to keep.
        checkpoint_interval: int, default 1000
            Interval at which to save model checkpoints.

        Returns
        -------
        float
            Average loss of most recent checkpoint
        """
        return self.fit_generator(
            self.default_generator(dataset=dataset, epochs=nb_epoch),
            checkpoint_interval=checkpoint_interval,
            max_checkpoints_to_keep=max_checkpoints_to_keep)

    def fit_generator(self,
                      generator: Iterable,
                      max_checkpoints_to_keep: int = 5,
                      checkpoint_interval: int = 1000,
                      *args,
                      **kwargs) -> float:
        """
        Fits the model on the input sequences using a generator.

        Parameters
        ----------
        generator: Iterable
            Generator that yields input, target and zero weight tensors for training the model.
        max_checkpoints_to_keep: int, default 5
            Maximum number of checkpoints to keep.
        checkpoint_interval: int, default 1000
            Interval at which to save model checkpoints.
        Returns
        -------
        float
            Average loss of most recent checkpoint
        """
        loss_float = 0.0
        current_step = 0
        optimizer = self.optimizer._create_pytorch_optimizer(
            self.model.parameters())
        for inputs, targets, _ in generator:
            input_val = inputs[0]
            input_val = input_val.to(self.device)
            target_val = targets[0]
            target_val = target_val.to(self.device)
            output = self.model(input_val)
            if isinstance(self.loss, Loss):
                # Convert DeepChem Loss to a PyTorch callable
                loss_fn = self.loss._create_pytorch_loss()
                loss_val = loss_fn(
                    output.reshape(-1, self.tokenizer.vocab_size),
                    target_val.reshape(-1))
            elif isinstance(self.loss, _Loss):
                # Already a PyTorch loss
                loss_val = self.loss(
                    output.reshape(-1, self.tokenizer.vocab_size),
                    target_val.reshape(-1))
            else:
                raise ValueError("Invalid loss type")

            # Backpropagation and optimize
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            if checkpoint_interval > 0 and current_step % checkpoint_interval == checkpoint_interval - 1:
                self.save_checkpoint(max_checkpoints_to_keep)
            loss_float = loss_val.item()
            current_step += 1

        return float(loss_float)

    def restore(self,
                checkpoint: Optional[str] = None,
                model_dir: Optional[str] = None,
                strict: Optional[bool] = True) -> None:
        """Restore the values of all variables from a checkpoint file.

        Parameters
        ----------
        checkpoint: str
            the path to the checkpoint file to load.  If this is None, the most recent
            checkpoint will be chosen automatically.  Call get_checkpoints() to get a
            list of all available checkpoints.
        model_dir: str, default None
            Directory to restore checkpoint from. If None, use self.model_dir.  If
            checkpoint is not None, this is ignored.
        strict: bool, default True
            Whether or not to strictly enforce that the keys in checkpoint match
            the keys returned by this model's get_variable_scope() method.
        """
        self._ensure_built()
        if checkpoint is None:
            checkpoints = sorted(self.get_checkpoints(model_dir))
            if len(checkpoints) == 0:
                raise ValueError('No checkpoint found')
            checkpoint = checkpoints[0]
        data = torch.load(checkpoint, map_location=self.device)
        self.model = LSTMNeuralNet(self.tokenizer.vocab_size,
                                   self.embedding_dim, self.hidden_dim,
                                   self.num_layers)
        if strict is not None:
            self.model.load_state_dict(data['model_state_dict'], strict=strict)
            return
        self.model.load_state_dict(data['model_state_dict'])

    def load_from_pretrained(self,
                             source_model: TorchModel,
                             assignment_map: Optional[Dict[Any, Any]] = None,
                             value_map: Optional[Dict[Any, Any]] = None,
                             checkpoint: Optional[str] = None,
                             model_dir: Optional[str] = None,
                             include_top: bool = True,
                             inputs: Optional[Sequence[Any]] = None,
                             **kwargs: Any) -> None:
        """
        Load model from a pretrained checkpoint.

        Parameters
        ----------
        source_model: TorchModel
            The source model to load from.
        assignment_map: dict, optional
            Parameter assignment (unused here).
        value_map: dict, optional
            Parameter assignment (unused here).
        checkpoint: str, optional
            Checkpoint file path.
        model_dir: str, optional
            Directory from which to restore the checkpoint.
        include_top: bool
            Whether to load the top layers (unused here).
        inputs: Sequence[Any], optional
            Input placeholders (unused here).
        """
        self._ensure_built()
        self.restore(model_dir=model_dir, checkpoint=checkpoint)
        self.model = self.model.to(self.device)

    def _predict(self, generator: Iterable[Tuple[Any, Any, Any]], *args: Any,
                 **kwargs: Any):
        """
        Predict the next token in the sequence.

        Parameters
        ----------
        generator: Iterable
            Input tensor of token ids.
        temperature: float, default 1.0
            (passed as keyword argument)
            Temperature to use for sampling.
        """
        if "temperature" in kwargs:
            temperature = kwargs["temperature"]
        else:
            temperature = 1.0
        for gen in generator:
            input_tensors, _, _ = gen
        input_tensor = torch.tensor(input_tensors)
        input_tensor = input_tensor.to(self.device)
        output = self.model(input_tensor)
        logits = output[:, -1, :] / temperature
        prbos = torch.softmax(logits, dim=-1)
        predicted_tokens = torch.multinomial(prbos, num_samples=1).item()
        return predicted_tokens

    def _single_sample(self,
                       max_len: int = 600,
                       temperature: float = 1.0) -> str:
        """
        Generate a sequence by single sampling.

        Parameters
        ----------
        max_len: int, default 600
            Maximum length of the generated sequence.
        temperature: float, default 1.0
            Temperature to use for sampling.

        Returns
        -------
        str
            Detokenized Generated sequence as str.
        """
        generated_sequence: list = [self.tokenizer.cls_token_id]
        for _ in range(max_len):
            input_tensor = torch.tensor(generated_sequence).unsqueeze(0)
            dataset = NumpyDataset(input_tensor)
            output_tensors: list = []

            def get_generator():
                for (X_, _, _,
                     _) in dataset.iterbatches(batch_size=self.batch_size):
                    yield X_, _, _

            output_tensor = self._predict(get_generator(),
                                          temperature=temperature)
            generated_sequence.append(output_tensor)
            if len(output_tensors) > 0:
                if output_tensors[0] == self.tokenizer.sep_token_id:
                    break
        generated_sequence_str: str = self.tokenizer.decode(
            generated_sequence, skip_special_tokens=True)
        generated_sequence_str = generated_sequence_str.replace(" ", "")
        return generated_sequence_str

    def sample(self,
               num_gen: int = 100,
               max_len: int = 600,
               temperature: float = 1.0) -> List[str]:
        """
        Generate sequences by repeated sampling.
        Parameters
        ----------
        num_gen: int, default 100
            Number of sequences to generate.
        max_len: int, default 600
            Maximum length of the generated sequence.
        temperature: float, default 1.0
            Temperature to use for sampling.
        Returns
        -------
        List[str]
            List of generated sequences as strings.
        """
        self.model.eval()
        results = []
        for i in range(num_gen):
            generated_sequence = self._single_sample(max_len=max_len,
                                                     temperature=temperature)
            results.append(generated_sequence)
        return results
