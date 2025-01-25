import torch.nn as nn
from torch import Tensor
from transformers import PreTrainedTokenizer, BertTokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.nn.modules.loss import _Loss
from deepchem.models.losses import Loss
from deepchem.models.torch_models import TorchModel
from deepchem.models.optimizers import Optimizer, Adam
from deepchem.models.losses import Loss, SparseSoftmaxCrossEntropy
from deepchem.utils.typing import LossFn
from deepchem.data import Dataset
import torch
from typing import Iterator, Optional, Union, Tuple, List


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
    The code is used in the Open-source Polymer Generative Pipeline [2]_ to generate 
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
    >>> generator = LSTMGenerator()
    >>> average_loss, least_loss = generator.fit(["CCC"], nb_epoch=1)
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
            device: str = "cpu") -> None:
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
        device: str, default "cpu"
            Device to use for training the model. Should be "cpu" or "cuda".
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
        model = model.to(device)
        optimizer = optimizer if optimizer else Adam(
            learning_rate=learning_rate)
        super(LSTMGenerator, self).__init__(model=model,
                                            loss=loss,
                                            batch_size=batch_size,
                                            model_dir=model_dir,
                                            optimizer=optimizer,
                                            device=device,
                                            learning_rate=learning_rate)

    def default_generator(self, dataset: Dataset, num_epochs: int = 1) -> Iterator:
        """
        Generates a default generator for the input sequences.

        Parameters
        ----------
        dataset: Dataset 
            Dataset of input sequences to tokenize.
        num_epochs: int, default 1
            Number of epochs to train the model.

        Returns
        -------
        Iterator
            Generator that yields input and target tensors for training the model.
        """
        sequences = []
        for (X_, _, _, _) in dataset.iterbatches(batch_size=self.batch_size):
            # Tokenize the sequence and add special tokens
            for sequence in  list(X_):
                tokens = self.tokenizer.encode(sequence, add_special_tokens=True)
                sequences.append(tokens)

        # Convert the list of sequences into tensors and pad them
        padded_sequences = pad_sequence(
            [torch.tensor(seq) for seq in sequences
            ],  # Convert each sequence to a tensor
            batch_first=True,
            padding_value=self.tokenizer.
            pad_token_id  # Use BERT's padding token ID
        )

        for epoch in range(num_epochs):
            for i in range(0, len(padded_sequences), self.batch_size):
                batch = padded_sequences[
                    i:min(i + self.batch_size, len(padded_sequences))]
                inputs = batch[:, :-1]
                targets = batch[:, 1:]
                inputs.to(self.device)
                targets.to(self.device)
                yield inputs, targets

    def fit(self,
            dataset: Dataset,
            nb_epoch: int = 1,
            checkpoint_interval: int = 1000,
            max_checkpoints_to_keep: int = 5,
            verbose: bool = False) -> Tuple[float, float]:
        """
        Fits the model on the input sequences.

        Parameters
        ----------
        dataset: Dataset 
            Dataset of input sequences to train the model.
        nb_epoch: int, default 1
            Number of epochs to train the model.
        checkpoint_interval: int, default 1000
            Interval at which to save model checkpoints.
        max_checkpoints_to_keep: int, default 5
            Maximum number of checkpoints to keep.
        verbose: bool, default False
            Whether to print training progress.

        Returns
        -------
        Tuple[float, float]
            Average loss and least loss during training.
        """
        return self.fit_generator(
            self.default_generator(dataset=dataset, num_epochs=nb_epoch),
            checkpoint_interval=checkpoint_interval,
            max_checkpoints_to_keep=max_checkpoints_to_keep,
            verbose=verbose)

    def fit_generator(self,
                      generator: Iterator,
                      checkpoint_interval: int = 1000,
                      max_checkpoints_to_keep: int = 5,
                      verbose: bool = False) -> Tuple[float, float]:
        """
        Fits the model on the input sequences using a generator.

        Parameters
        ----------
        generator: Iterator
            Generator that yields input and target tensors for training the model.
        checkpoint_interval: int, default 1000
            Interval at which to save model checkpoints.
        max_checkpoints_to_keep: int, default 5
            Maximum number of checkpoints to keep.
        verbose: bool, default False
            Whether to print training progress.


        Returns
        -------
        Tuple[float, float]
            Average loss and least loss during training
        """
        current_step = 1
        loss_float = 0.0
        least_loss = float('inf')
        optimizer = self.optimizer._create_pytorch_optimizer(
            self.model.parameters())
        for inputs, targets in generator:
            outputs = self.model(inputs)
            loss_val = self.loss(outputs.reshape(-1, self.tokenizer.vocab_size),
                                 targets.reshape(-1))
            # Backward and optimize
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            if checkpoint_interval > 0 and current_step % checkpoint_interval == checkpoint_interval - 1:
                self.save_checkpoint(max_checkpoints_to_keep)
            loss_float += loss_val.item()
            least_loss = min(least_loss, loss_val.item())
            if verbose:
                print(
                    f'Epoch [{epoch+1}/{num_epochs}], Step [{i}/{len(dataset)}], Loss: {loss.item():.4f}'
                )
            current_step += 1

        avg_loss = loss_float / current_step
        return avg_loss, least_loss

    def restore(self,
                model_dir: Optional[str] = None,
                checkpoint: Optional[str] = None,
                strict: Optional[bool] = True) -> None:
        """Reload the values of all variables from a checkpoint file.

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
        self.model.load_state_dict(data['model_state_dict'], strict=strict)

    def load_from_pretrained(self,
                             model_dir: Optional[str] = None,
                             checkpoint: Optional[str] = None) -> None:
        """
        Load the model from a pretrained model.

        Parameters
        ----------
        model_dir: str, default None
            Directory to restore checkpoint from. If None, use self.model_dir.
        checkpoint: str, default None
            the path to the checkpoint file to load.  If this is None, the most recent
            checkpoint will be chosen automatically.  Call get_checkpoints() to get a
            list of all available checkpoints.
        """
        self._ensure_built()
        self.restore(model_dir=model_dir, checkpoint=checkpoint)
        self.model = self.model.to(self.device)

    def _predict(self,
                 input_tensor: Tensor,
                 temperature: float = 1.0) -> Tensor:
        """
        Predict the next token in the sequence.

        Parameters
        ----------
        input_tensor: Tensor
            Input tensor of token ids.
        temperature: float, default 1.0
            Temperature to use for sampling.

        Returns
        -------
        Tensor
            Predicted token in pytorch tensor type
        """
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
        generated_sequence = [self.tokenizer.cls_token_id]
        for _ in range(max_len):
            input_tensor = torch.tensor(generated_sequence).unsqueeze(0)
            output_tensor = self._predict(input_tensor, temperature)
            generated_sequence.append(output_tensor)
            if output_tensor == self.tokenizer.sep_token_id:
                break
        generated_sequence = self.tokenizer.decode(generated_sequence,
                                                   skip_special_tokens=True)
        generated_sequence = generated_sequence.replace(" ", "")
        return generated_sequence

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
