import torch
import torch.nn as nn
from deepchem.utils.typing import OneOrMany
from deepchem.models.torch_models.layers import HighwayLayer, DTNNEmbedding
import numpy as np
from deepchem.models.torch_models.torch_model import TorchModel
from deepchem.models.losses import L2Loss, SoftmaxCrossEntropy
from typing import List, Tuple, Iterable, Union, Any, Dict
from deepchem.data import Dataset
from deepchem.metrics import to_one_hot
import copy
import sys

default_dict = {
    '#': 1,
    '(': 2,
    ')': 3,
    '+': 4,
    '-': 5,
    '/': 6,
    '1': 7,
    '2': 8,
    '3': 9,
    '4': 10,
    '5': 11,
    '6': 12,
    '7': 13,
    '8': 14,
    '=': 15,
    'C': 16,
    'F': 17,
    'H': 18,
    'I': 19,
    'N': 20,
    'O': 21,
    'P': 22,
    'S': 23,
    '[': 24,
    '\\': 25,
    ']': 26,
    '_': 27,
    'c': 28,
    'Cl': 29,
    'Br': 30,
    'n': 31,
    'o': 32,
    's': 33
}


class TextCNN(nn.Module):
    """
    A 1D convolutional neural network for both classification and regression tasks.

    Reimplementation of the discriminator module in ORGAN [1] .
    Originated from [2].

    The model converts the input smile strings to an embedding vector, the vector
    is convolved and pooled through a series of convolutional filters which are concatnated
    and later passed through a simple dense layer. The resulting vector goes through a Highway
    layer [3] which finally as per the nature of the task is passed through a dense layer.

    References
    ----------
    .. [1]  Guimaraes, Gabriel Lima, et al. "Objective-reinforced generative adversarial networks (ORGAN) for sequence generation models." arXiv preprint arXiv:1705.10843 (2017).
    .. [2] Kim, Yoon. "Convolutional neural networks for sentence classification." arXiv preprint arXiv:1408.5882 (2014).
    .. [3] Srivastava et al., "Training Very Deep Networks".https://arxiv.org/abs/1507.06228

    Examples
    --------
    >>> from deepchem.models.torch_models.text_cnn import default_dict, TextCNN
    >>> import torch
    >>> batch_size = 1
    >>> input_tensor = torch.randint(34, (batch_size, 64))
    >>> cls_model = TextCNN(1, default_dict, 1, mode="classification")
    >>> reg_model = TextCNN(1, default_dict, 1, mode="regression")
    >>> cls_output = cls_model.forward(input_tensor)
    >>> reg_output = reg_model.forward(input_tensor)
    >>> assert len(cls_output) == 2
    >>> assert len(reg_output) == 1
    """

    def __init__(self,
                 n_tasks: int,
                 char_dict: Dict[str, int],
                 seq_length: int,
                 n_embedding: int = 75,
                 kernel_sizes: List[int] = [
                     1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20
                 ],
                 num_filters: List[int] = [
                     100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160
                 ],
                 dropout: float = 0.25,
                 mode: str = "classification") -> None:
        """
        Parameters
        ----------
        n_tasks: int
            Number of tasks
        char_dict: dict
            Mapping from characters in smiles to integers
        seq_length: int
            Length of sequences(after padding)
        n_embedding: int, optional
            Length of embedding vector
        kernel_sizes: list of int, optional
            Properties of filters used in the conv net
        num_filters: list of int, optional
            Properties of filters used in the conv net
        dropout: float, optional
            Dropout rate
        mode: str
            Either "classification" or "regression" for type of model.
        """

        super(TextCNN, self).__init__()

        self.n_tasks = n_tasks
        self.char_dict = char_dict
        self.seq_length = max(seq_length, max(kernel_sizes))
        self.n_embedding = n_embedding
        self.kernel_sizes = kernel_sizes
        self.num_filters = num_filters
        self.dropout = dropout
        self.mode = mode

        self.conv_layers = nn.ModuleList()
        self.embedding_layer = DTNNEmbedding(
            n_embedding=self.n_embedding,
            periodic_table_length=len(self.char_dict.keys()) + 1)
        self.dropout_layer = nn.Dropout1d(p=self.dropout)
        for filter_size, num_filter in zip(self.kernel_sizes, self.num_filters):
            self.conv_layers.append(
                nn.Conv1d(in_channels=self.n_embedding,
                          out_channels=num_filter,
                          kernel_size=filter_size,
                          padding=0,
                          dtype=torch.float32))
        concat_emb_dim = sum(num_filters)
        self.linear1 = nn.Linear(in_features=concat_emb_dim, out_features=200)
        if (self.mode == "classification"):
            self.linear2 = nn.Linear(in_features=200,
                                     out_features=self.n_tasks * 2)
        else:
            self.linear2 = nn.Linear(in_features=200,
                                     out_features=self.n_tasks * 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=2)
        self.highway = HighwayLayer(200)

    def forward(self, input: OneOrMany[torch.Tensor]) -> List[Any]:
        """
        Parameters
        ----------
        input: torch.Tensor
            Input Tensor
        Returns
        -------
        torch.Tensor
            Output as per use case : regression/classification
        """
        input_emb = self.embedding_layer(input)
        input_emb = input_emb.permute(0, 2, 1)

        conv_outputs = []
        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(input_emb)
            x, _ = torch.max(x, dim=2)
            conv_outputs.append(x)
            if (i == 0):
                concat_output = x
            else:
                concat_output = torch.cat((concat_output, x), dim=1)

        x = self.relu(self.linear1(self.dropout_layer(concat_output)))
        x = self.highway(x)

        if self.mode == "classification":
            logits = self.linear2(x)
            logits = logits.view(-1, self.n_tasks, 2)
            output = self.softmax(logits)
            outputs = [output, logits]
        else:
            output = self.linear2(x)
            output = output.view(-1, self.n_tasks, 1)
            outputs = [output]
        return outputs


class TextCNNModel(TorchModel):
    """
   A 1D convolutional neural network to work on smiles strings for both
   classification and regression tasks.


   Reimplementation of the discriminator module in ORGAN [1] .
   Originated from [2].


   The model converts the input smile strings to an embedding vector, the vector
   is convolved and pooled through a series of convolutional filters which are concatnated
   and later passed through a simple dense layer. The resulting vector goes through a Highway
   layer [3] which finally as per the nature of the task is passed through a dense layer.


   References
   ----------
   .. [1]  Guimaraes, Gabriel Lima, et al. "Objective-reinforced generative adversarial networks (ORGAN) for sequence generation models." arXiv preprint arXiv:1705.10843 (2017).
   .. [2] Kim, Yoon. "Convolutional neural networks for sentence classification." arXiv preprint arXiv:1408.5882 (2014).
   .. [3] Srivastava et al., "Training Very Deep Networks".https://arxiv.org/abs/1507.06228


   Examples
   --------
   >>> import os
   >>> from deepchem.models.torch_models import TextCNNModel
   >>> from deepchem.models.torch_models.text_cnn import default_dict
   >>> n_tasks = 1
   >>> seq_len = 250
   >>> model = TextCNNModel(n_tasks, default_dict, seq_len)
   """

    def __init__(self,
                 n_tasks: int,
                 char_dict: Dict[str, int],
                 seq_length: int,
                 n_embedding: int = 75,
                 kernel_sizes: List[int] = [
                     1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20
                 ],
                 num_filters: List[int] = [
                     100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160
                 ],
                 dropout: float = 0.25,
                 mode: str = "classification",
                 **kwargs) -> None:
        """
       Parameters
       ----------
       n_tasks: int
           Number of tasks
       char_dict: dict
           Mapping from characters in smiles to integers
       seq_length: int
           Length of sequences(after padding)
       n_embedding: int, optional
           Length of embedding vector
       filter_sizes: list of int, optional
           Properties of filters used in the conv net
       num_filters: list of int, optional
           Properties of filters used in the conv net
       dropout: float, optional
           Dropout rate
       mode: str
           Either "classification" or "regression" for type of model.
       """

        self.n_tasks = n_tasks
        self.char_dict = char_dict
        self.seq_length = max(seq_length, max(kernel_sizes))
        self.n_embedding = n_embedding
        self.kernel_sizes = kernel_sizes
        self.num_filters = num_filters
        self.dropout = dropout
        self.mode = mode

        self.model = TextCNN(
            n_tasks=n_tasks,
            char_dict=char_dict,
            seq_length=seq_length,
            n_embedding=n_embedding,
            kernel_sizes=kernel_sizes,
            num_filters=num_filters,
            dropout=dropout,
            mode=mode,
        )
        loss: Union[SoftmaxCrossEntropy, L2Loss]
        if self.mode == "classification":

            loss = SoftmaxCrossEntropy()
            output_types = ['prediction', 'loss']
        else:
            loss = L2Loss()
            output_types = ['prediction']

        super(TextCNNModel, self).__init__(self.model,
                                           loss=loss,
                                           output_types=output_types,
                                           **kwargs)

    # Below functions were taken from DeepChem TextCNN tensorflow implementation

    def default_generator(
            self,
            dataset: Dataset,
            epochs: int = 1,
            mode: str = 'fit',
            deterministic: bool = True,
            pad_batches: bool = True) -> Iterable[Tuple[List, List, List]]:
        """
        Transfer smiles strings to fixed length integer vectors

        Parameters
        ----------
        dataset: `dc.data.Dataset`
            Dataset to convert
        epochs: int, optional (Default 1)
            Number of times to walk over `dataset`
        mode: str, optional (Default 'fit')
            Ignored in this implementation.
        deterministic: bool, optional (Default True)
            Whether the dataset should be walked in a deterministic fashion
        pad_batches: bool, optional (Default True)
            If true, each returned batch will have size `self.batch_size`.

        Returns
        -------
        Iterator which walks over the batches
        """
        for epoch in range(epochs):
            for (X_b, y_b, w_b,
                 ids_b) in dataset.iterbatches(batch_size=self.batch_size,
                                               deterministic=deterministic,
                                               pad_batches=pad_batches):
                if y_b is not None:
                    if self.mode == 'classification':
                        y_b = to_one_hot(y_b.flatten(),
                                         2).reshape(-1, self.n_tasks, 2)
                # Transform SMILES sequence to integers

                X_b = self.smiles_to_seq_batch(ids_b)
                yield ([X_b], [y_b], [w_b])

    @staticmethod
    def build_char_dict(dataset: Dataset,
                        default_dict: Dict[str, int] = default_dict):
        """
        Collect all unique characters(in smiles) from the dataset.
        This method should be called before defining the model to build appropriate char_dict

        Parameters
        ----------
        dataset: Dataset
           Dataset for which char_dict is built for
        default_dict: dict, optional
           Mapping from characters in smiles to integers, optional

        Returns
        -------
        out_dict: dict
            A dictionary containing mapping between unique characters in the dataset to integers
        seq_length: int
            The maximum sequence length of smile strings found in the dataset multiplied by 1.2
        """
        X = dataset.ids
        # Maximum length is expanded to allow length variation during train and inference
        seq_length = int(max([len(smile) for smile in X]) * 1.2)
        # '_' served as delimiter and padding
        all_smiles = '_'.join(X)
        tot_len = len(all_smiles)
        # Initialize common characters as keys
        keys = list(default_dict.keys())
        out_dict = copy.deepcopy(default_dict)
        current_key_val = len(keys) + 1
        # Include space to avoid extra keys
        keys.extend([' '])
        extra_keys = []
        i = 0
        while i < tot_len:
            # For 'Cl', 'Br', etc.
            if all_smiles[i:i + 2] in keys:
                i = i + 2
            elif all_smiles[i:i + 1] in keys:
                i = i + 1
            else:
                # Character not recognized, add to extra_keys
                extra_keys.append(all_smiles[i])
                keys.append(all_smiles[i])
                i = i + 1
        # Add all extra_keys to char_dict
        for extra_key in extra_keys:
            out_dict[extra_key] = current_key_val
            current_key_val += 1
        return out_dict, seq_length

    def smiles_to_seq(self, smiles: str):
        """
        Tokenize characters in smiles to integers

        Parameters
        ----------
        smiles: str
           A smile string

        Returns
        -------
        array: np.ndarray
            An array of integers representing the tokenized sequence of characters.
        """
        smiles_len = len(smiles)
        seq = [0]
        keys = self.char_dict.keys()
        i = 0
        while i < smiles_len:
            # Skip all spaces
            if smiles[i:i + 1] == ' ':
                i = i + 1
            # For 'Cl', 'Br', etc.
            elif smiles[i:i + 2] in keys:
                seq.append(self.char_dict[smiles[i:i + 2]])
                i = i + 2
            elif smiles[i:i + 1] in keys:
                seq.append(self.char_dict[smiles[i:i + 1]])
                i = i + 1
            else:
                raise ValueError('character not found in dict')
        for i in range(self.seq_length - len(seq)):
            # Padding with '_'
            seq.append(self.char_dict['_'])
        return np.array(seq, dtype=np.int32)

    @staticmethod
    def convert_bytes_to_char(s: bytes) -> str:
        """
        Convert bytes to string.

        Parameters
        ----------
        s: bytes
            Bytes to be converted to string.

        Returns
        -------
        str
            String representation of the bytes.
        """
        out_str = ''.join(chr(b) for b in s)
        return out_str

    def smiles_to_seq_batch(
            self, ids_b: Union[List[Union[bytes, str]],
                               np.ndarray]) -> np.ndarray:
        """
        Converts SMILES strings to np.array sequence.

        Parameters
        ----------
        ids_b: Union[List[Union[bytes, str]], np.ndarray]
            A list of SMILES strings, either as bytes or strings.

        Returns
        -------
        np.ndarray
            A numpy array containing the tokenized sequences of SMILES strings.
        """
        if isinstance(ids_b, np.ndarray):
            ids_b = ids_b.tolist()  # Convert ndarray to list
        converted_ids_b = []
        for smiles in ids_b:
            if isinstance(smiles, bytes) and sys.version_info[0] != 2:
                converted_ids_b.append(
                    TextCNNModel.convert_bytes_to_char(smiles))
            elif isinstance(smiles, str):
                converted_ids_b.append(smiles)
            else:
                raise TypeError("Expected bytes or str, received: {}".format(
                    type(smiles)))
        smiles_seqs = [self.smiles_to_seq(smiles) for smiles in converted_ids_b]
        return np.vstack(smiles_seqs)
