import torch
import torch.nn as nn
from typing import List, Any, Dict
from deepchem.utils.typing import OneOrMany
from deepchem.models.torch_models.layers import HighwayLayer, DTNNEmbedding

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
