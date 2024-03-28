import numpy as np
import torch.nn as nn
from torch.nn import GRU, LSTM
from torchvision import transforms




from typing import Dict
from deepchem.data.datasets import pad_batch
from deepchem.models.torch_models import TorchModel
from deepchem.models.losses import L2Loss, SoftmaxCrossEntropy, SigmoidCrossEntropy
from deepchem.metrics import to_one_hot
from deepchem.models import chemnet_layers


DEFAULT_INCEPTION_BLOCKS = {"A": 3, "B": 3, "C": 3}

INCEPTION_DICT = {
    "A": chemnet_layers.InceptionResnetA,
    "B": chemnet_layers.InceptionResnetB,
    "C": chemnet_layers.InceptionResnetC
}

RNN_DICT = {"GRU": GRU, "LSTM": LSTM}


import torch
import torch.nn as nn
from collections import OrderedDict

class Smiles2Vec(nn.Module):
    """
    Implements the Smiles2Vec model, that learns neural representations of SMILES
    strings which can be used for downstream tasks.
    """

    def __init__(self,
                 char_to_idx,
                 n_tasks=10,
                 max_seq_len=270,
                 embedding_dim=50,
                 n_classes=2,
                 use_bidir=True,
                 use_conv=True,
                 filters=192,
                 kernel_size=3,
                 strides=1,
                 rnn_sizes=[224, 384],
                 rnn_types=["GRU", "GRU"],
                 mode="regression",
                 **kwargs):
        super(Smiles2Vec, self).__init__()

        self.char_to_idx = char_to_idx
        self.n_classes = n_classes
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim
        self.use_bidir = use_bidir
        self.use_conv = use_conv
        if use_conv:
            self.kernel_size = kernel_size
            self.filters = filters
            self.strides = strides
        self.rnn_types = rnn_types
        self.rnn_sizes = rnn_sizes
        assert len(rnn_sizes) == len(
            rnn_types), "Should have same number of hidden units as RNNs"
        self.n_tasks = n_tasks
        self.mode = mode

        if use_conv:
            self.conv1d = nn.Conv1d(embedding_dim, filters, kernel_size, stride=strides)

        rnn_layers = []
        for i, (rnn_size, rnn_type) in enumerate(zip(rnn_sizes, rnn_types)):
            if rnn_type.upper() == "GRU":
                rnn_layer = nn.GRU(filters if use_conv else embedding_dim, rnn_size, bidirectional=use_bidir, batch_first=True)
            elif rnn_type.upper() == "LSTM":
                rnn_layer = nn.LSTM(filters if use_conv else embedding_dim, rnn_size, bidirectional=use_bidir, batch_first=True)
            else:
                raise ValueError(f"Unknown RNN type: {rnn_type}")

            rnn_layers.append((f"rnn_{i}", rnn_layer))

        self.rnn = nn.Sequential(OrderedDict(rnn_layers))

        self.fc = nn.Linear(rnn_sizes[-1] * (2 if use_bidir else 1), n_tasks * n_classes if mode == "classification" else n_tasks)

        if self.mode == "classification":
            if self.n_classes == 2:
                self.output_activation = nn.Sigmoid()
            else:
                self.output_activation = nn.Softmax(dim=-1)

    def forward(self, smiles_seqs):
        embedding = nn.Embedding(len(self.char_to_idx), self.embedding_dim)
        rnn_input = embedding(smiles_seqs)

        if self.use_conv:
            rnn_input = rnn_input.permute(0, 2, 1)  # Convert to (batch_size, embedding_dim, seq_len) for Conv1D
            rnn_input = self.conv1d(rnn_input)
            rnn_input = rnn_input.permute(0, 2, 1)  # Convert back to (batch_size, seq_len, filters)

        rnn_embeddings, _ = self.rnn(rnn_input)
        x = rnn_embeddings.mean(dim=1)  # Global Average Pooling

        output = self.fc(x)

        if self.mode == "classification":
            output = output.view(-1, self.n_tasks, self.n_classes)
            output = self.output_activation(output)
            loss = nn.BCELoss() if self.n_classes == 2 else nn.CrossEntropyLoss()
            outputs = [output, output]  # Assuming the second output is logits
            output_types = ['prediction', 'loss']
        else:
            output = output.view(-1, self.n_tasks, 1)
            loss = nn.L1Loss()  # L2 Loss for regression
            outputs = [output]
            output_types = ['prediction']

        return outputs, loss, output_types
    
    
    def default_generator(self,
                      dataset,
                      epochs=1,
                      mode='fit',
                      deterministic=True,
                      pad_batches=True):
         for epoch in range(epochs):
             for X_b, y_b, w_b, ids_b in dataset.iterbatches(batch_size=self.batch_size,
                                                              deterministic=deterministic,
                                                              pad_batches=pad_batches):
                 X_b = torch.tensor(X_b, dtype=torch.float32)
                 y_b = torch.tensor(y_b, dtype=torch.float32)
                 w_b = torch.tensor(w_b, dtype=torch.float32)
     
                 if self.mode == 'classification':
                     y_b = nn.functional.one_hot(y_b.flatten(), self.n_classes).view(
                         -1, self.n_tasks, self.n_classes)
     
                 yield X_b, y_b, w_b
