from typing import Iterable, List, Tuple
from deepchem.data import Dataset
import torch
import numpy as np
from deepchem.models.torch_models.torch_model import TorchModel
from deepchem.models.torch_models import layers
import torch.nn as nn
from deepchem.models.losses import L2Loss, SoftmaxCrossEntropy
import torch.nn as nn
from deepchem.metrics import to_one_hot
import copy
from rdkit import Chem
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

    def __init__(self,
                 n_tasks,
                 char_dict,
                 seq_length,
                 n_embedding=75,
                 kernel_sizes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20],
                 num_filters=[
                     100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160
                 ],
                 dropout=0.25,
                 mode="classification"):

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
        self.embedding_layer = layers.DTNNEmbedding(
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
        self.highway = layers.HighwayLayer(200)

    def forward(self, input):
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

    def __init__(self,
                 n_tasks,
                 char_dict,
                 seq_length,
                 n_embedding=75,
                 kernel_sizes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20],
                 num_filters=[
                     100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160
                 ],
                 dropout=0.25,
                 mode="classification",
                 **kwargs):

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
    
    """
    Below code was taken from TextCNN tensorflow implementation
    """

    def default_generator(self,
                          dataset,
                          epochs=1,
                          mode='fit',
                          deterministic=True,
                          pad_batches=True):
        """Transfer smiles strings to fixed length integer vectors"""
        for epoch in range(epochs):
            for (X_b, y_b, w_b,
                 ids_b) in dataset.iterbatches(batch_size=self.batch_size,
                                               deterministic=deterministic,
                                               pad_batches=pad_batches):
                from rdkit import Chem
                if y_b is not None:
                    if self.mode == 'classification':
                        y_b = to_one_hot(y_b.flatten(),
                                         2).reshape(-1, self.n_tasks, 2)
                # Transform SMILES sequence to integers

                X_b = self.smiles_to_seq_batch(ids_b)
                yield ([X_b], [y_b], [w_b])

    @staticmethod
    def build_char_dict(dataset, default_dict=default_dict):
        """ Collect all unique characters(in smiles) from the dataset.
        This method should be called before defining the model to build appropriate char_dict
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

    #############################################################
    def smiles_to_seq(self, smiles):
        """ Tokenize characters in smiles to integers
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
    def convert_bytes_to_char(s):
        s = ''.join(chr(b) for b in s)
        return s

    def smiles_to_seq_batch(self, ids_b):
        """Converts SMILES strings to np.array sequence.

        A tf.py_func wrapper is written around this when creating the input_fn for make_estimator
        """
        if isinstance(ids_b[0], bytes) and sys.version_info[
                0] != 2:  # Python 2.7 bytes and string are analogous
            ids_b = [
                TextCNNModel.convert_bytes_to_char(smiles) for smiles in ids_b
            ]
        smiles_seqs = [self.smiles_to_seq(smiles) for smiles in ids_b]
        smiles_seqs = np.vstack(smiles_seqs)
        return smiles_seqs
