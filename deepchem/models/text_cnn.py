"""
Created on Thu Sep 28 15:17:50 2017

@author: zqwu
"""
import numpy as np
import tensorflow as tf
import copy
import sys
import warnings
from deepchem.metrics import to_one_hot
from deepchem.models import KerasModel, layers
from deepchem.models.losses import L2Loss, SoftmaxCrossEntropy
from tensorflow.keras.layers import Input, Dense, Reshape, Softmax, Dropout, Conv1D, Concatenate, Lambda

# Common symbols in SMILES, note that Cl and Br are regarded as single symbols
default_dict = {
    '#': 1, '(': 2, ')': 3, '+': 4, '-': 5, '/': 6, '1': 7, '2': 8, '3': 9, '4': 10,
    '5': 11, '6': 12, '7': 13, '8': 14, '=': 15, 'C': 16, 'F': 17, 'H': 18, 'I': 19,
    'N': 20, 'O': 21, 'P': 22, 'S': 23, '[': 24, '\\': 25, ']': 26, '_': 27, 'c': 28,
    'Cl': 29, 'Br': 30, 'n': 31, 'o': 32, 's': 33
}


class TextCNNModel(KerasModel):
    def __init__(self, n_tasks, char_dict, seq_length, n_embedding=75,
                 kernel_sizes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20],
                 num_filters=[100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160],
                 dropout=0.25, mode="classification", **kwargs):
        self.n_tasks = n_tasks
        self.char_dict = char_dict
        self.seq_length = max(seq_length, max(kernel_sizes))
        self.n_embedding = n_embedding
        self.kernel_sizes = kernel_sizes
        self.num_filters = num_filters
        self.dropout = dropout
        self.mode = mode

        smiles_seqs = Input(shape=(self.seq_length,), dtype=tf.int32)
        embedding = layers.DTNNEmbedding(n_embedding=self.n_embedding,
                                         periodic_table_length=len(self.char_dict) + 1)(smiles_seqs)

        pooled_outputs = []
        for filter_size, num_filter in zip(self.kernel_sizes, self.num_filters):
            conv_layer = Conv1D(kernel_size=filter_size, filters=num_filter, padding='valid')(embedding)
            reduced = Lambda(lambda x: tf.reduce_max(x, axis=1))(conv_layer)
            pooled_outputs.append(reduced)

        concat_outputs = Concatenate(axis=1)(pooled_outputs)
        dropout_layer = Dropout(rate=self.dropout)(concat_outputs)
        dense = Dense(200, activation=tf.nn.relu)(dropout_layer)
        gather = layers.Highway()(dense)

        if self.mode == "classification":
            logits = Dense(self.n_tasks * 2)(gather)
            logits = Reshape((self.n_tasks, 2))(logits)
            output = Softmax()(logits)
            outputs = [output, logits]
            output_types = ['prediction', 'loss']
            loss = SoftmaxCrossEntropy()
        else:
            output = Dense(self.n_tasks)(gather)
            output = Reshape((self.n_tasks, 1))(output)
            outputs = [output]
            output_types = ['prediction']
            loss = L2Loss()

        model = tf.keras.Model(inputs=[smiles_seqs], outputs=outputs)
        super(TextCNNModel, self).__init__(model, loss, output_types=output_types, **kwargs)

    @staticmethod
    def build_char_dict(dataset, default_dict=default_dict):
        """Efficiently builds a character dictionary from the dataset."""
        X = dataset.ids
        seq_length = int(max(len(smile) for smile in X) * 1.2)

        # Extract unique characters from SMILES strings
        all_chars = set('_'.join(X))
        out_dict = copy.deepcopy(default_dict)

        # Add new characters not in the default dictionary
        next_index = max(out_dict.values()) + 1
        for char in sorted(all_chars - set(out_dict.keys())):
            out_dict[char] = next_index
            next_index += 1

        return out_dict, seq_length

    @staticmethod
    def convert_bytes_to_char(s):
        return ''.join(chr(b) for b in s)

    def smiles_to_seq_batch(self, ids_b):
        """Converts SMILES strings to np.array sequence."""
        if isinstance(ids_b[0], bytes) and sys.version_info[0] != 2:
            ids_b = [TextCNNModel.convert_bytes_to_char(smiles) for smiles in ids_b]
        return np.vstack([self.smiles_to_seq(smiles) for smiles in ids_b])

    def default_generator(self, dataset, epochs=1, mode='fit', deterministic=True, pad_batches=True):
        """Transform smiles strings into fixed-length integer vectors."""
        for epoch in range(epochs):
            for X_b, y_b, w_b, ids_b in dataset.iterbatches(batch_size=self.batch_size,
                                                             deterministic=deterministic,
                                                             pad_batches=pad_batches):
                if y_b is not None and self.mode == 'classification':
                    y_b = to_one_hot(y_b.flatten(), 2).reshape(-1, self.n_tasks, 2)
                X_b = self.smiles_to_seq_batch(ids_b)
                yield ([X_b], [y_b], [w_b])

    def smiles_to_seq(self, smiles):
        """Tokenize characters in SMILES to integers."""
        seq = [0]
        for i in range(len(smiles)):
            if smiles[i] in self.char_dict:
                seq.append(self.char_dict[smiles[i]])
            else:
                raise ValueError(f"Character '{smiles[i]}' not found in dictionary")

        seq.extend([self.char_dict['_']] * (self.seq_length - len(seq)))
        return np.array(seq, dtype=np.int32)


class TextCNNTensorGraph(TextCNNModel):
    def __init__(self, *args, **kwargs):
        warnings.warn("TextCNNTensorGraph is deprecated and has been renamed to TextCNNModel", FutureWarning)
        super(TextCNNTensorGraph, self).__init__(*args, **kwargs)
