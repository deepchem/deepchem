
"""
Created on Thu Sep 28 15:17:50 2017

@author: zqwu
"""
import torch
import copy
import sys
import torch.nn.functional as F

from deepchem.metrics import to_one_hot
from deepchem.models import TorchModel, layers
from torch.nn import  Linear, Unflatten, Dropout, Conv1d, Softmax, Dropout

# Common symbols in SMILES, note that Cl and Br are regarded as single symbol
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


class TextCNNModel(TorchModel):
    """ A Convolutional neural network on smiles strings

    Reimplementation of the discriminator module in ORGAN [1]_ .
    Originated from [2]_.

    This model applies multiple 1D convolutional filters to
    the padded strings, then max-over-time pooling is applied on
    all filters, extracting one feature per filter.  All
    features are concatenated and transformed through several
    hidden layers to form predictions.

    This model is initially developed for sentence-level
    classification tasks, with words represented as vectors. In
    this implementation, SMILES strings are dissected into
    characters and transformed to one-hot vectors in a similar
    way. The model can be used for general molecular-level
    classification or regression tasks. It is also used in the
    ORGAN model as discriminator.

    Training of the model only requires SMILES strings input,
    all featurized datasets that include SMILES in the `ids`
    attribute are accepted. PDBbind, QM7 and QM7b are not
    supported. To use the model, `build_char_dict` should be
    called first before defining the model to build character
    dict of input dataset, example can be found in
    examples/delaney/delaney_textcnn.py

    References
    ----------
    .. [1]  Guimaraes, Gabriel Lima, et al. "Objective-reinforced generative adversarial networks (ORGAN) for sequence generation models." arXiv preprint arXiv:1705.10843 (2017).
    .. [2] Kim, Yoon. "Convolutional neural networks for sentence classification." arXiv preprint arXiv:1408.5882 (2014).

    """

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
        """
        Parameters
        ----------
        n_tasks: int
            Number of tasks
        char_dict: dict
            Mapping from characters in smiles to integers
        seq_length: int
            Length of sequences (after padding)
        n_embedding: int, optional
            Length of embedding vector
        kernel_sizes: list of int, optional
            Properties of filters used in the convolutional network
        num_filters: list of int, optional
            Properties of filters used in the convolutional network
        dropout: float, optional
            Dropout rate
        mode: str
            Either "classification" or "regression" for the type of model
        """
        
        self.n_tasks = n_tasks
        self.char_dict = char_dict
        self.seq_length = max(seq_length, max(kernel_sizes))
        self.n_embedding = n_embedding
        self.kernel_sizes = kernel_sizes
        self.num_filters = num_filters
        self.dropout = dropout
        self.mode = mode


    @staticmethod
    def convert_bytes_to_char(s):
        s = ''.join(chr(b) for b in s)
        return s

    def smiles_to_seq_batch(self, ids_b):
        """Converts SMILES strings to np.array sequence.

        A tf.py_func wrapper is written around this when creating the input_fn for make_estimator
        """
        if isinstance(ids_b[0], bytes) and sys.version_info[0] != 2:  # Python 2.7 bytes and string are analogous
            ids_b = [
                TextCNNModel.convert_bytes_to_char(smiles) for smiles in ids_b
            ]
        smiles_seqs = [self.smiles_to_seq(smiles) for smiles in ids_b]
        smiles_seqs = torch.vstack(smiles_seqs)
        return smiles_seqs

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
                    if y_b is not None:
                        if self.mode == 'classification':
                            y_b = to_one_hot(y_b.flatten(),
                                             2).reshape(-1, self.n_tasks, 2)
                    # Transform SMILES sequence to integers
                    X_b = self.smiles_to_seq_batch(ids_b)
                    yield ([X_b], [y_b], [w_b])
