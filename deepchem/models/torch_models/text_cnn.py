import torch
import numpy as np
from deepchem.models.torch_models.torch_model import TorchModel
from deepchem.models.torch_models import layers
import torch.nn as nn
from deepchem.models.losses import L2Loss, SoftmaxCrossEntropy
from deepchem.metrics import to_one_hot
import copy
import sys
from typing import List, Any, Iterable, Tuple, Dict, Union
from deepchem.utils.typing import OneOrMany
from deepchem.data import Dataset

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

        self.model = layers.TextCNN(
            n_tasks=n_tasks,
            char_dict=char_dict,
            seq_length=seq_length,
            n_embedding=n_embedding,
            kernel_sizes=kernel_sizes,
            num_filters=num_filters,
            dropout=dropout,
            mode=mode,
        )
        LossType = Union[SoftmaxCrossEntropy, L2Loss]
        loss: LossType
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

    # Below functions were adapted from DeepChem TextCNN tensorflow implementation

    def default_generator(
            self,
            dataset: Dataset,
            epochs: int = 1,
            mode: str = 'fit',
            deterministic: bool = True,
            pad_batches: bool = True) -> Iterable[Tuple[List, List, List]]:
        """
        Create a generator that iterates batches for a dataset.

        Overrides the existing ``default_generator`` method to customize how model inputs are
        generated from the data.

        Here, the ``smiles_to_seq_batch`` helper function is used, to convert smile strings into fixed length integer vectors.
        If TextCNNModel's mode is classification then the labels are one hot encoded, not this mode is different from the mode defined in this functions arguments.

        Parameters
        ----------
        dataset: Dataset
            the data to iterate
        epochs: int
            the number of times to iterate over the full dataset
        mode: str
            allowed values are 'fit' (called during training), 'predict' (called
            during prediction), and 'uncertainty' (called during uncertainty
            prediction)
        deterministic: bool
            whether to iterate over the dataset in order, or randomly shuffle the
            data for each epoch
        pad_batches: bool
            whether to pad each batch up to this model's preferred batch size

        Examples
        --------
        >>> from deepchem.models.torch_models import TextCNNModel
        >>> import deepchem as dc
        >>> featurizer = dc.feat.RawFeaturizer()
        >>> tasks = ["outcome"]
        >>> n_tasks = 1
        >>> batch_size = 1
        >>> input_file = "deepchem/models/tests/assets/example_classification.csv"
        >>> loader = dc.data.CSVLoader(tasks=tasks,feature_field="smiles",featurizer=featurizer)
        >>> dataset = loader.create_dataset(input_file)
        >>> char_dict, length = TextCNNModel.build_char_dict(dataset)
        >>> torch_model = TextCNNModel(n_tasks,char_dict=char_dict,seq_length=length,batch_size=batch_size,learning_rate=0.001,use_queue=False,mode="classification")
        >>> generator = torch_model.default_generator(dataset = dataset)
        >>> first_batch = next(generator)
        >>> first_batch[0][0].shape
        (1, 64)

        Returns
        -------
        a generator that iterates batches, each represented as a tuple of lists:
        ([inputs], [outputs], [weights])
        Here, [inputs] is fixed length integer vectors representing input smile strings.
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
    def build_char_dict(dataset, default_dict=default_dict):
        """
        Collect all unique characters(in smiles) from the dataset.
        This method should be called before defining the model to build appropriate char_dict

        Examples
        --------
        >>> from deepchem.models.torch_models import TextCNNModel
        >>> import deepchem as dc
        >>> featurizer = dc.feat.RawFeaturizer()
        >>> tasks = ["outcome"]
        >>> input_file = "deepchem/models/tests/assets/example_classification.csv"
        >>> loader = dc.data.CSVLoader(tasks=tasks,feature_field="smiles",featurizer=featurizer)
        >>> dataset = loader.create_dataset(input_file)
        >>> char_dict, length = TextCNNModel.build_char_dict(dataset)
        >>> length
        64
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

    def smiles_to_seq(self, smiles):
        """
        Tokenize characters in smiles to integers
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
        """
        Converts SMILES strings to np.array sequence.
        """
        if isinstance(ids_b[0], bytes) and sys.version_info[
                0] != 2:  # Python 2.7 bytes and string are analogous
            ids_b = [
                TextCNNModel.convert_bytes_to_char(smiles) for smiles in ids_b
            ]
        smiles_seqs = [self.smiles_to_seq(smiles) for smiles in ids_b]
        smiles_seqs = np.vstack(smiles_seqs)
        return smiles_seqs
