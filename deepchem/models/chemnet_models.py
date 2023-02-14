"""
Implementation of Smiles2Vec and ChemCeption models as part of the ChemNet
transfer learning protocol.
"""

__author__ = "Vignesh Ram Somnath"
__license__ = "MIT"

import numpy as np
import tensorflow as tf

from typing import Dict
from deepchem.data.datasets import pad_batch
from deepchem.models import KerasModel
from deepchem.models.losses import L2Loss, SoftmaxCrossEntropy, SigmoidCrossEntropy
from deepchem.metrics import to_one_hot
from deepchem.models import chemnet_layers
from tensorflow.keras.layers import Input, Dense, Reshape, Softmax, Activation
from tensorflow.keras.layers import Conv1D, GRU, LSTM, Bidirectional
from tensorflow.keras.layers import GlobalAveragePooling2D

DEFAULT_INCEPTION_BLOCKS = {"A": 3, "B": 3, "C": 3}

INCEPTION_DICT = {
    "A": chemnet_layers.InceptionResnetA,
    "B": chemnet_layers.InceptionResnetB,
    "C": chemnet_layers.InceptionResnetC
}

RNN_DICT = {"GRU": GRU, "LSTM": LSTM}


class Smiles2Vec(KerasModel):
    """
    Implements the Smiles2Vec model, that learns neural representations of SMILES
    strings which can be used for downstream tasks.

    The model is based on the description in Goh et al., "SMILES2vec: An
    Interpretable General-Purpose Deep Neural Network for Predicting Chemical
    Properties" (https://arxiv.org/pdf/1712.02034.pdf). The goal here is to take
    SMILES strings as inputs, turn them into vector representations which can then
    be used in predicting molecular properties.

    The model consists of an Embedding layer that retrieves embeddings for each
    character in the SMILES string. These embeddings are learnt jointly with the
    rest of the model. The output from the embedding layer is a tensor of shape
    (batch_size, seq_len, embedding_dim). This tensor can optionally be fed
    through a 1D convolutional layer, before being passed to a series of RNN cells
    (optionally bidirectional). The final output from the RNN cells aims
    to have learnt the temporal dependencies in the SMILES string, and in turn
    information about the structure of the molecule, which is then used for
    molecular property prediction.

    In the paper, the authors also train an explanation mask to endow the model
    with interpretability and gain insights into its decision making. This segment
    is currently not a part of this implementation as this was
    developed for the purpose of investigating a transfer learning protocol,
    ChemNet (which can be found at https://arxiv.org/abs/1712.02734).
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
        """
        Parameters
        ----------
        char_to_idx: dict,
            char_to_idx contains character to index mapping for SMILES characters
        embedding_dim: int, default 50
            Size of character embeddings used.
        use_bidir: bool, default True
            Whether to use BiDirectional RNN Cells
        use_conv: bool, default True
            Whether to use a conv-layer
        kernel_size: int, default 3
            Kernel size for convolutions
        filters: int, default 192
            Number of filters
        strides: int, default 1
            Strides used in convolution
        rnn_sizes: list[int], default [224, 384]
            Number of hidden units in the RNN cells
        mode: str, default regression
            Whether to use model for regression or classification
        """

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

        model, loss, output_types = self._build_graph()
        super(Smiles2Vec, self).__init__(model=model,
                                         loss=loss,
                                         output_types=output_types,
                                         **kwargs)

    def _build_graph(self):
        """Build the model."""
        smiles_seqs = Input(dtype=tf.int32,
                            shape=(self.max_seq_len,),
                            name='Input')
        rnn_input = tf.keras.layers.Embedding(
            input_dim=len(self.char_to_idx),
            output_dim=self.embedding_dim)(smiles_seqs)

        if self.use_conv:
            rnn_input = Conv1D(filters=self.filters,
                               kernel_size=self.kernel_size,
                               strides=self.strides,
                               activation=tf.nn.relu,
                               name='Conv1D')(rnn_input)

        rnn_embeddings = rnn_input
        for idx, rnn_type in enumerate(self.rnn_types[:-1]):
            rnn_layer = RNN_DICT[rnn_type]
            layer = rnn_layer(units=self.rnn_sizes[idx], return_sequences=True)
            if self.use_bidir:
                layer = Bidirectional(layer)

            rnn_embeddings = layer(rnn_embeddings)

        # Last layer sequences not returned.
        layer = RNN_DICT[self.rnn_types[-1]](units=self.rnn_sizes[-1])
        if self.use_bidir:
            layer = Bidirectional(layer)
        rnn_embeddings = layer(rnn_embeddings)

        if self.mode == "classification":
            logits = Dense(self.n_tasks * self.n_classes)(rnn_embeddings)
            logits = Reshape((self.n_tasks, self.n_classes))(logits)
            if self.n_classes == 2:
                output = Activation(activation='sigmoid')(logits)
                loss = SigmoidCrossEntropy()
            else:
                output = Softmax()(logits)
                loss = SoftmaxCrossEntropy()
            outputs = [output, logits]
            output_types = ['prediction', 'loss']

        else:
            output = Dense(self.n_tasks * 1, name='Dense')(rnn_embeddings)
            output = Reshape((self.n_tasks, 1), name='Reshape')(output)
            outputs = [output]
            output_types = ['prediction']
            loss = L2Loss()

        model = tf.keras.Model(inputs=[smiles_seqs], outputs=outputs)
        return model, loss, output_types

    def default_generator(self,
                          dataset,
                          epochs=1,
                          mode='fit',
                          deterministic=True,
                          pad_batches=True):
        for epoch in range(epochs):
            for (X_b, y_b, w_b,
                 ids_b) in dataset.iterbatches(batch_size=self.batch_size,
                                               deterministic=deterministic,
                                               pad_batches=pad_batches):
                if self.mode == 'classification':
                    y_b = to_one_hot(y_b.flatten(), self.n_classes).reshape(
                        -1, self.n_tasks, self.n_classes)
                yield ([X_b], [y_b], [w_b])


class ChemCeption(KerasModel):
    """
    Implements the ChemCeption model that leverages the representational capacities
    of convolutional neural networks (CNNs) to predict molecular properties.

    The model is based on the description in Goh et al., "Chemception: A Deep
    Neural Network with Minimal Chemistry Knowledge Matches the Performance of
    Expert-developed QSAR/QSPR Models" (https://arxiv.org/pdf/1706.06689.pdf).
    The authors use an image based representation of the molecule, where pixels
    encode different atomic and bond properties. More details on the image repres-
    entations can be found at https://arxiv.org/abs/1710.02238

    The model consists of a Stem Layer that reduces the image resolution for the
    layers to follow. The output of the Stem Layer is followed by a series of
    Inception-Resnet blocks & a Reduction layer. Layers in the Inception-Resnet
    blocks process image tensors at multiple resolutions and use a ResNet style
    skip-connection, combining features from different resolutions. The Reduction
    layers reduce the spatial extent of the image by max-pooling and 2-strided
    convolutions. More details on these layers can be found in the ChemCeption
    paper referenced above. The output of the final Reduction layer is subject to
    a Global Average Pooling, and a fully-connected layer maps the features to
    downstream outputs.

    In the ChemCeption paper, the authors perform real-time image augmentation by
    rotating images between 0 to 180 degrees. This can be done during model
    training by setting the augment argument to True.
    """

    def __init__(self,
                 img_spec: str = "std",
                 img_size: int = 80,
                 base_filters: int = 16,
                 inception_blocks: Dict = DEFAULT_INCEPTION_BLOCKS,
                 n_tasks: int = 10,
                 n_classes: int = 2,
                 augment: bool = False,
                 mode: str = "regression",
                 **kwargs):
        """
        Parameters
        ----------
        img_spec: str, default std
            Image specification used
        img_size: int, default 80
            Image size used
        base_filters: int, default 16
            Base filters used for the different inception and reduction layers
        inception_blocks: dict,
            Dictionary containing number of blocks for every inception layer
        n_tasks: int, default 10
            Number of classification or regression tasks
        n_classes: int, default 2
            Number of classes (used only for classification)
        augment: bool, default False
            Whether to augment images
        mode: str, default regression
            Whether the model is used for regression or classification
        """
        if img_spec == "engd":
            self.input_shape = (img_size, img_size, 4)
        elif img_spec == "std":
            self.input_shape = (img_size, img_size, 1)
        self.base_filters = base_filters
        self.inception_blocks = inception_blocks
        self.n_tasks = n_tasks
        self.n_classes = n_classes
        self.mode = mode
        self.augment = augment

        model, loss, output_types = self._build_graph()
        super(ChemCeption, self).__init__(model=model,
                                          loss=loss,
                                          output_types=output_types,
                                          **kwargs)

    def _build_graph(self):
        smile_images = Input(shape=self.input_shape)
        stem = chemnet_layers.Stem(self.base_filters)(smile_images)

        inceptionA_out = self.build_inception_module(inputs=stem, type="A")
        reductionA_out = chemnet_layers.ReductionA(
            self.base_filters)(inceptionA_out)

        inceptionB_out = self.build_inception_module(inputs=reductionA_out,
                                                     type="B")
        reductionB_out = chemnet_layers.ReductionB(
            self.base_filters)(inceptionB_out)

        inceptionC_out = self.build_inception_module(inputs=reductionB_out,
                                                     type="C")
        avg_pooling_out = GlobalAveragePooling2D()(inceptionC_out)

        if self.mode == "classification":
            logits = Dense(self.n_tasks * self.n_classes)(avg_pooling_out)
            logits = Reshape((self.n_tasks, self.n_classes))(logits)
            if self.n_classes == 2:
                output = Activation(activation='sigmoid')(logits)
                loss = SigmoidCrossEntropy()
            else:
                output = Softmax()(logits)
                loss = SoftmaxCrossEntropy()
            outputs = [output, logits]
            output_types = ['prediction', 'loss']

        else:
            output = Dense(self.n_tasks * 1)(avg_pooling_out)
            output = Reshape((self.n_tasks, 1))(output)
            outputs = [output]
            output_types = ['prediction']
            loss = L2Loss()

        model = tf.keras.Model(inputs=[smile_images], outputs=outputs)
        return model, loss, output_types

    def build_inception_module(self, inputs, type="A"):
        """Inception module is a series of inception layers of similar type. This
        function builds that."""
        num_blocks = self.inception_blocks[type]
        inception_layer = INCEPTION_DICT[type]
        output = inputs
        for block in range(num_blocks):
            output = inception_layer(self.base_filters,
                                     int(inputs.shape[-1]))(output)
        return output

    def default_generator(self,
                          dataset,
                          epochs=1,
                          mode='fit',
                          deterministic=True,
                          pad_batches=True):
        for epoch in range(epochs):
            if mode == "predict" or (not self.augment):
                for (X_b, y_b, w_b,
                     ids_b) in dataset.iterbatches(batch_size=self.batch_size,
                                                   deterministic=deterministic,
                                                   pad_batches=pad_batches):
                    if self.mode == 'classification':
                        y_b = to_one_hot(y_b.flatten(), self.n_classes).reshape(
                            -1, self.n_tasks, self.n_classes)
                    yield ([X_b], [y_b], [w_b])

            else:
                if not pad_batches:
                    n_samples = dataset.X.shape[0]
                else:
                    n_samples = dataset.X.shape[0] + (
                        self.batch_size -
                        (dataset.X.shape[0] % self.batch_size))

                n_batches = 0
                image_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
                    rotation_range=180)
                for (X_b, y_b, w_b) in image_data_generator.flow(
                        dataset.X,
                        dataset.y,
                        sample_weight=dataset.w,
                        shuffle=not deterministic,
                        batch_size=self.batch_size):
                    if pad_batches:
                        ids_b = np.arange(X_b.shape[0])
                        X_b, y_b, w_b, _ = pad_batch(self.batch_size, X_b, y_b,
                                                     w_b, ids_b)
                    n_batches += 1
                    if n_batches > n_samples / self.batch_size:
                        # This is needed because ImageDataGenerator does infinite looping
                        break
                    if self.mode == "classification":
                        y_b = to_one_hot(y_b.flatten(), self.n_classes).reshape(
                            -1, self.n_tasks, self.n_classes)
                    yield ([X_b], [y_b], [w_b])
