import numpy as np
import tensorflow as tf
from deepchem.data import NumpyDataset
from deepchem.feat import CircularFingerprint
from deepchem.models import KerasModel
from deepchem.models.losses import HingeLoss
from tensorflow.keras.layers import Input, Dense, Dropout, Lambda


class ScScoreModel(KerasModel):
    """
    The SCScore model is a neural network model based on the work of Coley et al. [1]_ that predicts the synthetic complexity score (SCScore) of molecules and correlates it with the expected number of reaction steps required to produce the given target molecule.
    It is trained on a dataset of over 12 million reactions from the Reaxys database to impose a pairwise inequality constraint enforcing that on average the products of published chemical reactions should be more synthetically complex than their corresponding reactants.
    The learned metric (SCScore) exhibits highly desirable nonlinear behavior, particularly in recognizing increases in synthetic complexity throughout a number of linear synthetic routes.
    The SCScore model can accurately predict the synthetic complexity of a variety of molecules, including both drug-like and natural product molecules.
    SCScore has the potential to be a valuable tool for chemists who are working on drug discovery and other areas of chemistry.

    The learned metric (SCScore) exhibits highly desirable nonlinear behavior, particularly in recognizing increases in synthetic complexity throughout a number of linear synthetic routes.

    Our model uses hingeloss instead of the shifted relu loss as in the supplementary material [2]_ provided by the author.
    This could cause differentiation issues with compounds that are "close" to each other in "complexity".

    References
    ----------
    .. [1] Coley, C. W., Rogers, L., Green, W., & Jensen, K. F. (2018). "SCScore: Synthetic Complexity Learned from a Reaction Corpus". Journal of Chemical Information and Modeling, 58(2), 252-261. https://doi.org/10.1021/acs.jcim.7b00622

    .. [2] Coley, C. W., Rogers, L., Green, W., & Jensen, K. F. (2018). Supplementary material to "SCScore: Synthetic Complexity Learned from a Reaction Corpus". Journal of Chemical Information and Modeling, 58(2), 252-261. https://github.com/connorcoley/scscore
    """

    def __init__(self,
                 n_features,
                 layer_sizes=[300, 300, 300],
                 dropouts=0.0,
                 **kwargs):
        """
        Parameters
        ----------
        n_features: int
            number of features per molecule
        layer_sizes: list of int
            size of each hidden layer
        dropouts: int
            droupout to apply to each hidden layer
        kwargs
            This takes all kwards as TensorGraph
        """
        self.n_features = n_features
        self.layer_sizes = layer_sizes
        self.dropout = dropouts

        m1_features = Input(shape=(self.n_features,))
        m2_features = Input(shape=(self.n_features,))
        prev_layer1 = m1_features
        prev_layer2 = m2_features
        for layer_size in self.layer_sizes:
            layer = Dense(layer_size, activation=tf.nn.relu)
            prev_layer1 = layer(prev_layer1)
            prev_layer2 = layer(prev_layer2)
            if self.dropout > 0.0:
                prev_layer1 = Dropout(rate=self.dropout)(prev_layer1)
                prev_layer2 = Dropout(rate=self.dropout)(prev_layer2)

        readout_layer = Dense(1)
        readout_m1 = readout_layer(prev_layer1)
        readout_m2 = readout_layer(prev_layer2)
        outputs = [
            Lambda(lambda x: tf.sigmoid(x) * 4 + 1)(readout_m1),
            Lambda(lambda x: tf.sigmoid(x) * 4 + 1)(readout_m2),
            Lambda(lambda x: x[0] - x[1])([readout_m1, readout_m2])
        ]
        output_types = ['prediction', 'prediction', 'loss']
        model = tf.keras.Model(inputs=[m1_features, m2_features],
                               outputs=outputs)
        super(ScScoreModel, self).__init__(model,
                                           HingeLoss(),
                                           output_types=output_types,
                                           **kwargs)

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
                yield ([X_b[:, 0], X_b[:, 1]], [y_b], [w_b])

    def predict_mols(self, mols):
        featurizer = CircularFingerprint(size=self.n_features,
                                         radius=2,
                                         chiral=True)
        features = np.expand_dims(featurizer.featurize(mols), axis=1)
        features = np.concatenate([features, features], axis=1)
        ds = NumpyDataset(features, None, None, None)
        return self.predict(ds)[0][:, 0]
