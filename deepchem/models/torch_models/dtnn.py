import numpy as np

import torch
import torch.nn as nn

from deepchem.models.losses import L2Loss
from deepchem.models.torch_models import layers
from deepchem.models.torch_models import TorchModel

class DTNN(nn.Module):
    """Deep Tensor Neural Networks

    This class implements deep tensor neural networks as first defined in [1]_

    References
    ----------
    .. [1] Schütt, Kristof T., et al. "Quantum-chemical insights from deep
        tensor neural networks." Nature communications 8.1 (2017): 1-8.

    """
    def __init__(self,
             n_tasks,
             n_embedding=30,
             n_hidden=100,
             n_distance=100,
             distance_min=-1,
             distance_max=18,
             output_activation=True,
             mode="regression",
             dropout=0.0,
             **kwargs):
        """
        Parameters
        ----------
        n_tasks: int
            Number of tasks
        n_embedding: int, optional
            Number of features per atom.
        n_hidden: int, optional
            Number of features for each molecule after DTNNStep
        n_distance: int, optional
            granularity of distance matrix
            step size will be (distance_max-distance_min)/n_distance
        distance_min: float, optional
            minimum distance of atom pairs, default = -1 Angstorm
        distance_max: float, optional
            maximum distance of atom pairs, default = 18 Angstorm
        mode: str
            Only "regression" is currently supported.
        dropout: float
            the dropout probablity to use.

        """
        super(DTNN, self).__init__()
        if mode not in ['regression']:
            raise ValueError("Only 'regression' mode is currently supported")
        self.n_tasks = n_tasks
        self.n_embedding = n_embedding
        self.n_hidden = n_hidden
        self.n_distance = n_distance
        self.distance_min = distance_min
        self.distance_max = distance_max
        self.step_size = (distance_max - distance_min) / n_distance
        self.steps = np.array(
            [distance_min + i * self.step_size for i in range(n_distance)])
        self.steps = np.expand_dims(self.steps, 0)
        self.output_activation = output_activation
        self.mode = mode
        self.dropout = dropout

        # get DTNNEmbedding
        self.dtnn_embedding = layers.DTNNEmbedding(n_embedding=self.n_embedding)

        # get DTNNSteps
        self.dtnn_step = layers.DTNNStep(n_embedding=self.n_embedding, n_distance=self.n_distance)

        # get DTNNGather
        self.dtnn_gather = layers.DTNNGather(n_embedding=self.n_embedding, layer_sizes=[self.n_hidden], n_outputs=self.n_tasks, output_activation=self.output_activation)

    def forward(self, inputs):
        """
        atom_number,
        distance,
        atom_membership,
        distance_membership_i,
        distance_membership_j
        """
        dtnn_embedding = self.dtnn_embedding(inputs[0])
        if self.dropout > 0.0:
            dtnn_embedding = nn.Dropout(self.dropout)(dtnn_embedding)
        dtnn_step = self.dtnn_step([dtnn_embedding, inputs[1], inputs[3], inputs[4]])
        if self.dropout > 0.0:
            dtnn_step = nn.Dropout(self.dropout)(dtnn_step)
        dtnn_step = self.dtnn_step([dtnn_step, inputs[1], inputs[3], inputs[4]])
        if self.dropout > 0.0:
            dtnn_step = nn.Dropout(self.dropout)(dtnn_step)
        dtnn_gather = self.dtnn_gather([dtnn_step, inputs[2]])
        if self.dropout > 0.0:
            dtnn_gather = nn.Dropout(self.dropout)(dtnn_gather)
        output = nn.Linear(dtnn_gather.shape[-1], self.n_tasks)(dtnn_gather)
        return output
        
    
class DTNNModel(TorchModel):
    """Implements DTNN models for regression and classification.
    """

    def __init__(self,
                 n_tasks,
                 n_embedding=30,
                 n_hidden=100,
                 n_distance=100,
                 distance_min=-1,
                 distance_max=18,
                 output_activation=True,
                 mode="regression",
                 dropout=0.0,
                 **kwargs):
        """
        Parameters
        ----------
        n_tasks: int
            Number of tasks
        n_embedding: int, optional
            Number of features per atom.
        n_hidden: int, optional
            Number of features for each molecule after DTNNStep
        n_distance: int, optional
            granularity of distance matrix
            step size will be (distance_max-distance_min)/n_distance
        distance_min: float, optional
            minimum distance of atom pairs, default = -1 Angstorm
        distance_max: float, optional
            maximum distance of atom pairs, default = 18 Angstorm
        mode: str
            Only "regression" is currently supported.
        dropout: float
            the dropout probablity to use.

        """
        model = DTNN(
            n_tasks=n_tasks,
            n_embedding=n_embedding,
            n_hidden=n_hidden,
            n_distance=n_distance,
            distance_min=distance_min,
            distance_max=distance_max,
            output_activation=output_activation,
            mode=mode,
            dropout=dropout,
            **kwargs)
        if mode not in ['regression']:
            raise ValueError("Only 'regression' mode is currently supported")
        super(DTNNModel, self).__init__(model, L2Loss(), ["prediction"], **kwargs)
        
    def compute_features_on_batch(self, X_b):
        """Computes the values for different Feature Layers on given batch

        A tf.py_func wrapper is written around this when creating the
        input_fn for tf.Estimator

        """
        distance = []
        atom_membership = []
        distance_membership_i = []
        distance_membership_j = []
        num_atoms = list(map(sum, X_b.astype(bool)[:, :, 0]))
        atom_number = [
            np.round(
                np.power(2 * np.diag(X_b[i, :num_atoms[i], :num_atoms[i]]),
                         1 / 2.4)).astype(int) for i in range(len(num_atoms))
        ]
        start = 0
        for im, molecule in enumerate(atom_number):
            distance_matrix = np.outer(
                molecule, molecule) / X_b[im, :num_atoms[im], :num_atoms[im]]
            np.fill_diagonal(distance_matrix, -100)
            distance.append(np.expand_dims(distance_matrix.flatten(), 1))
            atom_membership.append([im] * num_atoms[im])
            membership = np.array([np.arange(num_atoms[im])] * num_atoms[im])
            membership_i = membership.flatten(order='F')
            membership_j = membership.flatten()
            distance_membership_i.append(membership_i + start)
            distance_membership_j.append(membership_j + start)
            start = start + num_atoms[im]

        atom_number = np.concatenate(atom_number).astype(np.int32)
        distance = np.concatenate(distance, axis=0)
        gaussian_dist = np.exp(-np.square(distance - self.steps) /
                               (2 * self.step_size**2))
        gaussian_dist = gaussian_dist.astype(np.float32)
        atom_mem = np.concatenate(atom_membership).astype(np.int32)
        dist_mem_i = np.concatenate(distance_membership_i).astype(np.int32)
        dist_mem_j = np.concatenate(distance_membership_j).astype(np.int32)

        features = [
            atom_number, gaussian_dist, atom_mem, dist_mem_i, dist_mem_j
        ]

        return features

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
                yield (self.compute_features_on_batch(X_b), [y_b], [w_b])
