import numpy as np
from typing import List

import torch.nn as nn
import torch

from deepchem.models.losses import L2Loss
from deepchem.models.torch_models import layers
from deepchem.models.torch_models import TorchModel
from deepchem.data.datasets import Dataset


class DTNN(nn.Module):
    """Deep Tensor Neural Networks

    In this class, we establish a sequential model for the Deep Tensor Neural Network (DTNN) [1]_.

    References
    ----------
    .. [1] Schütt, Kristof T., et al. "Quantum-chemical insights from deep
        tensor neural networks." Nature communications 8.1 (2017): 1-8.

    """

    def __init__(self,
                 n_tasks: int,
                 n_embedding: int = 30,
                 n_hidden: int = 100,
                 n_distance: int = 100,
                 distance_min: float = -1,
                 distance_max: float = 18,
                 output_activation: bool = True,
                 mode: str = "regression",
                 dropout: float = 0.0,
                 n_steps: int = 2):
        """
        Parameters
        ----------
        n_tasks: int
            Number of tasks
        n_embedding: int (default 30)
            Number of features per atom.
        n_hidden: int (default 100)
            Number of features for each molecule after DTNNStep
        n_distance: int (default 100)
            granularity of distance matrix
            step size will be (distance_max-distance_min)/n_distance
        distance_min: float (default -1)
            minimum distance of atom pairs (in Angstrom)
        distance_max: float (default 18)
            maximum distance of atom pairs (in Angstrom)
        output_activation: bool (default True)
            determines whether an activation function should be apply  to its output.
        mode: str (default "regression")
            Only "regression" is currently supported.
        dropout: float (default 0.0)
            the dropout probablity to use.
        n_steps: int (default 2)
            Number of DTNNStep Layers to use.

        """
        super(DTNN, self).__init__()
        self.n_tasks = n_tasks
        self.n_embedding = n_embedding
        self.n_hidden = n_hidden
        self.n_distance = n_distance
        self.distance_min = distance_min
        self.distance_max = distance_max
        self.output_activation = output_activation
        self.mode = mode
        self.dropout = dropout
        self.n_steps = n_steps

        # get DTNNEmbedding
        self.dtnn_embedding = layers.DTNNEmbedding(n_embedding=self.n_embedding)

        # get DTNNSteps
        self.dtnn_step = nn.ModuleList()
        for i in range(self.n_steps):
            self.dtnn_step.append(
                layers.DTNNStep(n_embedding=self.n_embedding,
                                n_distance=self.n_distance))

        # get DTNNGather
        self.dtnn_gather = layers.DTNNGather(
            n_embedding=self.n_embedding,
            layer_sizes=[self.n_hidden],
            n_outputs=self.n_tasks,
            output_activation=self.output_activation)

        # get Final Linear Layer
        self.linear = nn.LazyLinear(self.n_tasks)

    def forward(self, inputs: List[torch.Tensor]):
        """
        Parameters
        ----------
        inputs: List
            List of Tensors containing. (In the specified order)
            - atom_number
            - distance
            - atom_membership
            - distance_membership_i
            - distance_membership_j

        Returns
        -------
        output: torch.Tensor
            Predictions of the Molecular Energy.

        """
        dtnn_embedding = self.dtnn_embedding(inputs[0])

        for i in range(self.n_steps):
            dtnn_embedding = nn.Dropout(self.dropout)(dtnn_embedding)
            dtnn_embedding = self.dtnn_step[i](
                [dtnn_embedding, inputs[1], inputs[3], inputs[4]])

        dtnn_step = nn.Dropout(self.dropout)(dtnn_embedding)
        dtnn_gather = self.dtnn_gather([dtnn_step, inputs[2]])

        dtnn_gather = nn.Dropout(self.dropout)(dtnn_gather)
        output = self.linear(dtnn_gather)
        return output


class DTNNModel(TorchModel):
    """Implements DTNN models for regression.

    This class implements the Directed Message Passing Neural Network (D-MPNN) [1]_.

    Examples
    --------
    >>> import os
    >>> import numpy as np
    >>> from deepchem.data import SDFLoader
    >>> from deepchem.feat import CoulombMatrix
    >>> from deepchem.models.torch_models import DTNNModel
    >>> model_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    >>> dataset_file = os.path.join(model_dir, 'tests/assets/qm9_mini.sdf')
    >>> TASKS = ["alpha", "homo"]
    >>> loader = SDFLoader(tasks=TASKS, featurizer=CoulombMatrix(29), sanitize=True)
    >>> data = loader.create_dataset(dataset_file, shard_size=100)
    >>> n_tasks = data.y.shape[1]
    >>> model = DTNNModel(n_tasks,
    ...               n_embedding=20,
    ...               n_distance=100,
    ...               learning_rate=1.0,
    ...               mode="regression")
    >>> loss = model.fit(data, nb_epoch=250)
    >>> pred = model.predict(data)

    References
    ----------
    .. [1] Schütt, Kristof T., et al. "Quantum-chemical insights from deep
        tensor neural networks." Nature communications 8.1 (2017): 1-8.

    """

    def __init__(self,
                 n_tasks: int,
                 n_embedding: int = 30,
                 n_hidden: int = 100,
                 n_distance: int = 100,
                 distance_min: float = -1,
                 distance_max: float = 18,
                 output_activation: bool = True,
                 mode: str = "regression",
                 dropout: float = 0.0,
                 n_steps: int = 2,
                 **kwargs):
        """
        Parameters
        ----------
        n_tasks: int
            Number of tasks
        n_embedding: int (default 30)
            Number of features per atom.
        n_hidden: int (default 100)
            Number of features for each molecule after DTNNStep
        n_distance: int (default 100)
            granularity of distance matrix
            step size will be (distance_max-distance_min)/n_distance
        distance_min: float (default -1)
            minimum distance of atom pairs (in Angstrom)
        distance_max: float (default = 18)
            maximum distance of atom pairs (in Angstrom)
        output_activation: bool (default True)
            determines whether an activation function should be apply  to its output.
        mode: str (default "regression")
            Only "regression" is currently supported.
        dropout: float (default 0.0)
            the dropout probablity to use.
        n_steps: int (default 2)
            Number of DTNNStep Layers to use.

        """
        if dropout < 0 or dropout > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(dropout))
        model = DTNN(n_tasks=n_tasks,
                     n_embedding=n_embedding,
                     n_hidden=n_hidden,
                     n_distance=n_distance,
                     distance_min=distance_min,
                     distance_max=distance_max,
                     output_activation=output_activation,
                     mode=mode,
                     dropout=dropout,
                     n_steps=n_steps)
        if mode not in ['regression']:
            raise ValueError("Only 'regression' mode is currently supported")
        super(DTNNModel, self).__init__(model, L2Loss(), ["prediction"],
                                        **kwargs)

    def default_generator(self,
                          dataset: Dataset,
                          epochs: int = 1,
                          mode: str = 'fit',
                          deterministic: bool = True,
                          pad_batches: bool = True):
        """Create a generator that iterates batches for a dataset.

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

        Returns
        -------
        a generator that iterates batches, each represented as a tuple of lists:
        ([inputs], [outputs], [weights])

        """
        for epoch in range(epochs):
            for (X_b, y_b, w_b,
                 ids_b) in dataset.iterbatches(batch_size=self.batch_size,
                                               deterministic=deterministic,
                                               pad_batches=pad_batches):
                yield (_compute_features_on_batch(X_b, self.model.distance_max,
                                                  self.model.distance_min,
                                                  self.model.n_distance), [y_b],
                       [w_b])


def _compute_features_on_batch(X_b: np.ndarray,
                               distance_max: float = -1,
                               distance_min: float = 18,
                               n_distance: int = 100):
    """Computes the values for different Feature Layers on given batch.

    Parameters
    ----------
    X_b: np.ndarray
        It is a 3d Matrix containing information of each the atom's ionic interaction with other atoms in the molecule.
    distance_min: float (default -1)
        minimum distance of atom pairs (in Angstrom)
    distance_max: float (default = 18)
        maximum distance of atom pairs (in Angstrom)
    n_distance: int (default 100)
        granularity of distance matrix
        step size will be (distance_max-distance_min)/n_distance

    Computed Features
    -----------------
    atom_number: np.ndarray
        Atom numbers are assigned to each atom based on their atomic properties.
        The atomic numbers are derived from the periodic table of elements.
        For example, hydrogen -> 1, carbon -> 6, and oxygen -> 8.
    gaussian_dist: np.ndarray
        Gaussian distance refers to the method of representing the pairwise distances between atoms in a molecule using Gaussian functions.
        The Gaussian distance is calculated using the Euclidean distance between the Cartesian coordinates of two atoms.
        The distance value is then passed through a Gaussian function, which transforms it into a continuous value.
    atom_mem: np.ndarray
        Atom membership refers to the binary representation of whether an atom belongs to a specific group or property within a molecule.
        It allows the model to incorporate domain-specific information and enhance its understanding of the molecule's properties and interactions.
    dist_mem_i: np.ndarray
        Distance membership i are utilized to encode spatial information and capture the influence of atom distances on the properties and interactions within a molecule.
        The inner membership function assigns higher values to atoms that are closer to the atoms' interaction region, thereby emphasizing the impact of nearby atoms.
    dist_mem_j: np.ndarray
        It captures the long-range effects and influences between atoms that are not in direct proximity but still contribute to the overall molecular properties.
        Distance membership j are utilized to encode spatial information and capture the influence of atom distances on the properties and interactions outside a molecule.
        The outer membership function assigns higher values to atoms that are farther to the atoms' interaction region, thereby emphasizing the impact of farther atoms.

    """
    distance = []
    atom_membership = []
    distance_membership_i = []
    distance_membership_j = []

    # Calculation of Step Size and steps
    step_size = (distance_max - distance_min) / n_distance
    steps = np.array([distance_min + i * step_size for i in range(n_distance)])
    steps = np.expand_dims(steps, 0)

    # Number of atoms per molecule is calculated by counting all the non zero elements(numbers) of every molecule.
    num_atoms = list(map(sum, X_b.astype(bool)[:, :, 0]))

    # It loops over the molecules in the Coulomb matrix and rounds the square root of the diagonal of each molecule to the nearest integer.
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

    # Calculates the Gaussian Distance by passing distance by a gaussian function.
    gaussian_dist = np.exp(-np.square(distance - steps) / (2 * step_size**2))
    gaussian_dist = gaussian_dist.astype(np.float64)
    atom_mem = np.concatenate(atom_membership).astype(np.int64)
    dist_mem_i = np.concatenate(distance_membership_i).astype(np.int64)
    dist_mem_j = np.concatenate(distance_membership_j).astype(np.int64)
    features = [atom_number, gaussian_dist, atom_mem, dist_mem_i, dist_mem_j]
    return features
