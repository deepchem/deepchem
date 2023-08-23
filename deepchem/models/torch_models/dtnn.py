from typing import List

import torch.nn as nn
import torch

from deepchem.models.torch_models import layers


class DTNN(nn.Module):
    """Deep Tensor Neural Networks

    DTNN is based on the many-body Hamiltonian concept, which is a fundamental principle in quantum mechanics.
    The DTNN recieves a molecule's distance matrix and membership of its atom from its Coulomb Matrix representation.
    Then, it iteratively refines the representation of each atom by considering its interactions with neighboring atoms.
    Finally, it predicts the energy of the molecule by summing up the energies of the individual atoms.

    In this class, we establish a sequential model for the Deep Tensor Neural Network (DTNN) [1]_.

    Examples
    --------
    >>> import os
    >>> import torch
    >>> from deepchem.models.torch_models import DTNN
    >>> from deepchem.data import SDFLoader
    >>> from deepchem.feat import CoulombMatrix
    >>> from deepchem.utils import batch_coulomb_matrix_features
    >>> # Get Data
    >>> model_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    >>> dataset_file = os.path.join(model_dir, 'tests/assets/qm9_mini.sdf')
    >>> TASKS = ["alpha", "homo"]
    >>> loader = SDFLoader(tasks=TASKS, featurizer=CoulombMatrix(29), sanitize=True)
    >>> data = loader.create_dataset(dataset_file, shard_size=100)
    >>> inputs = batch_coulomb_matrix_features(data.X)
    >>> atom_number, distance, atom_membership, distance_membership_i, distance_membership_j = inputs
    >>> inputs = [torch.tensor(atom_number).to(torch.int64),
    ...           torch.tensor(distance).to(torch.float32),
    ...           torch.tensor(atom_membership).to(torch.int64),
    ...           torch.tensor(distance_membership_i).to(torch.int64),
    ...           torch.tensor(distance_membership_j).to(torch.int64)]
    >>> n_tasks = data.y.shape[0]
    >>> model = DTNN(n_tasks)
    >>> pred = model(inputs)

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
            A list of tensors containing atom_number, distance,
            atom_membership, distance_membership_i, and distance_membership_j.


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
