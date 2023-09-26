import torch
import numpy as np
from deepchem.models.torch_models.torch_model import TorchModel
from deepchem.models.torch_models.layers import AtomicConvolutionModule
from deepchem.models.losses import L2Loss
from deepchem.metrics import to_one_hot

from typing import Callable, Optional, Sequence
from deepchem.utils.typing import OneOrMany, ActivationFn


class AtomConv(TorchModel):
    """
    Implements an Atomic Convolution Model.

    Implements the atomic convolutional networks as introduced in

    Gomes, Joseph, et al. "Atomic convolutional networks for predicting protein-ligand binding affinity." arXiv preprint arXiv:1703.10603 (2017).

    The atomic convolutional networks function as a variant of
    graph convolutions. The difference is that the "graph" here is
    the nearest neighbors graph in 3D space. The AtomicConvModel
    leverages these connections in 3D space to train models that
    learn to predict energetic state starting from the spatial
    geometry of the model.
    
    Example
    """

    def __init__(self,
                 n_tasks: int,
                 frag1_num_atoms: int = 70,
                 frag2_num_atoms: int = 634,
                 complex_num_atoms: int = 701,
                 max_num_neighbors: int = 12,
                 batch_size: int = 24,
                 atom_types: Sequence[float] = [
                     6, 7., 8., 9., 11., 12., 15., 16., 17., 20., 25., 30., 35.,
                     53., -1.
                 ],
                 radial: Sequence[Sequence[float]] = [[
                     1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0,
                     7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0
                 ], [0.0, 4.0, 8.0], [0.4]],
                 layer_sizes=[100],
                 weight_init_stddevs: OneOrMany[float] = 0.02,
                 bias_init_consts: OneOrMany[float] = 1.0,
                 weight_decay_penalty: float = 0.0,
                 weight_decay_penalty_type: str = "l2",
                 dropouts: OneOrMany[float] = 0.5,
                 activation_fns: OneOrMany[ActivationFn] = ['relu'],
                 residual: bool = False,
                 learning_rate=0.001,
                 **kwargs) -> None:
        """TorchModel wrapper for Atomic Convolutions

        Parameters
        ----------
        n_tasks: int
            number of tasks
        frag1_num_atoms: int
            Number of atoms in first fragment
        frag2_num_atoms: int
            Number of atoms in sec
        max_num_neighbors: int
            Maximum number of neighbors possible for an atom. Recall neighbors
            are spatial neighbors.
        atom_types: list
            List of atoms recognized by model. Atoms are indicated by their
            nuclear numbers.
        radial: list
            Radial parameters used in the atomic convolution transformation.
        layer_sizes: list
            the size of each dense layer in the network.  The length of
            this list determines the number of layers.
        weight_init_stddevs: list or float
            the standard deviation of the distribution to use for weight
            initialization of each layer.  The length of this list should
            equal len(layer_sizes).  Alternatively this may be a single
            value instead of a list, in which case the same value is used
            for every layer.
        bias_init_consts: list or float
            the value to initialize the biases in each layer to.  The
            length of this list should equal len(layer_sizes).
            Alternatively this may be a single value instead of a list, in
            which case the same value is used for every layer.
        weight_decay_penalty: float
            the magnitude of the weight decay penalty to use
        weight_decay_penalty_type: str
            the type of penalty to use for weight decay, either 'l1' or 'l2'
        dropouts: list or float
            the dropout probablity to use for each layer.  The length of this list should equal len(layer_sizes).
            Alternatively this may be a single value instead of a list, in which case the same value is used for every layer.
        activation_fns: list or object
            the Pytorch activation function to apply to each layer.  The length of this list should equal
            len(layer_sizes).  Alternatively this may be a single value instead of a list, in which case the
            same value is used for every layer.
        residual: bool
            if True, the model will be composed of pre-activation residual blocks instead
            of a simple stack of dense layers.
        learning_rate: float
            Learning rate for the model.
        
        """

        self.n_tasks = n_tasks
        self.complex_num_atoms = complex_num_atoms
        self.frag1_num_atoms = frag1_num_atoms
        self.frag2_num_atoms = frag2_num_atoms
        self.max_num_neighbors = max_num_neighbors
        self.batch_size = batch_size
        self.atom_types = atom_types

        self.model = AtomicConvolutionModule(
            n_tasks=n_tasks,
            frag1_num_atoms=frag1_num_atoms,
            frag2_num_atoms=frag2_num_atoms,
            complex_num_atoms=complex_num_atoms,
            max_num_neighbors=max_num_neighbors,
            batch_size=batch_size,
            atom_types=atom_types,
            radial=radial,
            layer_sizes=layer_sizes,
            weight_init_stddevs=weight_init_stddevs,
            bias_init_consts=bias_init_consts,
            dropouts=dropouts,
            activation_fns=activation_fns)

        regularization_loss: Optional[Callable]

        if weight_decay_penalty != 0:
            weights = [layer.weight for layer in self.model.layers]
            if weight_decay_penalty_type == 'l1':
                regularization_loss = lambda: weight_decay_penalty * torch.sum(  # noqa: E731
                    torch.stack([torch.abs(w).sum() for w in weights]))
            else:
                regularization_loss = lambda: weight_decay_penalty * torch.sum(  # noqa: E731
                    torch.stack([torch.square(w).sum() for w in weights]))
        else:
            regularization_loss = None

        loss = L2Loss()

        super(AtomConv, self).__init__(self.model,
                                       loss=loss,
                                       batch_size=batch_size,
                                       regularization_loss=regularization_loss,
                                       **kwargs)

    def default_generator(self,
                          dataset,
                          epochs=1,
                          mode='fit',
                          deterministic=True,
                          pad_batches=True):

        batch_size = self.batch_size

        def replace_atom_types(z):
            np.putmask(z, np.isin(z, list(self.atom_types), invert=True), -1)
            return z

        for epoch in range(epochs):
            print(epoch)
            for ind, (F_b, y_b, w_b, ids_b) in enumerate(
                    dataset.iterbatches(batch_size,
                                        deterministic=True,
                                        pad_batches=pad_batches)):

                N = self.complex_num_atoms
                N_1 = self.frag1_num_atoms
                N_2 = self.frag2_num_atoms
                M = self.max_num_neighbors

                batch_size = F_b.shape[0]
                num_features = F_b[0][0].shape[1]
                frag1_X_b = np.zeros((batch_size, N_1, num_features))
                for i in range(batch_size):
                    frag1_X_b[i] = F_b[i][0]

                frag2_X_b = np.zeros((batch_size, N_2, num_features))
                for i in range(batch_size):
                    frag2_X_b[i] = F_b[i][3]

                complex_X_b = np.zeros((batch_size, N, num_features))
                for i in range(batch_size):
                    complex_X_b[i] = F_b[i][6]

                frag1_Nbrs = np.zeros((batch_size, N_1, M))
                frag1_Z_b = np.zeros((batch_size, N_1))
                for i in range(batch_size):
                    z = replace_atom_types(F_b[i][2])
                    frag1_Z_b[i] = z
                frag1_Nbrs_Z = np.zeros((batch_size, N_1, M))
                for atom in range(N_1):
                    for i in range(batch_size):
                        atom_nbrs = F_b[i][1].get(atom, "")
                        frag1_Nbrs[i,
                                   atom, :len(atom_nbrs)] = np.array(atom_nbrs)
                        for j, atom_j in enumerate(atom_nbrs):
                            frag1_Nbrs_Z[i, atom, j] = frag1_Z_b[i, atom_j]

                frag2_Nbrs = np.zeros((batch_size, N_2, M))
                frag2_Z_b = np.zeros((batch_size, N_2))
                for i in range(batch_size):
                    z = replace_atom_types(F_b[i][5])
                    frag2_Z_b[i] = z
                frag2_Nbrs_Z = np.zeros((batch_size, N_2, M))
                for atom in range(N_2):
                    for i in range(batch_size):
                        atom_nbrs = F_b[i][4].get(atom, "")
                        frag2_Nbrs[i,
                                   atom, :len(atom_nbrs)] = np.array(atom_nbrs)
                        for j, atom_j in enumerate(atom_nbrs):
                            frag2_Nbrs_Z[i, atom, j] = frag2_Z_b[i, atom_j]

                complex_Nbrs = np.zeros((batch_size, N, M))
                complex_Z_b = np.zeros((batch_size, N))
                for i in range(batch_size):
                    z = replace_atom_types(F_b[i][8])
                    complex_Z_b[i] = z
                complex_Nbrs_Z = np.zeros((batch_size, N, M))
                for atom in range(N):
                    for i in range(batch_size):
                        atom_nbrs = F_b[i][7].get(atom, "")
                        complex_Nbrs[i, atom, :len(atom_nbrs)] = np.array(
                            atom_nbrs)
                        for j, atom_j in enumerate(atom_nbrs):
                            complex_Nbrs_Z[i, atom, j] = complex_Z_b[i, atom_j]

                inputs = [
                    frag1_X_b, frag1_Nbrs, frag1_Nbrs_Z, frag1_Z_b, frag2_X_b,
                    frag2_Nbrs, frag2_Nbrs_Z, frag2_Z_b, complex_X_b,
                    complex_Nbrs, complex_Nbrs_Z, complex_Z_b
                ]

                y_b = np.reshape(y_b, newshape=(batch_size, 1))
                yield (inputs, [y_b], [w_b])
