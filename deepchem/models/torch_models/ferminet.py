"""
Implementation of the Ferminet class in pytorch
"""

from typing import List, Optional, Any, Tuple
# import torch.nn as nn
from rdkit import Chem
import numpy as np
from deepchem.utils.molecule_feature_utils import ALLEN_ELECTRONEGATIVTY
from deepchem.utils.geometry_utils import compute_pairwise_distances

# from deepchem.models.torch_models import TorchModel
# import deepchem.models.optimizers as optim
from deepchem.utils.electron_sampler import ElectronSampler

# TODO look for the loss function(Hamiltonian)


def test_f(x: np.ndarray) -> np.ndarray:
    # dummy function which can be passed as the parameter f. f gives the log probability
    # TODO replace this function with forward pass of the model in future
    return 2 * np.log(np.random.uniform(low=0, high=1.0, size=np.shape(x)[0]))


class Ferminet:
    """A deep-learning based Variational Monte Carlo method for calculating the ab-initio
solution of a many-electron system.

    This model aims to calculate the ground state energy of a multi-electron system
    using a baseline solution as the Hartree-Fock. An MCMC technique is used to sample
    electrons and DNNs are used to caluclate the square magnitude of the wavefunction,
    in which electron-electron repulsions also are included in the calculation(in the
    form of Jastrow factor envelopes). The model requires only the nucleus' coordinates
    as input.

    This method is based on the following paper:

    Spencer, James S., et al. Better, Faster Fermionic Neural Networks. arXiv:2011.07125,
    arXiv, 13 Nov. 2020. arXiv.org, http://arxiv.org/abs/2011.07125.
    """

    def __init__(
        self,
        nucleon_coordinates: List[List],
        spin: float,
        charge: int,
        seed: Optional[int] = None,
        batch_no: int = 10,
    ):
        """
        Parameters:
        -----------
        nucleon_coordinates:  List[List]
            A list containing nucleon coordinates as the values with the keys as the element's symbol.
        spin: float
            The total spin of the molecule system.
        charge: int
            The total charge of the molecule system.
        seed: int, optional (default None)
            Random seed to use for electron initialization.
        batch_no: int, optional (default 10)
            Number of batches of the electron's positions to be initialized.

        """
        # super(Ferminet, self).__init__()

        self.nucleon_coordinates = nucleon_coordinates
        self.seed = seed
        self.batch_no = batch_no
        self.spin = spin
        self.ion_charge = charge

    def prepare_input_stream(self,) -> Tuple[Any, Any, Any, Any]:
        """Prepares the one-electron and two-electron input stream for the model.

        Returns:
        --------
        one_electron_up: numpy.ndarray
            numpy array containing one-electron coordinates and distances for the up spin electrons.
        one_electron_down: numpy.ndarray
            numpy array containing one-electron coordinates and distances for the down spin electrons
        two_electron_up: numpy.ndarray
            numpy array containing two-electron coordinates and distances for the up spin electrons
        two_electron_down: numpy.ndarray
            numpy array containing two-electron coordinates and distances for the down spin electrons
        """

        no_electrons = []
        nucleons = []
        electronegativity = []

        table = Chem.GetPeriodicTable()
        index = 0
        for i in self.nucleon_coordinates:
            atomic_num = table.GetAtomicNumber(i[0])
            electronegativity.append([index, ALLEN_ELECTRONEGATIVTY[i[0]]])
            no_electrons.append([atomic_num])
            nucleons.append(i[1])
            index += 1

        self.electron_no: np.ndarray = np.array(no_electrons)
        self.charge: np.ndarray = self.electron_no.reshape(
            np.shape(self.electron_no)[0],)
        self.nucleon_pos: np.ndarray = np.array(nucleons)
        electro_neg = np.array(electronegativity)
        self.inter_atom: np.ndarray = compute_pairwise_distances(
            self.nucleon_pos, self.nucleon_pos)

        if np.sum(self.electron_no) < self.ion_charge:
            raise ValueError("Given charge is not initializable")

        # Initialization for ionic molecules
        if self.ion_charge != 0:
            if len(nucleons
                  ) == 1:  # for an atom, directly the charge is applied
                self.electron_no[0][0] -= self.ion_charge
            else:  # for a multiatomic molecule, the most electronegative atom gets a charge of -1 and vice versa. The remaining charges are assigned in terms of decreasing(for anionic charge) and increasing(for cationic charge) electronegativity.
                electro_neg = electro_neg[electro_neg[:, 1].argsort()]
                if self.ion_charge > 0:
                    for iter in range(self.ion_charge):
                        self.electron_no[int(electro_neg[iter][0])][0] -= 1
                else:
                    for iter in range(-self.ion_charge):
                        self.electron_no[int(electro_neg[-1 - iter][0])][0] += 1

        total_electrons = np.sum(self.electron_no)
        self.up_spin = (total_electrons + 2 * self.spin) // 2
        self.down_spin = (total_electrons - 2 * self.spin) // 2

        self.molecule: ElectronSampler = ElectronSampler(
            batch_no=self.batch_no,
            central_value=self.nucleon_pos,
            seed=self.seed,
            f=test_f,
            steps=1000)  # sample the electrons using the electron sampler
        self.molecule.gauss_initialize_position(
            self.electron_no)  # initialize the position of the electrons

        one_electron_vector = self.molecule.x - self.nucleon_pos

        shape = np.shape(self.molecule.x)
        two_electron_vector = self.molecule.x.reshape(
            [shape[0], 1, shape[1], 3]) - self.molecule.x

        one_electron_vector = one_electron_vector[0, :, :, :]
        two_electron_vector = two_electron_vector[0, :, :, :]

        self.one_electron_distance: np.ndarray = np.linalg.norm(
            one_electron_vector, axis=-1)
        self.two_electron_distance: np.ndarray = np.linalg.norm(
            two_electron_vector, axis=-1)

        # concatenating distance and vectors arrays
        one_shape = np.shape(self.one_electron_distance)
        one_distance = self.one_electron_distance.reshape(
            1, one_shape[0], one_shape[1], 1)
        one_electron = np.block([one_electron_vector, one_distance])
        two_shape = np.shape(self.two_electron_distance)
        two_distance = self.two_electron_distance.reshape(
            1, two_shape[0], two_shape[1], 1)
        two_electron = np.block([two_electron_vector, two_distance])

        one_electron_up = one_electron[:, :self.up_spin, :]
        one_electron_down = one_electron[:, self.up_spin:, :]

        two_electron_up = two_electron[:, :self.up_spin, :]
        two_electron_down = two_electron[:, self.up_spin:, :]

        return one_electron_up, one_electron_down, two_electron_up, two_electron_down
