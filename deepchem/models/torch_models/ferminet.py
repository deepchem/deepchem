"""
Implementation of the Ferminet class in pytorch
"""

from typing import List, Optional
from typing_extensions import reveal_type
# import torch.nn as nn
from rdkit import Chem
import numpy as np

# from deepchem.models.torch_models import TorchModel
# import deepchem.models.optimizers as optim
from deepchem.utils.electron_sampler import ElectronSampler

# TODO look for the loss function(Hamiltonian)


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

  def __init__(self,
               nucleon_coordinates: List[List],
               seed_no: Optional[int] = None,
               batch_number: int = 10):
    """
    Parameters:
    -----------
    nucleon_coordinates:  List[List]
      A list containing nucleon coordinates as the values with the keys as the element's symbol.
    seed_no: int, optional (default None)
      Random seed to use for electron initialization.
    batch_number: int, optional (default 10)
      Number of batches of the electron's positions to be initialized.

    """
    # super(Ferminet, self).__init__()

    self.nucleon_coordinates = nucleon_coordinates
    self.seed_no = seed_no
    self.batch_number = batch_number

  def test_f(x: np.ndarray) -> np.ndarray:
    # dummy function which can be passed as the parameter f. f gives the log probability
    # TODO replace this function with forward pass of the model in future
    return 2 * np.log(np.random.uniform(low=0, high=1.0, size=np.shape(x)[0]))

  def prepare_input_stream(self,) -> np.ndarray:
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
    self.charge: List = []

    for i in self.nucleon_coordinates:
      mol = Chem.MolFromSmiles('[' + i[0] + ']')
      for j in mol.GetAtoms():
        self.charge.append(j.GetAtomicNum())
        no_electrons.append([j.GetAtomicNum() - j.GetFormalCharge()])
      nucleons.append(i[1])

    self.electron_no: np.ndarray = np.array(no_electrons)
    self.nucleon_pos: np.ndarray = np.array(nucleons)

    spin = np.sum(self.electron_no) % 2
    self.up_spin: int = (spin + np.sum(self.electron_no)) // 2

    molecule = ElectronSampler(
        batch_no=self.batch_number,
        central_value=self.nucleon_pos,
        seed=self.seed_no,
        f=self.test_f,
        steps=1000)  # sample the electrons using the electron sampler
    molecule.gauss_initialize_position(
        self.electron_no)  # initialize the position of the electrons

    one_electron_vector = molecule.x - self.nucleon_pos

    shape = np.shape(molecule.x)
    two_electron_vector = molecule.x.reshape([shape[0], 1, shape[1], 3
                                             ]) - molecule.x

    one_electron_vector = one_electron_vector[0, :, :, :]
    two_electron_vector = two_electron_vector[0, :, :, :]

    self.one_electron_distance: np.ndarray = np.linalg.norm(one_electron_vector,
                                                            axis=-1)
    self.two_electron_distance: np.ndarray = np.linalg.norm(two_electron_vector,
                                                            axis=-1)

    # concatenating distance and vectors arrays
    one_shape = np.shape(self.one_electron_distance)
    one_distance = self.one_electron_distance.reshape(1, one_shape[0],
                                                      one_shape[1], 1)
    one_electron = np.block([one_electron_vector, one_distance])
    two_shape = np.shape(self.two_electron_distance)
    two_distance = self.two_electron_distance.reshape(1, two_shape[0],
                                                      two_shape[1], 1)
    two_electron = np.block([two_electron_vector, two_distance])

    one_electron_up = one_electron[:self.up_spin, :, :]
    one_electron_down = one_electron[self.up_spin:, :, :]

    two_electron_up = two_electron[:self.up_spin, :, :]
    two_electron_down = two_electron_vector[self.up_spin:, :, :]

    return one_electron_up, one_electron_down, two_electron_up, two_electron_down
