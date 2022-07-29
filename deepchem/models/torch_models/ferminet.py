"""
Implementation of the Ferminet class in pytorch
"""

from typing import List, Optional
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
      A dictionary containing nucleon coordinates as the values with the keys as the element's symbol.
    seed_no: int, optional (default None)
      Random seed to use for electron initialization.
    batch_number: int, optional (default 10)
      Number of batches of the electron's positions to be initialized.

    """
    super(Ferminet, self).__init__()

    self.nucleon_coordinates = nucleon_coordinates
    self.seed_no = seed_no
    self.batch_number = batch_number

  def test_f(x):
    # dummy function which can be passed as the parameter f. f gives the log probability
    # TODO replace this function with forward pass of the model in future
    return 2 * np.log(np.random.uniform(low=0, high=1.0, size=np.shape(x)[0]))

  def prepare_input_stream(self,):
    """Prepares the one-electron and two-electron input stream for the model.

    Returns:
    --------
    one_electron_vector: numpy.ndarray
      The one-electron input stream containing the distance vector between the electron and nucleus coordinates.
    one_electron_distance: numpy.ndarray
      The one-electron input stream containing the distance between the electron and nucleus coordinates.
    two_electron_vector: numpy.ndarray
      The two-electron input stream containing the distance vector between all the electrons coordinates.
    two_electron_distance: numpy.ndarray
    """

    no_electrons = []
    nucleons = []

    for i in self.nucleon_coordinates:
      mol = Chem.MolFromSmiles(i[0])
      for j in mol.GetAtoms():
        no_electrons.append([j.GetAtomicNum() - j.GetFormalCharge()])
      nucleons.append(i[1])

    electron_no = np.array(no_electrons)
    nucleon_pos = np.array(nucleons)

    molecule = ElectronSampler(
        batch_no=self.batch_number,
        central_value=nucleon_pos,
        seed=self.seed_no,
        f=self.test_f,
        steps=1000)  # sample the electrons using the electron sampler
    molecule.gauss_initialize_position(
        electron_no)  # initialize the position of the electrons

    self.one_electron_vector: np.ndarray = molecule.x - nucleon_pos

    shape = np.shape(molecule.x)
    self.two_electron_vector: np.ndarray = molecule.x.reshape(
        [shape[0], 1, shape[1], 3]) - molecule.x

    self.one_electron_vector = self.one_electron_vector[0, :, :, :]
    self.two_electron_vector = self.two_electron_vector[0, :, :, :]

    self.one_electron_distance: np.ndarray = np.linalg.norm(
        self.one_electron_vector, axis=-1)
    self.two_electron_distance: np.ndarray = np.linalg.norm(
        self.two_electron_vector, axis=-1)


# TODO """
# def loss():
# """Calculate the loss."""
# pass

# def scf():
# "" Perform the SCF calculation."""
# pass

# def pretrain():
# """ Perform the pretraining.
# """
# pass

# def forward(self, x):
# """ Forward pass of the model.
# """
# pass
