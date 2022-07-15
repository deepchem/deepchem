"""
Implementation of the Ferminet class in pytorch
"""

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyscf
import rdkit.Chem.rdchem as rdchem
import numpy as np

from deepchem.models.torch_models import TorchModel
import deepchem.models.optimizers as optim
from deepchem.utils.electron_sampler import ElectronSampler

# TODO look for the loss function(Hamiltonian)


class Ferminet(nn.Module):
  """A deep-learning based Variational Monte Carlo method for calculating the ab-initio
    solution of a many-electron system.

    This model aims to calculate the ground state energy of a multi-electron system
    using a baseline solution as the Hartree-Fock. An MCMC technique is used to sample
    electrons and DNNs are used to caluclate the square magnitude of the wavefunction,
    in which electron-electron repulsions also are included in the calculation(in the
    form of Jastrow factor envelopes). The model requires only the nucleus' coordinates
    as input.

    This method is based on the following paper:
    """

  def __init__(self, nucleon_coordinates: List[str, List[int]]):
    """
        Parameters:
        -----------
        nucleon_coordinates: Dict[str, List[int]]
            A dictionary containing nucleon coordinates as the values with the keys as the element's symbol.
        """
    super(Ferminet, self).__init__()

    self.nucleon_coordinates = nucleon_coordinates

  def test_f(x):
    # dummy function which can be passed as the parameter f. f gives the log probability
    # TODO replace this function with forward pass of the model in future
    return 2 * np.log(np.random.uniform(low=0, high=1.0, size=np.shape(x)[0]))

  def prepare_input_stream(self):
    """Prepares the one-electron and two-electron input stream for the model.
        """

    no_electrons = []
    nucleons = []

    for i in self.nucleon_coordinates:
      no_electrons.append([rdchem.GetAtomicNum(i[0])])
      nucleons.append(i[1])

    molecule = ElectronSampler(
        central_value=np.array(nucleons), seed=0, f=self.test_f,
        steps=1000)  # sample the electrons using the electron sampler
    molecule.gauss_initialize_position(
        np.array(no_electrons))  # initialize the position of the electrons

    # TODO calculate the electron stream according to the shape of "molecule.x"


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
