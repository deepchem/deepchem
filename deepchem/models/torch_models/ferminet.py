"""
Implementation of the Ferminet class in pytorch
"""

try:
  import torch
  from torch import Tensor
  import torch.nn as nn
  import torch.nn.functional as F
except ModuleNotFoundError:
  raise ImportError('These classes require PyTorch to be installed.')

from typing import List, Optional, Any, Tuple
from rdkit import Chem
import numpy as np
from deepchem.utils.molecule_feature_utils import ALLEN_ELECTRONEGATIVTY
from deepchem.utils.geometry_utils import compute_pairwise_distances

from deepchem.models.torch_models import TorchModel
import deepchem.models.optimizers as optim
from deepchem.utils.electron_sampler import ElectronSampler


def test_f(x: np.ndarray) -> np.ndarray:
  # dummy function which can be passed as the parameter f. f gives the log probability
  # TODO replace this function with forward pass of the model in future
  return 2 * np.log(np.random.uniform(low=0, high=1.0, size=np.shape(x)[0]))


class FerminetModel(torch.nn.Module):
  """Approximates the log probability of the wave function of a molecule system using DNNs.
  """

  def __init__(self,
               n_one: List = [256, 256, 256, 256],
               n_two: List = [32, 32, 32, 32],
               determinant: int = 16) -> None:
    """
    Parameters:
    -----------
    n_one: List
      List of hidden units for the one-electron stream in each layer
    n_two: List
      List of hidden units for the two-electron stream in each layer
    determinant: int
      Number of determinants for the final solution
    """
    if len(n_one) != len(n_two):
      raise ValueError(
          "The number of layers in one-electron and two-electron stream should be equal"
      )
    else:
      self.layers = len(n_one)
    super(FerminetModel, self).__init__()
    self.fermi_layer = nn.ModuleList()
    self.fermi_layer.append(nn.Linear(
        n_one[0], 20))  # TODO: Check the 2nd dimension of the linear weight
    self.fermi_layer.append(nn.Linear(n_two[0], 4))
    for i in range(1, self.layers):
      self.fermi_layer.append(
          nn.Linear(n_one[i], 3 * n_one[i - 1] + 2 * n_two[i]))
      self.fermi_layer.append(nn.Linear(n_two[i], n_two[i - 1]))

  def forward(self, one_electron_up: np.ndarray, one_electron_down: np.ndarray,
              two_electron_up: np.ndarray, two_electron_down: np.ndarray):
    """
    Parameters:
    -----------
    one_electron_up: np.ndarray
      Numpy array containing up-spin electron's one-electron feature
    one_electron_down: np.ndarray
      Numpy array containing down-spin electron's one-electron feature
    two_electron_up: np.ndarray
      Numpy array containing up-spin electron's two-electron feature
    two_electron_down: np.ndarray
      Numpy array containing down-spin electron's two-electron feature
    """
    one_up = torch.from_numpy(one_electron_up)
    one_down = torch.from_numpy(one_electron_down)
    two_up = torch.from_numpy(two_electron_up)
    two_down = torch.from_numpy(two_electron_down)

    # TODO: Look into batchwise feed of input
    for i in range(0, self.layers, 2):
      g_one_up = torch.sum(one_up.view(-1, 6), -1)
      g_one_down = torch.sum(one_down.view(-1, 6), -1)
      g_two_down = torch.sum(two_down.view(-1, 6), -1)
      for electron in one_up:
        g_two_up = torch.sum(two_up.view(-1, 6), -1)
        f_vector = torch.cat(electron, g_one_up, g_one_down, g_two_up,
                             g_two_down)
        one_up = torch.tanh(self.fermi_layer[i](f_vector)) + one_up
        two_up = torch.tanh(
            self.fermi_layer[i + 1](two_up)
        ) + two_up  # TODO: two_up should be replaced with corresponding elctron not whole.
      g_two_up = torch.sum(two_up.view(-1, 6), -1)
      for electron in one_down:
        g_two_down = torch.sum(two_down.view(-1, 6), -1)
        f_vector = torch.cat(electron, g_one_up, g_one_down, g_two_up,
                             g_two_down)
        one_down = torch.tanh(self.fermi_layer[i](f_vector)) + one_down
        two_down = torch.tanh(
            self.fermi_layer[i + 1](two_down)
        ) + two_down  # TODO: two_up should be replaced with corresponding elctron not whole.


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
    charge:int
      The total charge of the molecule system.
    seed_no: int, optional (default None)
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
      if len(nucleons) == 1:  # for an atom, directly the charge is applied
        self.electron_no[0][0] -= self.ion_charge
      elif len(
          nucleons
      ) == 2:  # for a diatomic molecule, the most electronegative atom will get the anionic charge
        electro_neg = electro_neg[electro_neg[:, 1].argsort()]
        if self.ion_charge > 0:
          pos = electro_neg[0][0]
        else:
          pos = electro_neg[-1][0]
        self.electron_no[int(pos)][0] -= self.ion_charge
      else:  # for a multiatomic molecule, the atom's electronegativity is averaged out with the weight sum of neighbouring atom's electronegativity and their interatomic distance
        electro_neg[:, 1] = np.sum(electro_neg[:, 1] / (1 + self.inter_atom),
                                   axis=-1)
        electro_neg = electro_neg[electro_neg[:, 1].argsort()]
        identical_pos = np.count_nonzero(electro_neg[:, 1] == electro_neg[0][1])
        identical_neg = np.count_nonzero(electro_neg[:,
                                                     1] == electro_neg[-1][1])
        if self.ion_charge > 1 and identical_pos > 1:
          per_atom_charge = self.ion_charge // identical_pos
          extra_charge = self.ion_charge % identical_pos
          pos = 0
          increment = 1
          for iter in range(identical_pos):
            self.electron_no[int(electro_neg[pos][0])][0] -= per_atom_charge
            pos += increment
          self.electron_no[int(electro_neg[pos -
                                           increment][0])][0] -= extra_charge
        elif self.ion_charge < -1 and identical_neg > 1:
          per_atom_charge = self.ion_charge // identical_neg
          extra_charge = self.ion_charge % identical_neg
          pos = -1
          increment = -1
          for iter in range(identical_neg):
            self.electron_no[int(electro_neg[pos][0])][0] -= per_atom_charge
            pos += increment
          self.electron_no[int(electro_neg[pos -
                                           increment][0])][0] += extra_charge
        else:
          if self.ion_charge > 0:
            pos = 0
          else:
            pos = -1
          self.electron_no[int(electro_neg[pos][0])][0] -= self.ion_charge

    total_electrons = np.sum(self.electron_no)
    self.up_spin = (total_electrons + 2 * self.spin) // 2
    self.down_spin = (total_electrons - 2 * self.spin) // 2

    self.molecule: ElectronSampler = ElectronSampler(
        batch_no=self.batch_no,
        central_value=self.nucleon_pos,
        seed=self.seed,
        f=test_f,
        steps=1000
    )  # sample the electrons using the electron sampler sample the electrons using the electron sampler sample the electrons using the electron sampler sample the electrons using the electron sampler sample the electrons using the electron sampler sample the electrons using the electron sampler sample the electrons using the electron sampler
    self.molecule.gauss_initialize_position(
        self.electron_no)  # initialize the position of the electrons

    one_electron_vector = self.molecule.x - self.nucleon_pos

    shape = np.shape(self.molecule.x)
    two_electron_vector = self.molecule.x.reshape([shape[0], 1, shape[1], 3
                                                  ]) - self.molecule.x

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

    one_electron_up = one_electron[:, :self.up_spin, :]
    one_electron_down = one_electron[:, self.up_spin:, :]

    two_electron_up = two_electron[:, :self.up_spin, :]
    two_electron_down = two_electron[:, self.up_spin:, :]

    return one_electron_up, one_electron_down, two_electron_up, two_electron_down

  def calculate_potential(self,) -> Any:
    """Calculates the potential of the molecule system system for to calculate the hamiltonian loss.
    Returns:
    --------
    potential: Any
      The potential energy of the system.
    """
    nuclear_charge = np.array(self.charge)

    # electron-nuclear potential
    electron_nuclear_potential = -1 * np.sum(nuclear_charge *
                                             (1 / self.one_electron_distance))

    # electron-electron potential
    electron_electron_potential = np.sum(
        np.tril(1 / self.two_electron_distance, -1))

    # nuclear-nuclear potential
    pos_shape = np.shape(self.nucleon_pos)
    charge_shape = np.shape(nuclear_charge)
    nuclear_nuclear_potential = np.sum(
        nuclear_charge * nuclear_charge.reshape(charge_shape[0], 1) *
        np.tril(1 / self.inter_atom, -1))

    return electron_nuclear_potential + electron_electron_potential + nuclear_nuclear_potential

  def local_energy(self, f: nn.Module) -> Any:
    """Calculates the hamiltonian of the molecule system.
    Returns:
    --------
    hamiltonian: Any
      The hamiltonian of the system.
    """
    shape = np.shape(self.molecule.x)[0]
    eye = torch.eye(shape)
    grad = torch.autograd.grad(f, self.molecule.x, retain_graph=True)
    jacobian_psi, hessian_psi = torch.autograd.functional.jvp(
        grad, self.molecule.x)
    val = 0
    for i in range(shape):
      val += hessian_psi(eye[i])[i]
    result = val.sum()
    potential = self.calculate_potential()
    return torch.from_numpy(potential) - 0.5 * (result +
                                                ((jacobian_psi.sum())**2))
