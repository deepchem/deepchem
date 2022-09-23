"""
Implementation of the Ferminet class in pytorch
"""

try:
  import torch
  import torch.nn as nn
except ModuleNotFoundError:
  raise ImportError('These classes require PyTorch to be installed.')

try:
  import pyscf
except ModuleNotFoundError:
  raise ImportError('These classes require pyscf to be installed.')

from typing import List, Optional, Any
from rdkit import Chem
import numpy as np
from deepchem.utils.molecule_feature_utils import ALLEN_ELECTRONEGATIVTY
from deepchem.utils.geometry_utils import compute_pairwise_distances

from deepchem.models.torch_models import TorchModel
import deepchem.models.optimizers as optim
from deepchem.utils.electron_sampler import ElectronSampler


class Ferminet(torch.nn.Module):
  """Approximates the log probability of the wave function of a molecule system using DNNs.
  """

  def __init__(self,
               nucleon_pos: torch.tensor,
               nuclear_charge: torch.tensor,
               spin: tuple,
               inter_atom: torch.tensor,
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
    super(Ferminet, self).__init__()
    if len(n_one) != len(n_two):
      raise ValueError(
          "The number of layers in one-electron and two-electron stream should be equal"
      )
    else:
      self.layers = len(n_one)
    self.nucleon_pos = nucleon_pos
    self.determinant = determinant
    self.spin = spin
    self.total_electron = spin[0] + spin[1]
    self.inter_atom = inter_atom
    self.nuclear_charge = nuclear_charge
    self.projection_matrix_two = nn.Linear(4, n_two[0])
    self.projection_matrix = nn.Linear(4 * nucleon_pos.size()[0], n_one[0])
    self.fermi_layer = nn.ModuleList()
    self.fermi_layer.append(nn.Linear(8 * nucleon_pos.size()[0] + 8, n_one[0]))
    self.fermi_layer.append(nn.Linear(4, n_two[0]))
    for i in range(1, self.layers):
      self.fermi_layer.append(
          nn.Linear((4 * nucleon_pos.size()[0] + 64) + n_one[i - 1], n_one[i]))
      self.fermi_layer.append(nn.Linear(n_two[i - 1], n_two[i]))

    self.w_up = nn.ParameterList()
    self.g_up = nn.ParameterList()
    self.w_down = nn.ParameterList()
    self.g_down = nn.ParameterList()
    self.pi_up = nn.ParameterList()
    self.sigma_up = nn.ParameterList()
    self.pi_down = nn.ParameterList()
    self.sigma_down = nn.ParameterList()

    for i in range(self.determinant):
      for j in range(self.spin[0]):
        self.w_up.append(
            nn.parameter.Parameter(nn.init.normal(torch.Tensor(n_one[-1]))))
        self.g_up.append(nn.parameter.Parameter(nn.init.normal(
            torch.Tensor(1))))
        for k in range(nucleon_pos.size()[0]):
          self.pi_up.append(
              nn.parameter.Parameter(nn.init.normal(torch.Tensor(1))))
          self.sigma_up.append(
              nn.parameter.Parameter(nn.init.normal(torch.Tensor(1))))
      for j in range(self.spin[1]):
        self.w_down.append(
            nn.parameter.Parameter(nn.init.normal(torch.Tensor(n_one[-1]))))
        self.g_down.append(
            nn.parameter.Parameter(nn.init.normal(torch.Tensor(1))))
        for k in range(nucleon_pos.size()[0]):
          self.pi_down.append(
              nn.parameter.Parameter(nn.init.normal(torch.Tensor(1))))
          self.sigma_down.append(
              nn.parameter.Parameter(nn.init.normal(torch.Tensor(1))))

  def forward(
      self,
      molecule: torch.tensor,
  ):
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
    one_electron_vector = molecule - self.nucleon_pos

    shape = molecule.size()
    two_electron_vector = molecule.reshape([shape[0], 1, shape[1], 3
                                           ]) - molecule

    one_electron_vector = one_electron_vector[0, :, :, :]
    two_electron_vector = two_electron_vector[0, :, :, :]

    self.one_electron_distance: torch.tensor = torch.linalg.norm(
        one_electron_vector, axis=-1)
    self.two_electron_distance: torch.tensor = torch.linalg.norm(
        two_electron_vector, axis=-1)

    # concatenating distance and vectors arrays
    one_shape = self.one_electron_distance.size()
    one_distance = self.one_electron_distance.reshape(one_shape[0],
                                                      one_shape[1], 1)
    one_electron = torch.cat(
        (one_electron_vector, one_distance.repeat(1, 1, 3)), dim=-1)
    one_electron = one_electron[:, :, 0:4]

    two_shape = self.two_electron_distance.size()
    two_distance = self.two_electron_distance.reshape(two_shape[0],
                                                      two_shape[1], 1)
    two_electron = torch.cat(
        (two_electron_vector, two_distance.repeat(1, 1, 3)), dim=-1)
    two_electron = two_electron[:, :, 0:4]

    one_up = one_electron[:, :self.spin[0], :]
    one_down = one_electron[:, self.spin[0]:, :]

    one_up = one_up.to(torch.float32)
    one_down = one_down.to(torch.float32)
    one_electron = one_electron.to(torch.float32)
    two_electron = two_electron.to(torch.float32)
    for i in range(0, 2 * self.layers, 2):
      g_one_up = torch.mean(one_up, dim=0).flatten()
      g_one_down = torch.mean(one_down, dim=0).flatten()
      tmp_one_electron = torch.tensor([], requires_grad=True)
      tmp_two_electron = torch.tensor([], requires_grad=True)
      # spin-up electrons
      for j in range(0, self.total_electron):
        one_stream = one_electron[j].flatten()
        two_stream = two_electron[j]
        g_two_up = torch.mean(two_stream[:self.spin[0]], dim=0).flatten()
        g_two_down = torch.mean(two_stream[self.spin[0]:], dim=0).flatten()
        f_vector = torch.vstack(
            (one_stream.reshape(one_stream.size()[0],
                                1), g_one_up.reshape(g_one_up.size()[0], 1),
             g_one_down.reshape(g_one_down.size()[0],
                                1), g_two_up.reshape(g_two_up.size()[0], 1),
             g_two_down.reshape(g_two_down.size()[0], 1)))
        f_vector = f_vector.to(torch.float32)
        f_vector = f_vector.flatten()
        if i == 0:
          one_stream = torch.tanh(self.fermi_layer[i](
              f_vector)) + self.projection_matrix(one_stream)
          two_stream = torch.tanh(self.fermi_layer[i + 1](
              two_stream)) + self.projection_matrix_two(two_stream)
        else:
          one_stream = torch.tanh(self.fermi_layer[i](f_vector)) + one_stream
          two_stream = torch.tanh(
              self.fermi_layer[i + 1](two_stream)) + two_stream
        tmp_one_electron = torch.cat(
            (tmp_one_electron, one_stream.expand(1, -1)))
        tmp_two_electron = torch.cat(
            (tmp_two_electron, two_stream.expand(1, -1, -1)))

      one_electron = tmp_one_electron
      two_electron = tmp_two_electron

    self.psi_up = torch.empty((self.determinant, self.spin[0], self.spin[0]))
    self.psi_down = torch.empty((self.determinant, self.spin[1], self.spin[1]))
    one_up = one_electron[:self.spin[0], :]
    one_down = one_electron[self.spin[0]:, :]

    for k in range(self.determinant):
      # spin-up orbitals
      for i in range(self.spin[0]):
        for j in range(self.spin[0]):
          envelope = 0
          for m in range(self.nucleon_pos.size()[0]):
            envelope += self.pi_up[k + i + j + m] * torch.exp(
                -torch.linalg.norm(self.sigma_up[k + i + j + m] *
                                   one_electron_vector[j][m]))
          self.psi_up[k][i][j] = (torch.dot(self.w_up[i + k], one_up[j]) +
                                  self.g_up[i + k]) * envelope
        # spin-down orbitals
        for j in range(self.spin[1]):
          envelope = 0
          for m in range(self.nucleon_pos.size()[0]):
            envelope += self.pi_up[k + i + j + m] * torch.exp(
                -torch.linalg.norm(self.sigma_up[k + i + j + m] *
                                   one_electron_vector[j][m]))
          self.psi_down[k][i][j] = (torch.dot(self.w_down[i + k], one_down[j]) +
                                    self.g_down[i + k]) * envelope
    psi = torch.sum(torch.det(self.psi_up) * torch.det(self.psi_down))
    psi_log = torch.log(torch.abs(psi))
    return 2 * psi_log

  def loss(self, log_psi, hamiltonian):
    """Calculates the loss of the system. """
    # TODO: change the concatenation of the hamiltonian

    loss = torch.tensor([], requires_grad=True)
    total_loss = torch.tensor([], requires_grad=True)
    mean = np.expand_dims(hamiltonian - torch.mean(hamiltonian), axis=1)
    batches = 1
    for i in range(batches):
      grad_psi = torch.autograd.grad(log_psi,
                                     self.parameters(),
                                     allow_unused=True)
      for j in range(len(grad_psi)):
        if grad_psi[j] is not None:
          loss = torch.cat((loss,
                            torch.tensor(torch.mean(grad_psi[j]) * mean[i],
                                         requires_grad=True)))
      total_loss = torch.cat(
          (total_loss, torch.tensor([torch.mean(loss)], requires_grad=True)))
      total_loss = torch.mean(total_loss)

    return total_loss

  def calculate_potential(self,) -> Any:
    """Calculates the potential of the molecule system system for to calculate the hamiltonian loss.
    Returns:
    --------
    potential: Any
      The potential energy of the system.
    """

    # electron-nuclear potential
    electron_nuclear_potential = -1 * torch.sum(
        self.nuclear_charge * (1 / self.one_electron_distance))

    # electron-electron potential
    electron_electron_potential = torch.sum(
        torch.tril(1 / self.two_electron_distance, -1))

    # nuclear-nuclear potential
    charge_shape = self.nuclear_charge.size()
    nuclear_nuclear_potential = torch.sum(
        self.nuclear_charge * self.nuclear_charge.reshape(charge_shape[0], 1) *
        torch.tril(1 / self.inter_atom, -1))
    potential = electron_nuclear_potential + electron_electron_potential + nuclear_nuclear_potential
    potential = potential.to(torch.float32)
    return potential

  def local_energy(self, output: torch.tensor, input: torch.tensor) -> Any:
    """Calculates the hamiltonian of the molecule system.
    Returns:
    --------
    hamiltonian: Any
      The hamiltonian of the system.
    """
    jacobian = torch.autograd.grad([output], [input], retain_graph=True)
    hessian = torch.autograd.functional.hessian(self.forward,
                                                input,
                                                create_graph=True)

    print(jacobian)
    print(torch.sum(hessian))
    potential = self.calculate_potential()
    total_energy = potential - 0.5 * torch.sum(torch.diag(
        hessian, 0)) + torch.sum(torch.pow(jacobian, 2))
    print(total_energy)
    return total_energy


class FerminetModel(TorchModel):
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
      batch_no: int = 1,
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
    self.nucleon_coordinates = nucleon_coordinates
    self.seed = seed
    self.batch_no = batch_no
    self.spin = spin
    self.ion_charge = charge
    self.hamiltonian = torch.tensor([], requires_grad=True)
    self.total_psi = np.array([])

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

    # Initialization for ionic molecules
    if np.sum(self.electron_no) < self.ion_charge:
      raise ValueError("Given charge is not initializable")

    # Initialization for ionic molecules
    if self.ion_charge != 0:
      if len(nucleons) == 1:  # for an atom, directly the charge is applied
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

    model = Ferminet(nucleon_pos=torch.from_numpy(self.nucleon_pos),
                     nuclear_charge=torch.from_numpy(self.charge),
                     spin=(self.up_spin, self.down_spin),
                     inter_atom=torch.from_numpy(self.inter_atom))

    self.molecule: ElectronSampler = ElectronSampler(
        batch_no=self.batch_no,
        central_value=self.nucleon_pos,
        seed=self.seed,
        f=lambda x: self.psi_log,
        steps=1000)  # sample the electrons using the electron sampler
    self.molecule.gauss_initialize_position(
        self.electron_no)  # initialize the position of the electrons
    # self.molecule.move()
    super(FerminetModel, self).__init__(model=model,
                                        loss=model.local_energy,
                                        optimizer=optim.KFAC(model=model,
                                                             lr=0.1),
                                        output_types=['psi_log'])

  def psi_log(self,) -> list:
    self.hamiltonian = torch.tensor([], requires_grad=True)
    self.total_psi = torch.tensor([], requires_grad=True)
    psi = np.array([])
    dataset = torch.from_numpy(self.molecule.x)
    dataset = np.expand_dims(dataset, axis=1)
    for i in range(self.batch_no):
      output = self.model.forward(dataset[i], axis=0)
      np.append(psi, torch.IntTensor.item(output))
      torch.cat((self.total_psi, output))
      torch.cat((self.hamiltonian,
                 torch.tensor([self.model.local_energy(dataset[i])],
                              requires_grad=True)))
    return psi

  # override the fit method to invoke the MCMC sampling after each epoch.
  def fit(self,):
    """Fit the model to the data.
    """
    loss_full = []
    for i in range(1000):  # number of epochs
      self.molecule.move()

      self.optimizer.zero_grad()
      loss = self.model.loss(self.hamiltonian, self.total_psi)
      loss.backward()
      self.optimizer.step()
      loss_full.append(loss)
    return loss_full
