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
import deepchem.models.optimizers as optimizers
from deepchem.utils.electron_sampler import ElectronSampler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
    self.epsilon = 1e-06
    self.projection_module = nn.ModuleList()
    self.projection_module.append(
        nn.Linear(4, n_two[0], bias=False, device=device))
    self.projection_module.append(
        nn.Linear(4 * nucleon_pos.size()[0],
                  n_one[0],
                  bias=False,
                  device=device))
    self.fermi_layer = nn.ModuleList()
    self.fermi_layer.append(
        nn.Linear(8 * nucleon_pos.size()[0] + 8,
                  n_one[0],
                  bias=True,
                  device=device))
    self.fermi_layer.append(nn.Linear(4, n_two[0], bias=True, device=device))
    for i in range(1, self.layers):
      self.fermi_layer.append(
          nn.Linear(2*n_two[i - 1] + 3*n_one[i - 1],
                    n_one[i], bias=True,
                    device=device))
      if(i!=(self.layers-1)):
        self.fermi_layer.append(nn.Linear(n_two[i - 1], n_two[i], bias=True, device=device))

    self.w_up = nn.ModuleList()
    self.w_down = nn.ModuleList()
    self.sigma_up = nn.ModuleList()
    self.sigma_down = nn.ModuleList()

    for i in range(self.determinant):
      for j in range(self.spin[0]):
        layer_w = nn.Linear(256, 1, bias=True, device=device)
        with torch.no_grad():
          layer_w.weight = torch.nn.Parameter(torch.squeeze(layer_w.weight,0)) # layer_w weight is 1D, which automatically does the dot product in linear layer calculation.
          layer_w.bias = torch.nn.Parameter(torch.squeeze(layer_w.bias,0))
        self.w_up.append(layer_w)
        for k in range(nucleon_pos.size()[0]):
          layer_sigma = nn.Linear(3, 3, bias=False, device=device)
          self.sigma_up.append(layer_sigma)
      for j in range(self.spin[1]):
        layer_w = nn.Linear(256, 1, bias=True, device=device)
        with torch.no_grad():
          layer_w.weight = torch.nn.Parameter(torch.squeeze(layer_w.weight,0))
          layer_w.bias = torch.nn.Parameter(torch.squeeze(layer_w.bias,0))
        self.w_down.append(layer_w)
        for k in range(nucleon_pos.size()[0]):
          layer_sigma = nn.Linear(3, 3, bias=False, device=device)
          self.sigma_down.append(layer_sigma)

  def forward(
      self,
      molecule: torch.tensor,
  ):
    """
    Forward pass of the network
    Parameters:
    -----------
    molecule: torch.tensor
      Tensor containing molecule's atom coordinates
    """
    # torch.autograd.set_detect_anomaly(True)
    self.molecule = molecule
    one_electron_vector = self.molecule - self.nucleon_pos

    shape = self.molecule.size()
    two_electron_vector = self.molecule.reshape([shape[0], 1, shape[1], 3
                                                ]) - self.molecule

    one_electron_vector = one_electron_vector[0, :, :, :]
    two_electron_vector = two_electron_vector[0, :, :, :]

    self.one_electron_distance = torch.linalg.norm(one_electron_vector +
                                                   self.epsilon,
                                                   axis=-1)
    self.two_electron_distance = torch.linalg.norm(two_electron_vector +
                                                   self.epsilon,
                                                   axis=-1)
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

    for i in range(0, self.layers):
      g_one_up = torch.mean(one_up, dim=0).flatten()
      g_one_down = torch.mean(one_down, dim=0).flatten()
      tmp_one_electron = torch.tensor([], requires_grad=True).to(device)
      tmp_two_electron = torch.tensor([], requires_grad=True).to(device)
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
        f_vector.retain_grad()
        if i == 0:
          one_stream = torch.tanh(self.fermi_layer[0](
              f_vector)) + self.projection_module[1](one_stream)
          two_stream = torch.tanh(self.fermi_layer[1](
              two_stream)) + self.projection_module[0](two_stream)
        else:
          one_stream = torch.tanh(self.fermi_layer[2*i](f_vector)) + one_stream
          if(i!=(self.layers-1)):
            two_stream = torch.tanh(
              self.fermi_layer[2*i + 1](two_stream)) + two_stream
        tmp_one_electron = torch.cat(
            (tmp_one_electron, one_stream.expand(1, -1)))
        tmp_two_electron = torch.cat(
            (tmp_two_electron, two_stream.expand(1, -1, -1)))
      one_electron = tmp_one_electron
      two_electron = tmp_two_electron
      one_up = one_electron[:self.spin[0], :]
      one_down = one_electron[self.spin[0]:, :]


    # self.psi_up = torch.ones((self.determinant, self.spin[0], self.spin[0])).to(device)
    # self.psi_down = torch.ones((self.determinant, self.spin[1], self.spin[1])).to(device)
    # self.psi_up.requires_grad = True
    # self.psi_down.requires_grad = True
    # self.psi_up.retain_grad()
    # self.psi_down.retain_grad()
    self.psi_up = torch.tensor([],requires_grad=True).to(device)
    self.psi_down = torch.tensor([],requires_grad=True).to(device)
    one_up = one_electron[:self.spin[0], :]
    one_down = one_electron[self.spin[0]:, :]

    for k in range(self.determinant):
      # spin-up orbitals
      for i in range(self.spin[0]):
        for j in range(self.spin[0]):
          envelope_up = torch.tensor(0.0, requires_grad=True).to(device)
          for m in range(self.nucleon_pos.size()[0]):
            envelope_up = envelope_up+torch.unsqueeze(torch.exp(
                -torch.linalg.norm(self.sigma_up[k*(self.total_electron)+i*(self.spin[0])+m](
                                   one_electron_vector[j][m].to(torch.float32)))),0)
          self.psi_up=torch.cat((self.psi_up,(self.w_up[k*(self.spin[0])+i](one_up[j])) * envelope_up))

      # spin-down orbitals
      for i in range(self.spin[1]):
        for j in range(self.spin[1]):
          envelope_down = torch.tensor(0.0,requires_grad=True).to(device)
          for m in range(self.nucleon_pos.size()[0]):
            envelope_down = envelope_down+torch.unsqueeze(torch.exp(
                -torch.linalg.norm(self.sigma_down[k*(self.total_electron)+i*(self.spin[1])+m](
                                   one_electron_vector[j+self.spin[0]][m].to(torch.float32)))),0) #m
          self.psi_down=torch.cat((self.psi_down,(self.w_down[k*(self.spin[1])+i](one_down[j])) * envelope_down))
    self.psi_up=torch.reshape(self.psi_up,(self.determinant, self.spin[0], self.spin[0]))
    self.psi_down=torch.reshape(self.psi_down,(self.determinant, self.spin[1], self.spin[1]))
    self.psi_predicted = torch.sum(torch.det(self.psi_up) *
                    torch.det(self.psi_down)).to(device)
    self.psi_log = torch.log(torch.abs(self.psi_predicted)).to(device)
    # print(self.psi_log)
    return self.psi_log

  def train_loss(self, log_psi:torch.tensor, hamiltonian:torch.tensor, energy_clip: float = 5.0):
    """Calculates the loss of the system during training.
    Parameters
    ----------
    log_psi : torch.tensor
      Log of the wavefunction 
    
     """
    # average_psi=torch.mean(log_psi).to(device)
    loss_batch_wise = torch.tensor([], requires_grad=True).to(device)
    # total_loss = torch.tensor([], requires_grad=True).to(device)
    loss=torch.mean(hamiltonian)
    tv=torch.mean(torch.abs(hamiltonian-loss))
    mean = torch.clip(hamiltonian,loss-energy_clip*tv,energy_clip*tv+loss) - loss
    print("spin")
    print(self.spin)
    print("mean")
    print(mean)
    print(torch.mean(hamiltonian))
    batches = hamiltonian.size()[0]
    print(batches)
    for i in range(batches):
      grad_psi = torch.autograd.grad(log_psi[i],
                                     self.parameters(),
                                     retain_graph=True,
                                     allow_unused=True)[0].to(device)
      loss_batch_wise = torch.cat(
          (loss_batch_wise, torch.unsqueeze(grad_psi * mean[i], 0)))
    print("loss")
    print(loss_batch_wise)
    print("grad")
    print(grad_psi)
    total_loss = 2 * torch.mean(loss_batch_wise)
    print("final")
    print(total_loss)
    return total_loss

  def calculate_potential(self,) -> Any:
    """Calculates the potential of the molecule system system for to calculate the hamiltonian loss.
    Returns:
    --------
    potential: Any
      The potential energy of the system.
    """

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

  def local_energy(self, mol: torch.tensor) -> Any:
    """Calculates the hamiltonian of the molecule system.
    Returns:
    --------
    hamiltonian: Any
      The hamiltonian of the system.
    """
    hessian = torch.autograd.functional.hessian(self.forward,
                                                mol,
                                                create_graph=True)
    jacobian = torch.autograd.grad([self.psi_log], [self.molecule],
                                   create_graph=True,
                                   allow_unused=True)[0]

    sum = torch.tensor([0.0], requires_grad=True).to(device)
    for i in range(self.molecule.size(1)):
      for j in range(self.molecule.size(3)):
        sum = sum + hessian[0, i, 0, j, 0, i, 0, j]

    potential = self.calculate_potential()
    total_energy = potential - 0.5 * (sum + torch.sum(torch.pow(jacobian, 2)))
    return (total_energy).to(torch.float32)

  def pretraining_loss(self, hartree_mo):
    """Calculates the loss of the system during pretraining """
    # Up-spin electrons
    loss = torch.tensor(0.0, requires_grad=True)
    p_pre = torch.tensor(1.0, requires_grad=True)
    for i in range(self.spin[0]):
      for j in range(self.spin[0]):
        for k in range(self.determinant):
          loss = loss + (self.psi_up[k][i][j] - hartree_mo[i][j])**2
          p_pre = p_pre*(hartree_mo[i][j]**2 + self.psi_predicted**2
                   )
    # Down-spin electrons
    for i in range(self.spin[0],self.spin[1]):
      for j in range(self.spin[0],self.spin[1]):
        for k in range(self.determinant):
          loss = loss+(self.psi_down[k][i][j] -
                   hartree_mo[i][j])**2
          p_pre = p_pre*(hartree_mo[i][j]**2 +
                    self.psi_predicted**2)
    p_pre = 0.5*p_pre
    return p_pre.to(torch.float64), loss.to(torch.float64)


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

  def __init__(self,
               nucleon_coordinates: List[List],
               spin: float,
               charge: int,
               seed: Optional[int] = None,
               batch_no: int = 4,
               pretrain=True):
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
    self.pretrain = pretrain
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    nucl = torch.from_numpy(self.nucleon_pos)
    nucl.requires_grad = True
    charges_total = torch.from_numpy(self.charge.astype(float))
    charges_total.requires_grad = True
    inter = torch.from_numpy(self.inter_atom)
    inter.requires_grad = True
    self.model = Ferminet(nucleon_pos=nucl.to(self.device),
                          nuclear_charge=charges_total.to(self.device),
                          spin=(self.up_spin, self.down_spin),
                          inter_atom=inter.to(self.device)).to(self.device)

    self.molecule: ElectronSampler = ElectronSampler(
        batch_no=self.batch_no,
        central_value=self.nucleon_pos,
        seed=self.seed,
        f=None,
        steps=1)  # sample the electrons using the electron sampler
    self.molecule.f = lambda x: 2 * self.psi_log()
    self.molecule.gauss_initialize_position(
        self.electron_no)  # initialize the position of the electrons
    adam = optimizers.Adam(learning_rate=0.0001)
    super(FerminetModel, self).__init__(model=self.model,
                                        loss=self.model.train_loss,
                                        optimizer=adam,
                                        output_types=['psi_log'])

  def psi_log(self,) -> torch.Tensor:
    if self.pretrain:
      psi = np.array([])
      self.pre_loss = torch.tensor([], requires_grad=True).to(device).to(torch.float64)
      self.dataset = torch.from_numpy(np.expand_dims(self.molecule.x,
                                                     axis=1)).to(self.device)
      self.dataset.requires_grad = True
      # for j in range(mini_batch_no):
      for i in range(self.batch_no):
        output = self.model.forward(self.dataset[i])
        ppre, loss = self.model.pretraining_loss(self.mo_values)
        psi = np.append(psi, torch.DoubleTensor.item(torch.log(ppre)/2))
        self.pre_loss = torch.cat(
            (self.pre_loss, torch.unsqueeze(loss,0)))
    else:
      self.psi = torch.tensor([], requires_grad=True).to(self.device)
      psi = np.array([])
      self.dataset = torch.from_numpy(np.expand_dims(self.molecule.x,
                                                     axis=1)).to(self.device)
      self.dataset.requires_grad = True
      for i in range(self.batch_no):
        output = self.model.forward(self.dataset[i])
        psi = np.append(psi, torch.IntTensor.item(output))
        self.psi = torch.cat((self.psi, output.unsqueeze(0)))
    return psi

  def prepare_hf_solution(self,):
    """Prepares the HF solution for the molecule system."""
    # preparing hartree focck molecular orbital solution
    molecule = ""
    for i in range(len(self.nucleon_pos)):
      molecule = molecule + self.nucleon_coordinates[i][0] + " " + str(
          self.nucleon_coordinates[i][1][0]) + " " + str(
              self.nucleon_coordinates[i][1][1]) + " " + str(
                  self.nucleon_coordinates[i][1][2]) + ";"
    mol = pyscf.gto.Mole(atom=molecule, basis='sto-3g')
    mol.unit = 'Ang'
    mol.spin = 2 * (self.up_spin - self.down_spin)
    mol.charge = self.ion_charge
    mol.build()
    mf = pyscf.scf.RHF(mol)
    kernel = mf.kernel()
    coeffs = mf.mo_coeff
    ao_values = mol.eval_gto("GTOval_sph", coeffs)
    mo_values = tuple(torch.from_numpy(np.matmul(ao_values, coeff)) for coeff in coeffs)
    self.mo_values = mo_values  # molecular orbital values
    print(self.mo_values)

  # override the fit method to invoke the MCMC sampling after each epoch.
  def fit(self,):
    """Fit the model to the data.
    """
    loss_full = []
    optimizer = self.optimizer._create_pytorch_optimizer(
        self.model.parameters())
    self.hamiltonian = torch.tensor([], requires_grad=True).to(device)
    self.hamiltonian.retain_grad()
    for i in range(1):  # number of epochs
      optimizer.zero_grad()
      self.molecule.move()
      loss = torch.mean(self.pre_loss)
      loss.backward()
      # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.001)
      optimizer.step()
      loss_full.append(loss)
    for j in range(self.batch_no):
        local_en = self.model.local_energy(self.dataset[j])
        self.hamiltonian = torch.cat(
            (self.hamiltonian, torch.unsqueeze(local_en, 0)))
    print("energy")
    print(torch.mean(self.hamiltonian))

    self.loss = self.model.train_loss
    self.pretrain = False
    self.optimizer = optimizers.KFAC(model=self.model,
                                     lr=0.0001,
                                     Tinv=10,
                                     mean=True)
    optimizer = self.optimizer._create_pytorch_optimizer()

    for i in range(2):  # number of epochs
      self.hamiltonian = torch.tensor([], requires_grad=True).to(device)
      self.hamiltonian.retain_grad()
      optimizer.zero_grad()
      self.molecule.move()
      for j in range(self.batch_no):
        local_en = self.model.local_energy(self.dataset[j])
        self.hamiltonian = torch.cat(
            (self.hamiltonian, torch.unsqueeze(local_en, 0)))
      loss = self.model.train_loss(self.psi, self.hamiltonian)
      loss.backward()
      torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.001)
      for name, param in self.model.named_parameters():
        print(name, param.grad)
      optimizer.step()
      loss_full.append(loss)
    return loss_full