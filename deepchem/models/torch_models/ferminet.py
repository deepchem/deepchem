"""
Implementation of the Ferminet class in pytorch
"""

from typing import List, Optional, Tuple
# import torch.nn as nn
from rdkit import Chem
import numpy as np
from deepchem.utils.molecule_feature_utils import ALLEN_ELECTRONEGATIVTY
from deepchem.models.torch_models import TorchModel
import torch
from deepchem.models.torch_models.layers import FerminetElectronFeature, FerminetEnvelope

from deepchem.utils.electron_sampler import ElectronSampler


class Ferminet(torch.nn.Module):
    """A deep-learning based Variational Monte Carlo method [1]_ for calculating the ab-initio
    solution of a many-electron system.

    This model must be pre-trained on the HF baseline and then can be used to improve on it using electronic interactions
    via the learned electron distance features. Ferminet models the orbital values using envelope functions and the
    calculated electron features, which can then be used to approximate the wavefunction for better sampling of electrons.

    Example
    -------
    >>> import numpy as np
    >>> import deepchem as dc
    >>> import torch
    >>> H2_molecule =  torch.Tensor([[0, 0, 0.748], [0, 0, 0]])
    >>> H2_charge = torch.Tensor([[1], [1]])
    >>> model = dc.models.Ferminet(nucleon_pos=H2_molecule, nuclear_charge=H2_charge, batch_size=1)
    >>> electron = np.random.rand(1, 2*3)
    >>> wavefunction = model.forward(electron)

    References
    ----------
    .. [1] Spencer, James S., et al. Better, Faster Fermionic Neural Networks. arXiv:2011.07125, arXiv, 13 Nov. 2020. arXiv.org, http://arxiv.org/abs/2011.07125.

    """

    def __init__(self,
                 nucleon_pos: torch.Tensor,
                 nuclear_charge: torch.Tensor,
                 spin: tuple,
                 n_one: List = [256, 256, 256, 256],
                 n_two: List = [32, 32, 32, 32],
                 determinant: int = 16,
                 batch_size: int = 8) -> None:
        """
        Parameters:
        -----------
        nucleon_pos: torch.Tensor
            tensor containing the nucleons coordinates, it is in the shape of (number of atoms, 3)
        nuclear_charge: torch.Tensor
            tensor containing the electron number associated with each nucleon, it is in the shape of (number of atoms, no_of_electron)
        n_one: List
            List of hidden units for the one-electron stream in each layer
        n_two: List
            List of hidden units for the two-electron stream in each layer
        determinant: int
            Number of determinants for the final solution
        batch_size: int
            Number of molecule samples to be in each batch

        Attributes
        ----------
        running_diff: torch.Tensor
            torch tensor containing the loss which gets updated for each random walk performed
        ferminet_layer: torch.nn.ModuleList
            Modulelist containing the ferminet electron feature layer
        ferminet_layer_envelope: torch.nn.ModuleList
            Modulelist containing the ferminet envelope electron feature layer
        nuclear_nuclear_potential: torch.Tensor
            Torch tensor containing the inter-nuclear potential in the molecular system
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
        self.batch_size = batch_size
        self.spin = spin
        self.total_electron = spin[0] + spin[1]
        self.nuclear_charge = nuclear_charge
        self.n_one = n_one
        self.n_two = n_two
        self.ferminet_layer: torch.nn.ModuleList = torch.nn.ModuleList()
        self.ferminet_layer_envelope: torch.nn.ModuleList = torch.nn.ModuleList(
        )
        self.running_diff: torch.Tensor = torch.zeros(self.batch_size)
        self.nuclear_nuclear_potential: torch.Tensor = self.calculate_nuclear_nuclear(
        )

        self.ferminet_layer.append(
            FerminetElectronFeature(self.n_one, self.n_two,
                                    self.nucleon_pos.size()[0], self.batch_size,
                                    self.total_electron,
                                    [self.spin[0], self.spin[1]]))
        self.ferminet_layer_envelope.append(
            FerminetEnvelope(self.n_one, self.n_two, self.total_electron,
                             self.batch_size, [self.spin[0], self.spin[1]],
                             self.nucleon_pos.size()[0], self.determinant))

    def forward(self, input: np.ndarray) -> torch.Tensor:
        """
        forward function

        Parameters:
        -----------
        input: np.ndarray
            contains the sampled electrons coordinate in the shape (batch_size,number_of_electrons*3)

        Returns:
        --------
        psi: torch.Tensor
            contains the wavefunction - 'psi' value. It is in the shape (batch_size), where each row corresponds to the solution of one of the batches
        """
        # creating one and two electron features
        eps = torch.tensor(1e-36)
        self.input = torch.from_numpy(input)
        self.input.requires_grad = True
        self.input = self.input.reshape((self.batch_size, -1, 3))
        two_electron_vector = self.input.unsqueeze(1) - self.input.unsqueeze(2)
        two_electron_distance = torch.linalg.norm(two_electron_vector + eps,
                                                  dim=3).unsqueeze(3)
        two_electron = torch.cat((two_electron_vector, two_electron_distance),
                                 dim=3)
        two_electron = torch.reshape(
            two_electron,
            (self.batch_size, self.total_electron, self.total_electron, -1))

        one_electron_vector = self.input.unsqueeze(
            1) - self.nucleon_pos.unsqueeze(1)
        one_electron_distance = torch.linalg.norm(one_electron_vector, dim=3)
        one_electron = torch.cat(
            (one_electron_vector, one_electron_distance.unsqueeze(-1)), dim=3)
        one_electron = torch.reshape(one_electron.permute(0, 2, 1, 3),
                                     (self.batch_size, self.total_electron, -1))
        one_electron_vector_permuted = one_electron_vector.permute(0, 2, 1, 3)

        one_electron, _ = self.ferminet_layer[0].forward(
            one_electron.to(torch.float32), two_electron.to(torch.float32))
        self.psi, self.psi_up, self.psi_down = self.ferminet_layer_envelope[
            0].forward(one_electron, one_electron_vector_permuted)
        return self.psi

    def loss(self,
             psi_up_mo: List[Optional[np.ndarray]] = [None],
             psi_down_mo: List[Optional[np.ndarray]] = [None],
             pretrain: List[bool] = [True]):
        """
        Implements the loss function for both pretraining and the actual training parts.

        Parameters
        ----------
        psi_up_mo: List[Optional[np.ndarray]] (default [None])
            numpy array containing the sampled hartreee fock up-spin orbitals
        psi_down_mo: List[Optional[np.ndarray]] (default [None])
            numpy array containing the sampled hartreee fock down-spin orbitals
        pretrain: List[bool] (default [True])
            indicates whether the model is pretraining
        """
        criterion = torch.nn.MSELoss()
        if pretrain:
            psi_up_mo_torch = torch.from_numpy(psi_up_mo).unsqueeze(1)
            psi_down_mo_torch = torch.from_numpy(psi_down_mo).unsqueeze(1)
            self.running_diff = self.running_diff + criterion(
                self.psi_up, psi_up_mo_torch.float()) + criterion(
                    self.psi_down, psi_down_mo_torch.float())

    def calculate_nuclear_nuclear(self,) -> torch.Tensor:
        """
        Function to calculate where only the nucleus terms are involved and does not change when new electrons are sampled.
        atom-atom potential term = Zi*Zj/|Ri-Rj|, where Zi, Zj are the nuclear charges and Ri, Rj are nuclear coordinates

        Returns:
        --------
        A torch tensor of a scalar value containing the nuclear-nuclear potential term (does not change for the molecule system with sampling of electrons)
        """

        potential = torch.nan_to_num(
            (self.nuclear_charge * 1 /
             torch.cdist(self.nucleon_pos.float(), self.nucleon_pos.float()) *
             self.nuclear_charge.unsqueeze(1)),
            posinf=0.0,
            neginf=0.0)
        potential = torch.sum(potential) / 2
        return potential

    def calculate_electron_nuclear(self,) -> torch.Tensor:
        """
        Function to calculate the expected electron-nuclear potential term per batch
        nuclear-electron potential term = Zi/|Ri-rj|, rj is the electron coordinates, Ri is the nuclear coordinates, Zi is the nuclear charge

        Returns:
        --------
        A torch tensor of a scalar value containing the electron-nuclear potential term.
        """

        potential = torch.sum(
            (1 / torch.cdist(self.input.float(), self.nucleon_pos.float())) *
            self.nuclear_charge) / 2
        return (potential / self.batch_size)

    def calculate_electron_electron(self,):
        """
        Function to calculate the expected electron-nuclear potential term per batch
        nuclear-electron potential term = 1/|ri-rj|, ri, rj is the electron coordinates

        Returns:
        --------
        A torch tensor of a scalar value containing the electron-electron potential term.
        """
        potential = torch.sum(
            torch.nan_to_num(
                (1 / torch.cdist(self.input.float(), self.input.float())),
                posinf=0.0,
                neginf=0.0)) / 2
        return (potential / self.batch_size)

    def calculate_kinetic_energy(self,):
        """
        Function to calculate the expected kinetic energy term per batch
        It is calculated via:
        \sum_{ri}^{}[(\pdv[]{log|\Psi|}{(ri)})^2 + \pdv[2]{log|\Psi|}{(ri)}]

        Returns:
        --------
        A torch tensor of a scalar value containing the electron-electron potential term.
        """
        log_probability = torch.log(torch.abs(self.psi))
        jacobian = list(
            map(
                lambda x: torch.autograd.grad(x, self.input, create_graph=True)[
                    0], log_probability))
        jacobian_square = list(
            map(lambda x: torch.sum(torch.pow(x, 2)), jacobian))
        jacobian_square_sum = torch.tensor(0.0)
        hessian = torch.tensor(0.0)
        for i in range(self.batch_size):
            jacobian_square_sum = jacobian_square_sum + jacobian_square[i]
            for j in range(self.total_electron):
                for k in range(3):
                    hessian = hessian + torch.autograd.grad(
                        jacobian[i][i][j][k], self.input,
                        create_graph=True)[0][i][j][k]
        kinetic_energy = -1 * 0.5 * (jacobian_square_sum +
                                     hessian) / (self.batch_size)
        return kinetic_energy


class FerminetModel(TorchModel):
    """A deep-learning based Variational Monte Carlo method [1]_ for calculating the ab-initio
    solution of a many-electron system.

    This model aims to calculate the ground state energy of a multi-electron system
    using a baseline solution as the Hartree-Fock. An MCMC technique is used to sample
    electrons and DNNs are used to caluclate the square magnitude of the wavefunction,
    in which electron-electron repulsions also are included in the calculation(in the
    form of Jastrow factor envelopes). The model requires only the nucleus' coordinates
    as input.

    This method is based on the following paper:

    Example
    -------
    >>> from deepchem.models.torch_models.Ferminet import FerminetModel
    >>> H2_molecule = [['H', [0, 0, 0]], ['H', [0, 0, 0.748]]]
    >>> mol = FerminetModel(H2_molecule, spin=0, ion_charge=0, training='pretraining')
    >>> mol.train(nb_epoch=3)
    >>> print(mol.model.psi_up.size())
    torch.Size([1, 1])

    References
    ----------
    .. [1] Spencer, James S., et al. Better, Faster Fermionic Neural Networks. arXiv:2011.07125, arXiv, 13 Nov. 2020. arXiv.org, http://arxiv.org/abs/2011.07125.

    Note
    ----
    This class requires pySCF to be installed.
    """

    def __init__(self,
                 nucleon_coordinates: List[List],
                 spin: int,
                 ion_charge: int,
                 seed: Optional[int] = None,
                 batch_no: int = 8,
                 random_walk_steps: int = 10,
                 steps_per_update: int = 10,
                 tasks: str = 'pretraining'):
        """
    Parameters:
    -----------
    nucleon_coordinates: List[List]
      A list containing nucleon coordinates as the values with the keys as the element's symbol.
    spin: int
      The total spin of the molecule system.
    ion_charge: int
      The total charge of the molecule system.
    seed_no: int, optional (default None)
      Random seed to use for electron initialization.
    batch_no: int, optional (default 8)
      Number of batches of the electron's positions to be initialized.
    random_walk_steps: int (default 10)
        Number of random walk steps to be performed in a single move.
    steps_per_update: int (default: 10)
        Number of steps after which the electron sampler should update the electron parameters.
    tasks: str (default: 'pretraining')
        The type of task to be performed - 'pretraining', 'training'

    Attributes:
    -----------
    nucleon_pos: np.ndarray
        numpy array value of nucleon_coordinates
    electron_no: np.ndarray
        Torch tensor containing electrons for each atom in the nucleus
    molecule: ElectronSampler
        ElectronSampler object which performs MCMC and samples electrons
    loss_value: torch.Tensor (default torch.tensor(0))
        torch tensor storing the loss value from the last iteration
    """
        self.nucleon_coordinates = nucleon_coordinates
        self.seed = seed
        self.batch_no = batch_no
        self.spin = spin
        self.ion_charge = ion_charge
        self.batch_no = batch_no
        self.random_walk_steps = random_walk_steps
        self.steps_per_update = steps_per_update
        self.loss_value: torch.Tensor = torch.tensor(0)
        self.tasks = tasks

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
        charge: np.ndarray = self.electron_no.reshape(
            np.shape(self.electron_no)[0])
        self.nucleon_pos: np.ndarray = np.array(nucleons)
        electro_neg = np.array(electronegativity)

        # Initialization for ionic molecules
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

        if self.spin >= 0:
            self.up_spin = (total_electrons + 2 * self.spin) // 2
            self.down_spin = total_electrons - self.up_spin
        else:
            self.down_spin = (total_electrons - 2 * self.spin) // 2
            self.up_spin = total_electrons - self.down_spin

        if self.up_spin - self.down_spin != self.spin:
            raise ValueError("Given spin is not feasible")

        nucl = torch.from_numpy(self.nucleon_pos)
        self.model = Ferminet(nucl,
                              spin=(self.up_spin, self.down_spin),
                              nuclear_charge=torch.Tensor(charge),
                              batch_size=self.batch_no)

        self.molecule: ElectronSampler = ElectronSampler(
            batch_no=self.batch_no,
            central_value=self.nucleon_pos,
            seed=self.seed,
            f=lambda x: self.random_walk(x
                                        ),  # Will be replaced in successive PR
            steps=self.random_walk_steps,
            steps_per_update=self.steps_per_update
        )  # sample the electrons using the electron sampler
        self.molecule.gauss_initialize_position(
            self.electron_no)  # initialize the position of the electrons
        self.prepare_hf_solution()
        super(FerminetModel, self).__init__(
            self.model,
            loss=torch.nn.MSELoss())  # will update the loss in successive PR

    def evaluate_hf(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Helper function to calculate orbital values at sampled electron's position.

        Parameters:
        -----------
        x: np.ndarray
            Contains the sampled electrons coordinates in a numpy array.

        Returns:
        --------
        2 numpy arrays containing the up-spin and down-spin orbitals in a numpy array respectively.
        """
        x = np.reshape(x, [-1, 3 * (self.up_spin + self.down_spin)])
        leading_dims = x.shape[:-1]
        x = np.reshape(x, [-1, 3])
        coeffs = self.mf.mo_coeff
        gto_op = 'GTOval_sph'
        ao_values = self.mol.eval_gto(gto_op, x)
        mo_values = tuple(np.matmul(ao_values, coeff) for coeff in coeffs)
        mo_values_list = [
            np.reshape(mo, leading_dims + (self.up_spin + self.down_spin, -1))
            for mo in mo_values
        ]
        return mo_values_list[0][
            ..., :self.up_spin, :self.up_spin], mo_values_list[1][
                ..., self.up_spin:, :self.down_spin]

    def prepare_hf_solution(self):
        """Prepares the HF solution for the molecule system which is to be used in pretraining
        """
        try:
            import pyscf
        except ModuleNotFoundError:
            raise ImportError("This module requires pySCF")

        molecule = ""
        for i in range(len(self.nucleon_pos)):
            molecule = molecule + self.nucleon_coordinates[i][0] + " " + str(
                self.nucleon_coordinates[i][1][0]) + " " + str(
                    self.nucleon_coordinates[i][1][1]) + " " + str(
                        self.nucleon_coordinates[i][1][2]) + ";"
        self.mol = pyscf.gto.Mole(atom=molecule, basis='sto-3g')
        self.mol.parse_arg = False
        self.mol.unit = 'Bohr'
        self.mol.spin = (self.up_spin - self.down_spin)
        self.mol.charge = self.ion_charge
        self.mol.build(parse_arg=False)
        self.mf = pyscf.scf.UHF(self.mol)
        _ = self.mf.kernel()

    def random_walk(self, x: np.ndarray) -> np.ndarray:
        """
        Function to be passed on to electron sampler for random walk and gets called at each step of sampling

        Parameters
        ----------
        x: np.ndarray
            contains the sampled electrons coordinate in the shape (batch_size,number_of_electrons*3)

        Returns:
        --------
        A numpy array containing the joint probability of the hartree fock and the sampled electron's position coordinates
        """
        output = self.model.forward(x)
        np_output = output.detach().cpu().numpy()
        up_spin_mo, down_spin_mo = self.evaluate_hf(x)
        hf_product = np.prod(
            np.diagonal(up_spin_mo, axis1=1, axis2=2)**2, axis=1) * np.prod(
                np.diagonal(down_spin_mo, axis1=1, axis2=2)**2, axis=1)
        self.model.loss(up_spin_mo, down_spin_mo, pretrain=True)
        return np.log(hf_product + np_output**2) + np.log(0.5)

    def train(self,
              nb_epoch: int = 200,
              lr: float = 0.0075,
              weight_decay: float = 0.0001):
        """
        function to run training or pretraining.

        Parameters
        ----------
        nb_epoch: int (default: 200)
            contains the number of pretraining steps to be performed
        lr : float (default: 0.0075)
            contains the learning rate for the model fitting
        weight_decay: float (default: 0.0001)
            contains the weight_decay for the model fitting
        """
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=lr,
                                     weight_decay=weight_decay)
        if (self.tasks == 'pretraining'):
            for _ in range(nb_epoch):
                optimizer.zero_grad()
                self.molecule.move()
                self.loss_value = (torch.mean(self.model.running_diff) /
                                   self.random_walk_steps)
                self.loss_value.backward()
                optimizer.step()
                self.model.running_diff = torch.zeros(self.batch_no)
