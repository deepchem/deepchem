"""
Implementation of the Ferminet class in pytorch
"""
import logging
from typing import List, Optional, Tuple
# import torch.nn as nn
from rdkit import Chem
import numpy as np
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
    >>> from deepchem.models.torch_models.ferminet import Ferminet
    >>> import torch
    >>> H2_molecule =  torch.Tensor([[0, 0, 0.748], [0, 0, 0]])
    >>> H2_charge = torch.Tensor([[1], [1]])
    >>> model = Ferminet(nucleon_pos=H2_molecule, nuclear_charge=H2_charge, spin=(1,1), batch_size=1)
    >>> electron = torch.rand(1, 2*3)
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

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        forward function

        Parameters:
        -----------
        input: torch.Tensor
            contains the sampled electrons coordinate in the shape (batch_size,number_of_electrons*3)

        Returns:
        --------
        psi: torch.Tensor
            contains the wavefunction - 'psi' value. It is in the shape (batch_size), where each row corresponds to the solution of one of the batches
        """
        # creating one and two electron features
        eps = torch.tensor(1e-36)
        self.input = input.reshape((self.batch_size, -1, 3))
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
        one_electron_vector_permuted = one_electron_vector.permute(0, 2, 1,
                                                                   3).float()
        # setting the fermient layer and fermient envelope layer batch size to be that of the current batch size of the model. This enables for vectorized calculations of hessians and jacobians.
        self.ferminet_layer[0].batch_size = self.batch_size
        self.ferminet_layer_envelope[0].batch_size = self.batch_size
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
        if pretrain[0]:
            psi_up_mo_torch = torch.from_numpy(psi_up_mo).unsqueeze(1)
            psi_down_mo_torch = torch.from_numpy(psi_down_mo).unsqueeze(1)
            self.running_diff = self.running_diff.float() + criterion(
                self.psi_up.float(),
                psi_up_mo_torch.float()).float() + criterion(
                    self.psi_down.float(), psi_down_mo_torch.float()).float()
        else:
            energy = self.calculate_electron_electron(
            ) - self.calculate_electron_nuclear(
            ) + self.nuclear_nuclear_potential + self.calculate_kinetic_energy(
            )
            return energy.detach()

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
        potential = (torch.sum(potential) / 2).unsqueeze(0)
        return potential.detach()

    def calculate_electron_nuclear(self,):
        """
        Function to calculate the expected electron-nuclear potential term per batch
        nuclear-electron potential term = Zi/|Ri-rj|, rj is the electron coordinates, Ri is the nuclear coordinates, Zi is the nuclear charge

        Returns:
        --------
        A torch tensor of a scalar value containing the electron-nuclear potential term.
        """

        potential = torch.sum(torch.sum(
            (1 / torch.cdist(self.input.float(), self.nucleon_pos.float())) *
            self.nuclear_charge,
            axis=-1),
                              axis=-1)
        return potential.detach()

    def calculate_electron_electron(self,):
        """
        Function to calculate the expected electron-nuclear potential term per batch
        nuclear-electron potential term = 1/|ri-rj|, ri, rj is the electron coordinates

        Returns:
        --------
        A torch tensor of a scalar value containing the electron-electron potential term.
        """
        potential = torch.sum(torch.sum(torch.nan_to_num(
            (1 / torch.cdist(self.input.float(), self.input.float())),
            posinf=0.0,
            neginf=0.0),
                                        axis=-1),
                              axis=-1) / 2
        return potential.detach()

    def calculate_kinetic_energy(self,):
        """
        Function to calculate the expected kinetic energy term per batch
        It is calculated via:
        \sum_{ri}^{}[(\pdv[]{log|\Psi|}{(ri)})^2 + \pdv[2]{log|\Psi|}{(ri)}]

        Returns:
        --------
        A torch tensor of a scalar value containing the electron-electron potential term.
        """
        # using functorch to calcualte hessian and jacobian in one go
        # using index tensors to index out the hessian elemennts corresponding to the same variable (cross-variable derivatives are ignored)
        i = torch.arange(self.batch_size).view(self.batch_size, 1, 1, 1, 1)
        j = torch.arange(self.total_electron).view(1, self.total_electron, 1, 1,
                                                   1)
        k = torch.arange(3).view(1, 1, 3, 1, 1)
        # doing all the calculation and detaching from graph to save memory, which allows larger batch size
        # cloning self.input which will serve as the new input for the vectorized functions.
        input = torch.clone(self.input).detach()
        # lambda function for calculating the log of absolute value of the wave function.
        # using jacrev for the jacobian and jacrev twice for to calculate the hessian. The functorch's hessian function if directly used does not give stable results.
        jac = torch.func.jacrev(lambda x: torch.log(torch.abs(self.forward(x))))
        hess = torch.func.jacrev(jac)
        # making the batch size temporarily as 1 for the vectorization of hessian and jacobian.
        tmp_batch_size = self.batch_size
        self.batch_size = 1
        jacobian_square_sum = torch.sum(torch.pow(
            torch.func.vmap(jac)(input).detach().squeeze(1), 2),
                                        axis=(1, 2))
        vectorized_hessian = torch.func.vmap(hess)
        hessian_sum = torch.sum(
            vectorized_hessian(input).detach().squeeze(1)[i, j, k, j, k],
            axis=(1, 2)).squeeze(1).squeeze(1)
        self.batch_size = tmp_batch_size
        kinetic_energy = -1 * 0.5 * (jacobian_square_sum + hessian_sum)
        return kinetic_energy.detach()


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
    >>> from deepchem.models.torch_models.ferminet import FerminetModel
    >>> H2_molecule = [['H', [0, 0, 0]], ['H', [0, 0, 0.748]]]
    >>> mol = FerminetModel(H2_molecule, spin=0, ion_charge=0, tasks='pretraining') # doctest: +IGNORE_RESULT
    converged SCF energy = -0.895803169899509  <S^2> = 0  2S+1 = 1
     ** Mulliken pop alpha/beta on meta-lowdin orthogonal AOs **
     ** Mulliken pop       alpha | beta **
    pop of  0 H 1s        0.50000 | 0.50000
    pop of  1 H 1s        0.50000 | 0.50000
    In total             1.00000 | 1.00000
     ** Mulliken atomic charges   ( Nelec_alpha | Nelec_beta ) **
    charge of  0H =      0.00000  (     0.50000      0.50000 )
    charge of  1H =      0.00000  (     0.50000      0.50000 )
    converged SCF energy = -0.895803169899509  <S^2> = 0  2S+1 = 1
    >>> mol.train(nb_epoch=3)
    >>> print(mol.model.psi_up.size())
    torch.Size([8, 16, 1, 1])

    References
    ----------
    .. [1] Spencer, James S., et al. Better, Faster Fermionic Neural Networks. arXiv:2011.07125, arXiv, 13 Nov. 2020. arXiv.org, http://arxiv.org/abs/2011.07125.

    Note
    ----
    This class requires pySCF to be installed.
    """.replace('+IGNORE_RESULT', '+ELLIPSIS\n<...>')

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

        table = Chem.GetPeriodicTable()
        index = 0
        for i in self.nucleon_coordinates:
            atomic_num = table.GetAtomicNumber(i[0])
            no_electrons.append([atomic_num])
            nucleons.append(i[1])
            index += 1

        self.electron_no: np.ndarray = np.array(no_electrons)
        charge: np.ndarray = self.electron_no.reshape(
            np.shape(self.electron_no)[0])
        self.nucleon_pos: np.ndarray = np.array(nucleons)

        # Initialization for ionic molecules
        if np.sum(self.electron_no) < self.ion_charge:
            raise ValueError("Given charge is not initializable")

        total_electrons = np.sum(self.electron_no) - self.ion_charge

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
            self.electron_no,
            stddev=1.0)  # initialize the position of the electrons
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
        self.mol = pyscf.gto.Mole(atom=molecule, basis='sto-6g')
        self.mol.parse_arg = False
        self.mol.unit = 'Bohr'
        self.mol.spin = (self.up_spin - self.down_spin)
        self.mol.charge = self.ion_charge
        self.mol.build(parse_arg=False)
        self.mf = pyscf.scf.UHF(self.mol)
        self.mf.run()
        dm = self.mf.make_rdm1()
        _, chg = pyscf.scf.uhf.mulliken_meta(self.mol, dm)
        excess_charge = np.array(chg)
        tmp_charge = self.ion_charge
        while tmp_charge != 0:
            if (tmp_charge < 0):
                charge_index = np.argmin(excess_charge)
                tmp_charge += 1
                self.electron_no[charge_index] += 1
                excess_charge[charge_index] += 1
            elif (tmp_charge > 0):
                charge_index = np.argmax(excess_charge)
                tmp_charge -= 1
                self.electron_no[charge_index] -= 1
                excess_charge[charge_index] -= 1

        self.molecule.gauss_initialize_position(
            self.electron_no,
            stddev=2.0)  # initialize the position of the electrons
        _ = self.mf.kernel()

    def random_walk(self, x: np.ndarray):
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
        x_torch = torch.from_numpy(x).view(self.batch_no, -1, 3)
        if self.tasks == 'pretraining':
            x_torch.requires_grad = True
        else:
            x_torch.requires_grad = False
        output = self.model.forward(x_torch)
        np_output = output.detach().cpu().numpy()

        if self.tasks == 'pretraining':
            up_spin_mo, down_spin_mo = self.evaluate_hf(x)
            hf_product = np.prod(
                np.diagonal(up_spin_mo, axis1=1, axis2=2), axis=1) * np.prod(
                    np.diagonal(down_spin_mo, axis1=1, axis2=2), axis=1)
            self.model.loss(up_spin_mo, down_spin_mo)
            np_output[:int(self.batch_no / 2)] = hf_product[:int(self.batch_no /
                                                                 2)]
            return 2 * np.log(np.abs(np_output))

        if self.tasks == 'burn':
            return 2 * np.log(np.abs(np_output))

        if self.tasks == 'training':
            energy = self.model.loss(pretrain=[False])
            self.energy_sampled: torch.Tensor = torch.cat(
                (self.energy_sampled, energy.unsqueeze(0)))
            return 2 * np.log(np.abs(np_output))

    def prepare_train(self, burn_in: int = 100):
        """
        Function to perform burn-in and to change the model parameters for training.

        Parameters
        ----------
        burn_in:int (default: 100)
            number of steps for to perform burn-in before the aactual training.
        """
        self.tasks = 'burn'
        self.molecule.gauss_initialize_position(self.electron_no, stddev=1.0)
        tmp_x = self.molecule.x
        for _ in range(burn_in):
            self.molecule.move(stddev=0.02)
            self.molecule.x = tmp_x
        self.molecule.move(stddev=0.02)
        self.tasks = 'training'

    def train(self,
              nb_epoch: int = 200,
              lr: float = 0.002,
              weight_decay: float = 0,
              std: float = 0.08,
              std_init: float = 0.02,
              steps_std: int = 100):
        """
        function to run training or pretraining.

        Parameters
        ----------
        nb_epoch: int (default: 200)
            contains the number of pretraining steps to be performed
        lr : float (default: 0.002)
            contains the learning rate for the model fitting
        weight_decay: float (default: 0.002)
            contains the weight_decay for the model fitting
        std: float (default: 0.08)
            The standard deviation for the electron update during training
        std_init: float (default: 0.02)
            The standard deviation for the electron update during pretraining
        steps_std: float (default 100)
            The number of steps for standard deviation increase
        """

        # hook function below is an efficient way modifying the gradients on the go rather than looping
        def energy_hook(grad, random_walk_steps):
            """
            hook function to modify the gradients
            """
            # using non-local variables as a means of parameter passing
            nonlocal energy_local, energy_mean
            new_grad = (2 / random_walk_steps) * (
                (energy_local - energy_mean) * grad)
            return new_grad.float()

        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=lr,
                                     weight_decay=weight_decay)

        if (self.tasks == 'pretraining'):
            for iteration in range(nb_epoch):
                optimizer.zero_grad()
                accept = self.molecule.move(stddev=std_init)
                if iteration % steps_std == 0:
                    if accept > 0.55:
                        std_init *= 1.1
                    else:
                        std_init /= 1.1
                self.loss_value = (torch.mean(self.model.running_diff) /
                                   self.random_walk_steps)
                self.loss_value.backward()
                optimizer.step()
                logging.info("The loss for the pretraining iteration " +
                             str(iteration) + " is " +
                             str(self.loss_value.item()))
                self.model.running_diff = torch.zeros(self.batch_no)

        if (self.tasks == 'training'):
            energy_local = None
            optimizer = torch.optim.Adam(self.model.parameters(),
                                         lr=lr,
                                         weight_decay=weight_decay)
            self.final_energy = torch.tensor(0.0)
            with torch.no_grad():
                hooks = list(
                    map(
                        lambda param: param.
                        register_hook(lambda grad: energy_hook(
                            grad, self.random_walk_steps)),
                        self.model.parameters()))
            for iteration in range(nb_epoch):
                optimizer.zero_grad()
                self.energy_sampled = torch.tensor([])
                # the move function calculates the energy of sampled electrons and samples new set of electrons (does not calculate loss)
                accept = self.molecule.move(stddev=std)
                if iteration % steps_std == 0:
                    if accept > 0.55:
                        std_init *= 1.2
                    else:
                        std_init /= 1.2
                median, _ = torch.median(self.energy_sampled, 0)
                variance = torch.mean(torch.abs(self.energy_sampled - median))
                # clipping local energies which are away 5 times the variance from the median
                clamped_energy = torch.clamp(self.energy_sampled,
                                             max=median + 5 * variance,
                                             min=median - 5 * variance)
                energy_mean = torch.mean(clamped_energy)
                logging.info("The mean energy for the training iteration " +
                             str(iteration) + " is " + str(energy_mean.item()))
                self.final_energy = self.final_energy + energy_mean
                # using the sampled electrons from the electron sampler for bacward pass and modifying gradients
                sample_history = torch.from_numpy(
                    self.molecule.sampled_electrons).view(
                        self.random_walk_steps, self.batch_no, -1, 3)
                optimizer.zero_grad()
                for i in range(self.random_walk_steps):
                    # going through each step of random walk and calculating the modified gradients with local energies
                    input_electron = sample_history[i]
                    input_electron.requires_grad = True
                    energy_local = torch.mean(clamped_energy[i])
                    self.model.forward(input_electron)
                    self.loss_value = torch.mean(
                        torch.log(torch.abs(self.model.psi)))
                    self.loss_value.backward()
                optimizer.step()
            self.final_energy = self.final_energy / nb_epoch
            list(map(lambda hook: hook.remove(), hooks))
