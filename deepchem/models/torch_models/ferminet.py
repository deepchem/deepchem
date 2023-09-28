"""
Implementation of the Ferminet class in pytorch
"""
import logging
from typing import List, Optional, Tuple, Dict, Callable, Any
from deepchem.utils.typing import LossFn
import time
import torch.nn as nn
from rdkit import Chem
import numpy as np
from deepchem.utils.molecule_feature_utils import ALLEN_ELECTRONEGATIVTY
from deepchem.models.torch_models.modular import ModularTorchModel
import torch
from deepchem.models.optimizers import LearningRateSchedule
from deepchem.models.torch_models.layers import FerminetElectronFeature, FerminetEnvelope

from deepchem.utils.electron_sampler import ElectronSampler

logger = logging.getLogger(__name__)


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
        ferminet_layer: torch.nn.ModuleList
            Modulelist containing the ferminet electron feature layer
        ferminet_layer_envelope: torch.nn.ModuleList
            Modulelist containing the ferminet envelope electron feature layer
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

        self.ferminet_layer.append(
            FerminetElectronFeature(self.n_one, self.n_two,
                                    self.nucleon_pos.size()[0], self.batch_size,
                                    self.total_electron,
                                    [self.spin[0], self.spin[1]]))
        self.ferminet_layer_envelope.append(
            FerminetEnvelope(self.n_one, self.n_two, self.total_electron,
                             self.batch_size, [self.spin[0], self.spin[1]],
                             self.nucleon_pos.size()[0], self.determinant))

    def forward(self, input) -> torch.Tensor:
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
        self.input = torch.from_numpy(input)
        self.input.requires_grad = True
        self.input = self.input.reshape((self.batch_size, -1, 3))
        two_electron_vector = self.input.unsqueeze(1) - self.input.unsqueeze(2)
        two_electron_distance = torch.norm(two_electron_vector,
                                           dim=3).unsqueeze(3)
        two_electron = torch.cat((two_electron_vector, two_electron_distance),
                                 dim=3)
        two_electron = torch.reshape(
            two_electron,
            (self.batch_size, self.total_electron, self.total_electron, -1))

        one_electron_vector = self.input.unsqueeze(
            1) - self.nucleon_pos.unsqueeze(1)
        one_electron_distance = torch.norm(one_electron_vector, dim=3)
        one_electron = torch.cat(
            (one_electron_vector, one_electron_distance.unsqueeze(-1)), dim=3)
        one_electron = torch.reshape(one_electron.permute(0, 2, 1, 3),
                                     (self.batch_size, self.total_electron, -1))
        one_electron_vector_permuted = one_electron_vector.permute(0, 2, 1, 3)

        one_electron, _ = self.ferminet_layer[0].forward(
            one_electron.to(torch.float32), two_electron.to(torch.float32))
        psi, self.psi_up, self.psi_down = self.ferminet_layer_envelope[
            0].forward(one_electron, one_electron_vector_permuted)
        return psi


class FerminetModel(ModularTorchModel):
    """A deep-learning based Variational Monte Carlo method [1]_ for calculating the ab-initio
    solution of a many-electron system.

    This model aims to calculate the ground state energy of a multi-electron system
    using a baseline solution as the Hartree-Fock. An MCMC technique is used to sample
    electrons and DNNs are used to caluclate the square magnitude of the wavefunction,
    in which electron-electron repulsions also are included in the calculation(in the
    form of Jastrow factor envelopes). The model requires only the nucleus' coordinates
    as input.

    This method is based on the following paper:

    References
    ----------
    .. [1] Spencer, James S., et al. Better, Faster Fermionic Neural Networks. arXiv:2011.07125, arXiv, 13 Nov. 2020. arXiv.org, http://arxiv.org/abs/2011.07125.

    Note
    ----
    This class requires pySCF to be installed.

    Example
    -------
    >>> from deepchem.models.torch_models.ferminet import FerminetModel
    >>> import torch
    >>> import tempfile
    >>> H2_molecule = [['H', [0, 0, 0]], ['H', [0, 0, 0.748]]]
    >>> pretrain_model = FerminetModel(H2_molecule, spin=0, ion_charge=0, tasks='pretraining', model_dir=tempdir.name)
    >>> pretraining_loss = pretrain_model.train(nb_epoch=1)
    >>> pretrain_model.save_checkpoint()
    >>> finetune_model = FerminetModel(H2_molecule, spin=0, ion_charge=0, tasks='pretraining', model_dir=tempdir.name)
    >>> finetune_model.restore(components=['electron-features'])
    >>> finetuning_loss = finetune_model.train()
    """

    def __init__(self,
                 nucleon_coordinates: List[List],
                 spin: int,
                 ion_charge: int,
                 seed: Optional[int] = None,
                 batch_no: int = 8,
                 random_walk_steps=10,
                 steps_per_update=10,
                 task='pretraining',
                 learning_rate: float = 1e-3,
                 **kwargs):
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
    task: str (default: 'pretraining')
        The type of task to run - 'pretraining' or 'training'
    learning_rate: flaot (default: 1e-3)
        The learning rate to be used by the Adam optimizer

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
    running_diff: torch.Tensor
        torch tensor containing the loss which gets updated for each random walk performed
    pretrain_criterion:torch.nn.MSELoss
        torch.nn.MSELoss which serves as the criterion for pretrain tasks
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
        self.running_diff: torch.Tensor = torch.tensor(self.batch_no)
        self.task = task
        self.pretrain_criterion: torch.nn.MSELoss = torch.nn.MSELoss()
        self.learning_rate = learning_rate

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

        self.nucl = torch.from_numpy(self.nucleon_pos)
        self.model = self.build_model()
        self._pytorch_optimizer = torch.optim.Adam(
            self.model.parameters(),  # type: ignore
            lr=self.learning_rate)  # type: ignore
        self._lr_schedule = None
        self.components = self.build_components()
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
        super().__init__(self.model, self.components,
                         **kwargs)  # will update the loss in successive PR

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
        self.loss_func(inputs=[self.model.psi_up, self.model.psi_down],
                       labels=[
                           torch.from_numpy(up_spin_mo).unsqueeze(1),
                           torch.from_numpy(down_spin_mo).unsqueeze(1)
                       ],
                       weights=[None])
        return np.log(hf_product + np_output**2) + np.log(0.5)

    def build_components(self) -> dict:
        """
        Build the components of the model. Ferminet consists of 2 torch layers - electron-features and electron-envelope

        Components list, type and description:
        --------------------------------------
        electron-features layer: Geometry independent layers, calculates one and  two electron features

        electron-envelope: Geometry dependent layers, calculates the envelope function and orbital values
        """
        components: Dict[str, nn.Module] = {}
        components['electron-features'] = self.model.ferminet_layer[0]
        components['electron-envelope'] = self.model.ferminet_layer_envelope[0]

        return components

    def build_model(self) -> nn.Module:
        """
        Builds the Ferminet model
        """
        model = Ferminet(self.nucl,
                         spin=(self.up_spin, self.down_spin),
                         nuclear_charge=torch.Tensor(self.charge),
                         batch_size=self.batch_no)
        return model

    def loss_func(self, inputs, labels, weights) -> None:
        if self.task == 'pretraining':
            self.running_diff = self.running_diff + self.pretrain_criterion(
                inputs[0], labels[0].float()) + self.pretrain_criterion(
                    inputs[0], labels[0].float())

    def train(self,
              nb_epoch=10,
              burn_in: int = 0,
              max_checkpoints_to_keep: int = 5,
              checkpoint_interval: int = 1000,
              variables: Optional[List[torch.nn.Parameter]] = None,
              loss: Optional[LossFn] = None,
              all_losses: Optional[List[torch.tensor]] = None) -> torch.tensor:
        """Function to run pretraining or training

        Parameters
        ----------
        nb_epoch: int
            the number of epochs to train for
        burn-in: int
            the number epochs for burn-in, used at the start of pretraining and training to get the best possible starting point
        max_checkpoints_to_keep: int
            the maximum number of checkpoints to keep.  Older checkpoints are discarded.
        checkpoint_interval: int
            the frequency at which to write checkpoints, measured in training steps.
            Set this to 0 to disable automatic checkpointing.
        variables: list of torch.nn.Parameter
            the variables to train.  If None (the default), all trainable variables in
            the model are used.
        loss: function
            a function of the form f(outputs, labels, weights) that computes the loss
            for each batch.  If None (the default), the model's standard loss function
            is used.
        all_losses: Optional[List[float]], optional (default None)
            If specified, all logged losses are appended into this list. Note that
            you can call `fit()` repeatedly with the same list and losses will
            continue to be appended.

        Returns
        -------
        The average loss tensor over the most recent checkpoint interval
        """

        self._ensure_built()
        self.model.train()

        if variables is None:
            optimizer = self._pytorch_optimizer
            lr_schedule = self._lr_schedule
        else:
            var_key = tuple(variables)
            if var_key in self._optimizer_for_vars:
                optimizer, lr_schedule = self._optimizer_for_vars[var_key]
            else:
                optimizer = self.optimizer._create_pytorch_optimizer(variables)
                if isinstance(self.optimizer.learning_rate,
                              LearningRateSchedule):
                    lr_schedule = self.optimizer.learning_rate._create_pytorch_schedule(
                        optimizer)
                else:
                    lr_schedule = None
                self._optimizer_for_vars[var_key] = (optimizer, lr_schedule)
        time1 = time.time()

        # Execute the loss function, accumulating the gradients.
        for _ in range(burn_in):
            self.molecule.gauss_initialize_position(
                self.electron_no)  # initialize the position of the electrons

        for _ in range(nb_epoch):
            if self.task == 'pretraining':
                optimizer.zero_grad()
                self.molecule.move()
                self.loss_value = (torch.mean(self.running_diff) /
                                   self.random_walk_steps)
                self.loss_value.backward()
                optimizer.step()
                if lr_schedule is not None:
                    lr_schedule.step()
                self.running_diff = torch.zeros(self.batch_no)

            # TODO: self.task == 'training'

            self._global_step += 1
            current_step = self._global_step
            should_log = (current_step % 1 == 0)
            # Report progress and write checkpoints.
            if should_log:
                logger.info('Ending global_step %d: Average loss %.10f' %
                            (current_step, self.loss_value))
                if all_losses is not None:
                    all_losses.append(self.loss_value)
                # Capture the last avg_loss in case of return since we're resetting to 0 now
            if checkpoint_interval > 0 and current_step % checkpoint_interval == checkpoint_interval - 1:
                self.save_checkpoint(max_checkpoints_to_keep)
            if self.tensorboard and should_log:
                self._log_scalar_to_tensorboard('loss', self.loss_value,
                                                current_step)
            if (self.wandb_logger is not None) and should_log:
                all_data = dict({'train/loss': self.loss_value})
                self.wandb_logger.log_data(all_data, step=current_step)

            if checkpoint_interval > 0:
                self.save_checkpoint(max_checkpoints_to_keep)
            time2 = time.time()
            logger.info("TIMING: model fitting took %0.3f s" % (time2 - time1))
            time1 = time2
        return self.loss_value

    def restore(  # type: ignore
            self,
            components: Optional[List[str]] = None,
            checkpoint: Optional[str] = None,
            model_dir: Optional[str] = None,
            map_location: Optional[torch.device] = None) -> None:
        """
        Overriden function for restore
        """
        if checkpoint is None:
            checkpoints = sorted(self.get_checkpoints(model_dir))
            if len(checkpoints) == 0:
                raise ValueError('No checkpoint found')
            checkpoint = checkpoints[0]
        data = torch.load(checkpoint, map_location=map_location)
        for name, state_dict in data.items():
            if name != 'model' and name in self.components.keys():
                self.components[name].load_state_dict(state_dict)

        self.build_model()
