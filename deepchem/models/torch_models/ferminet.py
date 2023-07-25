"""
Implementation of the Ferminet class in pytorch
"""

from typing import List, Optional
# import torch.nn as nn
from rdkit import Chem
import numpy as np
from deepchem.utils.molecule_feature_utils import ALLEN_ELECTRONEGATIVTY
from deepchem.utils.geometry_utils import compute_pairwise_distances
from deepchem.models.torch_models import TorchModel
from deepchem.models.losses import L2Loss
import deepchem.models.optimizers as optimizers
import torch

from deepchem.utils.electron_sampler import ElectronSampler

# TODO look for the loss function(Hamiltonian)


def test_f(x: np.ndarray) -> np.ndarray:
    # dummy function which can be passed as the parameter f. f gives the log probability
    # TODO replace this function with forward pass of the model in future
    return 2 * np.log(np.random.uniform(low=0, high=1.0, size=np.shape(x)[0]))


class Ferminet(torch.nn.Module):
    """Approximates the log probability of the wave function of a molecule system using DNNs.
  """

    def __init__(self,
                 nucleon_pos: torch.Tensor,
                 nuclear_charge: torch.Tensor,
                 spin: tuple,
                 inter_atom: torch.Tensor,
                 n_one: List = [256, 256, 256, 256],
                 n_two: List = [32, 32, 32, 32],
                 determinant: int = 16) -> None:
        """
    Parameters:
    -----------
    nucleon_pos: torch.Tensor
        Torch tensor containing nucleus information of the molecule
    nuclear_charge: torch.Tensor
        Torch tensor containing the number of electron for each atom in the molecule
    spin: tuple
        Tuple in the format of (up_spin, down_spin)
    inter_atom: torch.Tensor
        Torch tensor containing the pairwise distances between the atoms in the molecule
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
        self.inter_atom = inter_atom
        self.n_one = n_one
        self.n_two = n_two
        self.determinant = determinant


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

    References
    ----------
    .. [1] Spencer, James S., et al. Better, Faster Fermionic Neural Networks. arXiv:2011.07125, arXiv, 13 Nov. 2020. arXiv.org, http://arxiv.org/abs/2011.07125.

    Note
    ----
    This class requires pySCF to be installed.
    """

    def __init__(
        self,
        nucleon_coordinates: List[List],
        spin: int,
        ion_charge: int,
        seed: Optional[int] = None,
        batch_no: int = 10,
        pretrain=True,
    ):
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
    batch_no: int, optional (default 10)
      Number of batches of the electron's positions to be initialized.

    Attributes:
    -----------
    nucleon_pos: np.ndarray
        numpy array value of nucleon_coordinates
    electron_no: np.ndarray
        Torch tensor containing electrons for each atom in the nucleus
    molecule: ElectronSampler
        ElectronSampler object which performs MCMC and samples electrons
    """
        self.nucleon_coordinates = nucleon_coordinates
        self.seed = seed
        self.batch_no = batch_no
        self.spin = spin
        self.ion_charge = ion_charge
        self.batch_no = batch_no

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
        model = Ferminet(nucl,
                         spin=(self.up_spin, self.down_spin),
                         nuclear_charge=torch.tensor(charge),
                         inter_atom=torch.tensor(
                             compute_pairwise_distances(self.nucleon_pos,
                                                        self.nucleon_pos)))

        self.molecule: ElectronSampler = ElectronSampler(
            batch_no=self.batch_no,
            central_value=self.nucleon_pos,
            seed=self.seed,
            f=lambda x: test_f(x),  # Will be replaced in successive PR
            steps=1000,
            steps_per_update=20
        )  # sample the electrons using the electron sampler
        self.molecule.gauss_initialize_position(
            self.electron_no)  # initialize the position of the electrons
        adam = optimizers.AdamW()
        super(FerminetModel, self).__init__(
            model, optimizer=adam,
            loss=L2Loss())  # will update the loss in successive PR

    def prepare_hf_solution(self, x: np.ndarray) -> np.ndarray:
        """Prepares the HF solution for the molecule system which is to be used in pretraining

        Parameters
        ----------
        x: np.ndarray
        Numpy array of shape (number of electrons,3), which indicates the sampled electron's positions

        Returns
        -------
        hf_value: np.ndarray
        Numpy array of shape (number of electrons, number of electrons ) where ith row & jth value corresponds to the ith hartree fock orbital at the jth electron's coordinate
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
        mol = pyscf.gto.Mole(atom=molecule, basis='sto-3g')
        mol.parse_arg = False
        mol.unit = 'Bohr'
        mol.spin = (self.up_spin - self.down_spin)
        mol.charge = self.ion_charge
        mol.build(parse_arg=False)
        mf = pyscf.scf.RHF(mol)
        mf.kernel()

        coefficients_all = mf.mo_coeff[:, :mol.nelectron]
        # Get the positions of all the electrons
        electron_positions = mol.atom_coords()[:mol.nelectron]
        # Evaluate all molecular orbitals at the positions of all the electrons
        orbital_values = np.dot(mol.eval_gto("GTOval", electron_positions),
                                coefficients_all)
        return orbital_values
