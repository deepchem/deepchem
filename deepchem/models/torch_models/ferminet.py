"""
Implementation of the Ferminet class in pytorch
"""

from typing import List, Optional, Any, Tuple
# import torch.nn as nn
from rdkit import Chem
import numpy as np
from deepchem.utils.molecule_feature_utils import ALLEN_ELECTRONEGATIVTY
from deepchem.utils.geometry_utils import compute_pairwise_distances
from deepchem.models.torch_models import TorchModel
import deepchem.models.optimizers as optimizers
import torch

from deepchem.utils.electron_sampler import ElectronSampler

# TODO look for the loss function(Hamiltonian)


def test_f(x: np.ndarray) -> np.ndarray:
    # dummy function which can be passed as the parameter f. f gives the log probability
    # TODO replace this function with forward pass of the model in future
    return 2 * np.log(np.random.uniform(low=0, high=1.0, size=np.shape(x)[0]))


class Ferminet(torch.nn.Module):
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
        charge: int
            The total charge of the molecule system.
        seed: int, optional (default None)
            Random seed to use for electron initialization.
        batch_no: int, optional (default 10)
            Number of batches of the electron's positions to be initialized.

        """
        super(Ferminet, self).__init__()

        self.nucleon_coordinates = nucleon_coordinates
        self.seed = seed
        self.batch_no = batch_no
        self.spin = spin
        self.ion_charge = charge


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
               spin: int,
               ion_charge: int,
               seed: Optional[int] = None,
               batch_no: int = 10,
               pretrain=True):
    """
    Parameters:
    -----------
    nucleon_coordinates:  List[List]
      A list containing nucleon coordinates as the values with the keys as the element's symbol.
    spin: int
      The total spin of the molecule system.
    ion_charge:int
      The total charge of the molecule system.
    seed_no: int, optional (default None)
      Random seed to use for electron initialization.
    batch_no: int, optional (default 10)
      Number of batches of the electron's positions to be initialized.

    Attributes:
    -----------
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
    self.charge: np.ndarray = self.electron_no.reshape(
        np.shape(self.electron_no)[0])
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

    if self.spin >= 0:
        self.up_spin = (total_electrons + 2 * self.spin) // 2
        self.down_spin = total_electrons - self.up_spin
    else:
        self.down_spin = (total_electrons - 2 * self.spin) // 2
        self.up_spin = total_electrons - self.down_spin
    
    if self.up_spin - self.down_spin != self.spin:
        raise ValueError("Given spin is not feasible")

    nucl = torch.from_numpy(self.nucleon_pos)
    nucl.requires_grad = True
    charges_total = torch.from_numpy(self.charge.astype(float))
    charges_total.requires_grad = True
    inter = torch.from_numpy(self.inter_atom)
    inter.requires_grad = True
    self.model = Ferminet(nucleon_coordinates=nucl,
                          spin=self.up_spin,
                          charge=self.charge)

    self.molecule: ElectronSampler = ElectronSampler(
        batch_no=self.batch_no,
        central_value=self.nucleon_pos,
        seed=self.seed,
        f=lambda x: self.psi_log(x), # Will be implemented in successive PRs
        steps=1000,
        steps_per_update=20)  # sample the electrons using the electron sampler
    self.molecule.gauss_initialize_position(
        self.electron_no)  # initialize the position of the electrons
    adam = optimizers.Adam()
    super(FerminetModel, self).__init__(model=self.model,
                                        optimizer=adam,
                                        loss=torch.nn.CrossEntropyLoss)
