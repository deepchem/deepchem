import torch
import torch.nn as nn


class DM21(nn.Module):
    """DM21 accurately models complex systems such as hydrogen chains,
    charged DNA base pairs and diradical transition states. It extends
    DFT (Density Functional Theory) which is a well established method
    for investigating electronic structure of many-body systems.

    Density Functional Theory (DFT) is a quantum mechanical method used
    to investigate the electronic structure (principally the ground state)
    of many-body systems, particularly atoms, molecules, and condensed
    phases. The main idea behind DFT is to describe the complex system
    of interacting particles (usually electrons) using the electron
    density, a function of spatial coordinates, rather than the many-body
    wavefunction.

    DeepMind 21 specifically address two long-standing problems with
    traditional functionals:

    1. The delocalization error: Most existing density functional
    approximations prefer electron densities that are unrealistically
    spread out over several atoms or molecules rather than being
    correctly localized around a single molecule or atom.

    2. Spin symmetry breaking: When describing the breaking of chemical
    bonds, existing functionals tend to unrealistically prefer configurations
    in which a fundamental symmetry known as spin symmetry is broken.

    These longstanding challenges are both related to how functionals
    behave when presented with a system that exhibits
    “fractional electron character.” By using a neural network to
    represent the functional and tailoring our training dataset to
    capture the fractional electron behaviour.

    Examples
    --------
    >>> import torch
    >>> from deepchem.models.dft.dm21 import DM21
    >>> model = DM21()
    >>> input = torch.rand((100, 11))
    >>> output = model(input)
    >>> output.shape
    torch.Size([100, 3])

    References
    ----------
    .. [1] Simulating matter on the quantum scale with AI. (2024, May 14). Google DeepMind.
        https://deepmind.google/discover/blog/simulating-matter-on-the-quantum-scale-with-ai/
    .. [2] James Kirkpatrick et al. ,Pushing the frontiers of density functionals by solving
        the fractional electron problem. Science374,1385-1389(2021).DOI:10.1126/science.abj6511
    .. [3] Density functional theory. (2024, April 24).
        In Wikipedia. https://en.wikipedia.org/wiki/Density_functional_theory

    """

    def __init__(self, hidden_size: int = 256, n_layers: int = 6):
        """Initialise the DeepMind 21 Model.

        Parameters
        ----------
        hidden_size: int (default 256)
            Size of Linear/Dense (Fully Connected) Layers to use in the model.
        n_layers: int (default 6)
            Number of Linear/Dense (Fully Connected) Layers to use in the model.

        """
        super(DM21, self).__init__()

        self.hidden_size: int = hidden_size
        self.n_layers: int = n_layers

        # Layer Initialisation
        self.lin_tanh = nn.Linear(11, self.hidden_size)
        self.lin_elu = nn.ModuleList()
        for i in range(self.n_layers):
            self.lin_elu.append(nn.Linear(self.hidden_size, self.hidden_size))
        self.final = nn.Linear(self.hidden_size, 3)
        self.acti_tanh = nn.Tanh()
        self.acti_elu = nn.ELU()
        self.acti_scaled_sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward Method for the DeepMind 21 Model.

        Parameters
        ----------
        x: torch.Tensor
            A torch tensor with 11 features.

        Returns
        -------
        torch.Tensor
            Predicted Output of the model.

        """
        x = torch.log(torch.abs(x) + torch.tensor([1e-4], device=x.device))
        x = self.acti_tanh(self.lin_tanh(x))
        for i in range(self.n_layers):
            x = self.acti_elu(self.lin_elu[i](x))
        x = self.acti_scaled_sigmoid(self.final(x))
        return x
