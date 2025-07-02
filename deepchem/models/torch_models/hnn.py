import torch
import torch.nn as nn
from torch.autograd import grad
from typing import Tuple
from deepchem.models.torch_models.layers import MultilayerPerceptron


class HNN(nn.Module):
    """Model for learning hamiltonian dynamics using Hamiltonian Neural
        Network.

    Hamiltonian Neural Networks (HNNs) are a class of physics-informed
    models that learn the underlying Hamiltonian function of a dynamical
    system directly from data. Instead of predicting derivatives directly,
    an HNN learns a scalar-valued Hamiltonian function H(q, p) and computes
    time evolution using Hamilton's equations.

    Parameters
    ----------
    d_input : int, default 2
        This are pairs of (q, p) values.
    d_hidden : Tuple[int, ...], default (32, 32)
        Hidden layer dimensions for the multilayer perceptron that approximates
        the Hamiltonian function.
    activation_fn : str, default 'tanh'
        Activation function to use in the hidden layers.

    Examples
    --------
    >>> import deepchem as dc
    >>> from deepchem.models.torch_models.hnn import HNN
    >>> import torch
    >>> hnn = HNN(d_input=2, d_hidden=(64, 64), activation_fn='tanh')
    >>> # Phase space coordinates [q1, q2, p1, p2]
    >>> z = torch.randn(10, 2, requires_grad=True)
    >>> # Get Hamiltonian value
    >>> _ = hnn.eval()
    >>> H = hnn(z)  # Shape: (10,)
    >>> # Get time derivatives for training
    >>> _ = hnn.train()
    >>> dz_dt = hnn(z)  # Shape: (10, 2)

    References
    ----------
    .. [1] Greydanus, S., Dzamba, M., & Yosinski, J. (2019).
        "Hamiltonian Neural Networks."
       Advances in Neural Information Processing Systems (NeurIPS) 32.
       https://arxiv.org/abs/1906.01563

    """

    def __init__(self,
                 d_input: int = 2,
                 d_hidden: Tuple[int, ...] = (32, 32),
                 activation_fn: str = 'tanh') -> None:
        """Initialize the Hamiltonian Neural Network.

        Parameters
        ----------
        d_input : int, default 2
            Dimensionality of the input phase space. Should be even.
        d_hidden : Tuple[int, ...], default (32, 32)
            Hidden layer dimensions for the MLP.
        activation_fn : str, default 'tanh'
            Activation function for hidden layers.
        """
        super().__init__()
        self.net = MultilayerPerceptron(d_input=d_input,
                                        d_hidden=d_hidden,
                                        d_output=1,
                                        activation_fn=activation_fn)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass through the HNN.

        The behavior depends on the training mode:
        - Training mode: Returns symplectic gradient (time derivatives)
        - Evaluation mode: Returns Hamiltonian value

        Parameters
        ----------
        z : torch.Tensor
            Phase space coordinates of shape (..., d_input) where the first
            d_input//2 dimensions are coordinates q and the last d_input//2
            dimensions are momentum p.

        Returns
        -------
        torch.Tensor
            If training: Time derivatives dz/dt of shape (..., d_input)
            If evaluation: Hamiltonian values H(z) of shape (...,)

        Examples
        --------
        >>> hnn = HNN(d_input=2)
        >>> z = torch.randn(5, 2, requires_grad=True)
        >>> _ = hnn.train()
        >>> dz_dt = hnn(z)  # Shape: (5, 4)
        >>> _ = hnn.eval()
        >>> H = hnn(z)  # Shape: (5,)
        """
        if self.training:
            return self.symplectic_gradient(z)
        else:
            H = self.net(z).squeeze(-1)
            return H

    def hamiltonian(self, z: torch.Tensor) -> torch.Tensor:
        """Compute the Hamiltonian function H(q, p).

        This method directly evaluates the learned Hamiltonian function without
        considering the training mode. It always returns the Hamiltonian value.

        Parameters
        ----------
        z : torch.Tensor
            Phase space coordinates of shape (..., d_input).

        Returns
        -------
        torch.Tensor
            Hamiltonian values H(z) of shape (...,).

        Examples
        --------
        >>> hnn = HNN(d_input=2)
        >>> z = torch.randn(3, 2)
        >>> H = hnn.hamiltonian(z)  # Shape: (3,)
        """
        H = self.net(z).squeeze(-1)
        return H

    def symplectic_gradient(self, z: torch.Tensor) -> torch.Tensor:
        """Compute the symplectic gradient using Hamilton's equations.

        This method computes the time derivatives of the phase space
        coordinates using Hamilton's equations:

        The gradients are computed using automatic differentiation to ensure
        exact computation of the partial derivatives.

        Parameters
        ----------
        z : torch.Tensor
            Phase space coordinates of shape (..., d_input) where the first
            d_input//2 dimensions are coordinates q and the last d_input//2
            dimensions are momenta p. Must have requires_grad=True or will be
            set automatically.

        Returns
        -------
        torch.Tensor
            Time derivatives dz/dt of shape (..., d_input) where the first
            d_input//2 dimensions are dq/dt and the last d_input//2 dimensions
            are dp/dt.

        Examples
        --------
        >>> hnn = HNN(d_input=2)
        >>> z = torch.randn(4, 2, requires_grad=True)
        >>> dz_dt = hnn.symplectic_gradient(z)  # Shape: (4, 2)
        >>> dq_dt = dz_dt[..., :1]
        >>> dq_dt.shape
        torch.Size([4, 1])
        >>> dp_dt = dz_dt[..., 1:]
        >>> dp_dt.shape
        torch.Size([4, 1])

        Notes
        -----
        The symplectic structure is preserved by construction through Hamilton
        equations. This ensures that the learned dynamics conserve energy and
        maintain the geometric properties of Hamiltonian systems.
        """
        # Ensure z requires gradients
        if not z.requires_grad:
            z = z.requires_grad_(True)

        H = self.net(z).squeeze(-1)

        grad_H = grad(H.sum(), z, create_graph=True)[0]

        dim = z.shape[-1] // 2
        dH_dq = grad_H[..., :dim]
        dH_dp = grad_H[..., dim:]

        dq_dt = dH_dp
        dp_dt = -dH_dq

        return torch.cat([dq_dt, dp_dt], dim=-1)
