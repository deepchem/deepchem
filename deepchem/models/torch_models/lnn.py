import torch
import torch.nn as nn
from typing import Tuple
from deepchem.models.torch_models.layers import MultilayerPerceptron
from torch.func import jacfwd, hessian, vmap


class LNN(nn.Module):
    """Model for learning lagrangian dynamics using Lagrangian Neural
        Network.
    Lagrangian Neural Networks (LNNs) are a class of physics-informed
    models that learn the underlying Lagrangian function of a dynamical
    system directly from data. Instead of predicting derivatives directly,
    an LNN learns a scalar-valued Lagrangian function L(q, q_dot) and computes
    time evolution using Euler-Lagrange equations.

    Parameters
    ----------
    n_dof : int
        Number of degrees of freedom in the system. The input dimension
        will be 2*n_dof (positions + velocities).
    d_hidden : Tuple[int, ...], default (32, 32)
        Hidden layer dimensions for the multilayer perceptron that approximates
        the Lagrangian function.
    activation_fn : str, default 'softplus'
        Activation function to use in the hidden layers. Softplus is preferred
        for Lagrangian learning as it ensures smooth derivatives.

    Examples
    --------
    >>> import deepchem as dc
    >>> from deepchem.models.torch_models.lnn import LNN
    >>> import torch
    >>> lnn = LNN(n_dof=2, d_hidden=(64, 64), activation_fn='softplus')
    >>> z = torch.randn(10, 4, requires_grad=True)
    >>> # Get Lagrangian value
    >>> _ = lnn.eval()
    >>> L = lnn.lagrangian(z)  # Shape: (10,)
    >>> # Get accelerations for training
    >>> _ = lnn.train()
    >>> q_ddot = lnn(z)  # Shape: (10, 2)

    References
    ----------
    .. [1] Cranmer, M., Greydanus, S., Hoyer, S., Battaglia, P., Spergel, D., & Ho, S. (2020).
        "Lagrangian Neural Networks."
       International Conference on Learning Representations (ICLR).
       https://arxiv.org/abs/2003.04630
    """

    def __init__(
            self,
            n_dof: int,  # number of degrees of freedom
            d_hidden: Tuple[int, ...] = (32, 32),
            activation_fn: str = 'softplus') -> None:
        """Initialize the Lagrangian Neural Network.

        Parameters
        ----------
        n_dof : int
            Number of degrees of freedom in the system. The input dimension
            will be 2*n_dof (positions + velocities).
        d_hidden : Tuple[int, ...], default (32, 32)
            Hidden layer dimensions for the multilayer perceptron that approximates
            the Lagrangian function.
        activation_fn : str, default 'softplus'
            Activation function to use in the hidden layers. Softplus is preferred
            for Lagrangian learning as it ensures smooth derivatives.

        """

        super().__init__()
        self.n_dof = n_dof
        # Input is always 2*n_dof (positions + velocities)
        d_input = 2 * n_dof

        self.net = MultilayerPerceptron(d_input=d_input,
                                        d_hidden=d_hidden,
                                        d_output=1,
                                        activation_fn=activation_fn)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass through the LNN.
        The behavior depends on the training mode:
        - Training mode: Returns accelerations computed via Euler-Lagrange equations
        - Evaluation mode: Returns accelerations computed via Euler-Lagrange equations

        Parameters
        ----------
        z : torch.Tensor
            State space coordinates of shape (..., 2*n_dof) where the first
            n_dof dimensions are positions q and the last n_dof dimensions
            are velocities q_dot.

        Returns
        -------
        torch.Tensor
            Accelerations q̈ of shape (..., n_dof) computed by solving the
            Euler-Lagrange equations: d/dt(∂L/∂q̇) - ∂L/∂q = 0

        Examples
        --------
        >>> lnn = LNN(n_dof=2)
        >>> z = torch.randn(5, 4, requires_grad=True)  # Shape: (5, 4)
        >>> q_ddot = lnn(z) # Shape: (5, 2)
        """
        return self.calculate_dynamics(z)

    def calculate_dynamics(self, z: torch.Tensor) -> torch.Tensor:
        """Compute accelerations using Euler-Lagrange equations from learned Lagrangian.

        This method implements the core physics computation by:
        1. Computing first and second derivatives of the learned Lagrangian L(q, q_dot)
        2. Extracting the required partial derivatives for Euler-Lagrange equations
        3. Applying Euler Lagrange equation to calculate accelerations

        Parameters
        ----------
        x : torch.Tensor
            State space coordinates of shape (..., 2*n_dof) where the first
            n_dof dimensions are positions q and the last n_dof dimensions
            are velocities q_dot.

        Returns
        -------
        torch.Tensor
            Accelerations of shape (..., n_dof) computed by solving the
            Euler-Lagrange equations.

        Examples
        --------
        >>> lnn = LNN(n_dof=2)
        >>> x = torch.randn(10, 4, requires_grad=True)  # Shape: (10, 4)
        >>> accelerations = lnn.calculate_dynamics(x)  # Shape: (10, 2)
        """

        # separating current state into position and velocities
        n = z.shape[1] // 2
        z = z.requires_grad_(True)

        # matrix first order derivative
        jacobians = vmap(
            jacfwd(lambda xi: self.net(xi.unsqueeze(0)).squeeze()))(z)
        # matrix second order derivative
        hessians = vmap(
            hessian(lambda xi: self.net(xi.unsqueeze(0)).squeeze()))(z)

        q_dot = z[:, n:]  # velocity variable
        dL_dq = jacobians[:, :n]  # derivative w.r.t positions
        d2L_dqdot2 = hessians[:, n:, n:]  # derivative w.r.t velocities
        d2L_dqdot_dq = hessians[:, n:, :n]  # derivative w.r.t both variables

        # rhs of euler lagrangian equation to simplify it
        rhs = dL_dq - torch.bmm(d2L_dqdot_dq, q_dot.unsqueeze(-1)).squeeze(-1)
        # accelerations values
        q_ddot = torch.linalg.solve(d2L_dqdot2, rhs.unsqueeze(-1)).squeeze(-1)

        return q_ddot

    def lagrangian(self, z: torch.Tensor) -> torch.Tensor:
        """Compute the learned Lagrangian function L(q, q_dot) for given state.

        The Lagrangian is a scalar function that encodes the dynamics of the system.
        In classical mechanics, L = T - V (kinetic energy - potential energy).
        This method returns the neural network's approximation of this function.

        Parameters
        ----------
        z : torch.Tensor
            State space coordinates of shape (..., 2*n_dof) where the first
            n_dof dimensions are positions q and the last n_dof dimensions
            are velocities q_dot.

        Returns
        -------
        torch.Tensor
            Lagrangian values L(q, q_dot) of shape (...,) - one scalar value per
            input state configuration.

        Notes
        -----
        The Lagrangian function is the fundamental quantity from which all dynamics
        are derived via the Euler-Lagrange equations. The neural network learns to
        approximate this function such that the resulting dynamics match the training data.

        Examples
        --------
        >>> lnn = LNN(n_dof=2)
        >>> z = torch.randn(8, 4)  # Shape: (8, 4)
        >>> L_values = lnn.lagrangian(z)  # Shape: (8,)
        """
        L = self.net(z).squeeze(-1)
        return L
