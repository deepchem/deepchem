
import torch
import torch.nn as nn
from torch.autograd import grad
from typing import Tuple
from deepchem.models.torch_models.layers import MultilayerPerceptron
from deepchem.models.torch_models.torch_model import TorchModel
from deepchem.models.losses import L2Loss


class HNN(nn.Module):
    """Hamiltonian Neural Network (HNN) implementation.
    
    References
    ----------
    Greydanus, Samuel, Misko Dzamba, and Jason Yosinski. "Hamiltonian neural
    networks." Advances in neural information processing systems 32 (2019).
    
    Examples
    --------
    >>> import torch
    >>> from deepchem.models.torch_models import HNN
    >>> 
    >>> # Create HNN for 2D harmonic oscillator (q, p)
    >>> hnn = HNN(d_input=2, d_hidden=(32, 32), activation_fn='tanh')
    >>> 
    >>> # Generate some phase space coordinates
    >>> z = torch.randn(10, 2, requires_grad=True)
    >>> 
    >>> # Compute Hamiltonian
    >>> H = hnn.hamiltonian(z)
    >>> print(H.shape)  # torch.Size([10])
    >>> 
    >>> # Compute symplectic gradient (time derivatives)
    >>> dz_dt = hnn.symplectic_gradient(z)
    >>> print(dz_dt.shape)  # torch.Size([10, 2])
    """

    def __init__(self,
                 d_input: int = 2,
                 d_hidden: Tuple[int, ...] = (32, 32),
                 activation_fn: str = 'tanh') -> None:

        super().__init__()
        self.net = MultilayerPerceptron(d_input=d_input,
                                        d_hidden=d_hidden,
                                        d_output=1,
                                        activation_fn=activation_fn)

    def forward(self, z: torch.Tensor) -> torch.Tensor:

        """Forward pass of the HNN.
        
        The behavior depends on the training mode:
        - Training mode: Returns symplectic gradient (time derivatives)
        - Evaluation mode: Returns Hamiltonian values
        
        Parameters
        ----------
        z : torch.Tensor
            Input phase space coordinates of shape (batch_size, d_input).
            For Hamiltonian systems, this should be [q1, q2, ..., p1, p2, ...]
            where q are positions and p are momenta.
            
        Returns
        -------
        torch.Tensor
            If training: symplectic gradient of shape (batch_size, d_input)
            If evaluation: Hamiltonian values of shape (batch_size,)
        """
        
        if self.training:
            return self.symplectic_gradient(z)
        else:
            H = self.net(z).squeeze(-1)
            return H

    def hamiltonian(self, z: torch.Tensor) -> torch.Tensor:
        """Compute the Hamiltonian function H(q, p).
        
        Parameters
        ----------
        z : torch.Tensor
            Phase space coordinates of shape (batch_size, d_input).
            
        Returns
        -------
        torch.Tensor
            Hamiltonian values of shape (batch_size,). These represent the
            total energy of the system at each phase space point.
        """
        
        
        H = self.net(z).squeeze(-1)
        return H

    def symplectic_gradient(self, z: torch.Tensor) -> torch.Tensor:

        """Compute the symplectic gradient following Hamilton's equations.
        
        This method computes the time derivatives of the phase space coordinates
        according to Hamilton's equations:
        dq/dt = ∂H/∂p (derivative of Hamiltonian w.r.t. momentum)
        dp/dt = -∂H/∂q (negative derivative of Hamiltonian w.r.t. position)
        
        Parameters
        ----------
        z : torch.Tensor
            Phase space coordinates of shape (batch_size, d_input).
            The input is assumed to be structured as [q1, q2, ..., qn, p1, p2, ..., pn]
            where the first half are positions and the second half are momenta.
            
        Returns
        -------
        torch.Tensor
            Time derivatives dz/dt of shape (batch_size, d_input).
            The output follows the same structure as input: [dq1/dt, dq2/dt, ..., dp1/dt, dp2/dt, ...]
            
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


class HNNModel(TorchModel):

    """DeepChem TorchModel wrapper for Hamiltonian Neural Networks.
    
    
    Examples
    --------
    >>> import numpy as np
    >>> from deepchem.models.torch_models import HNNModel
    >>> 
    >>> # Create model for 4D phase space (2 position + 2 momentum coordinates)
    >>> model = HNNModel(d_input=4, d_hidden=(64, 64), activation_fn='tanh')
    >>> 
    >>> # Generate some sample data
    >>> X = np.random.randn(100, 4)
    >>> 
    >>> # Predict time derivatives (dynamics)
    >>> dX_dt = model.predict_on_batch(X)
    >>> print(dX_dt.shape)  # (100, 4)
    >>> 
    >>> # Predict Hamiltonian (energy)
    >>> H = model.predict_hamiltonian(X)
    >>> print(H.shape)  # (100,)
    """

    def __init__(self,
                 d_input: int = 2,
                 d_hidden: Tuple[int, ...] = (32, 32),
                 activation_fn: str = 'tanh',
                 **kwargs) -> None:

        model = HNN(d_input=d_input,
                    d_hidden=d_hidden,
                    activation_fn=activation_fn)
        super().__init__(model, loss=L2Loss(), **kwargs)


    def predict_on_batch(self, X):
        """Initialize the HNNModel.
        
        Parameters
        ----------
        d_input : int, default 2
            Dimensionality of the input phase space. Should be even for proper
            symplectic structure (d_input = 2 * degrees_of_freedom).
        d_hidden : Tuple[int, ...], default (32, 32)
            Hidden layer dimensions for the neural network.
        activation_fn : str, default 'tanh'
            Activation function for the neural network.
        **kwargs
            Additional keyword arguments passed to the parent TorchModel class.
            Common options include learning_rate, batch_size, etc.
        """
      
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device, requires_grad=True)
            predictions = self.model.symplectic_gradient(X_tensor)
            return predictions.cpu().numpy()

    def predict_hamiltonian(self, X):
        """Predict time derivatives (dynamics) for a batch of phase space coordinates.
        
        Parameters
        ----------
        X : array-like
            Input phase space coordinates of shape (batch_size, d_input).
            Should be structured as [q1, q2, ..., qn, p1, p2, ..., pn] where
            q are positions and p are momenta.
            
        Returns
        -------
        numpy.ndarray
            Time derivatives dX/dt of shape (batch_size, d_input).
            These represent the velocities and forces in the system.
            
        Examples
        --------
        >>> model = HNNModel(d_input=4)
        >>> X = np.array([[1.0, 0.0, 0.0, 1.0]])  # Simple harmonic oscillator state
        >>> dX_dt = model.predict_on_batch(X)
        >>> print(dX_dt.shape)  # (1, 4)
        """
      
        self.model.eval()
        # with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        H = self.model.hamiltonian(X_tensor)
        return H.cpu().numpy()

    def symplectic_gradient(self, z: torch.Tensor) -> torch.Tensor:
        """Predict Hamiltonian (total energy) for a batch of phase space coordinates.
        
        The Hamiltonian represents the total energy of the system and should be
        conserved over time for isolated systems. This method is useful for
        monitoring energy conservation during system evolution.
        
        Parameters
        ----------
        X : array-like
            Input phase space coordinates of shape (batch_size, d_input).
            
        Returns
        -------
        numpy.ndarray
            Hamiltonian values of shape (batch_size,).
            These represent the total energy at each phase space point.
            
        Examples
        --------
        >>> model = HNNModel(d_input=2)
        >>> X = np.array([[1.0, 0.0], [0.0, 1.0]])
        >>> H = model.predict_hamiltonian(X)
        >>> print(H.shape)  # (2,)
        """
      
        return self.model.symplectic_gradient(z)