import torch
import torch.nn as nn
import numpy as np
from torch.autograd import grad
from typing import Tuple
from deepchem.models.torch_models.layers import MultilayerPerceptron
from deepchem.models.torch_models.torch_model import TorchModel
from deepchem.models.losses import L2Loss
from deepchem.utils.differentiation_utils import leapfrog


class SRNN(nn.Module):
    """
    Parameters
    ----------
    d_input: int, default 2
        input state variables (q, p)
    d_hidden: Tuple[int, ...], default (128, 128)
        hidden layer dimensions for the MultilayerPerceptron that returns scalar hamiltonian value
    activation_fn: str, default 'tanh'
        activation function to use in layers except output layer
    dt: float, default 0.1
        delta t value
    T: int, default 10
        total timesteps value

    Examples
    --------
    >>> import torch
    >>> import deepchem
    >>> from deepchem.models.torch_models.srnn import SRNN
    >>> srnn = SRNN()
    >>> # co-ordinates from observed phase space trajectory (q, p)
    >>> z = torch.randn(10, 2, requires_grad=True)
    >>> results = srnn(z) # shape: (10, 10, 2)

    References
    ----------
    .. [1] Chen, Zhengdao and Zhang, Jianyu and Arjovsky, Martin and Bottou, Leon (2019)/
        "Symplectic recurrent neural networks"
        arXiv preprint arXiv:1909.13334
        https://arxiv.org/pdf/1909.13334
    """
    def __init__(self, d_input:int = 2,
                 d_hidden:Tuple[int, ...] = (128, 128),
                 activation_fn: str = 'tanh',
                 dt:float = 0.1,
                 T: int = 10) -> None:
        """Initialize the Symplectic Recurrent Neural network.

        Parameters
        ----------
        d_input: int, default 2
        input state variables (q, p)
        d_hidden: Tuple[int, ...], default (128, 128)
            hidden layer dimensions for the MultilayerPerceptron that returns scalar hamiltonian value
        activation_fn: str, default 'tanh'
            activation function to use in layers except output layer
        dt: float, default 0.1
            delta t value
        T: int, default 10
            total timesteps value
        """
        super().__init__()
        self.h_net = MultilayerPerceptron(d_input=d_input,
                                          d_hidden=d_hidden,
                                          d_output=1,
                                          activation_fn=activation_fn)
        
        self.dt = dt 
        self.T = T 
    
    def get_hamiltonian(self, q0: torch.Tensor, p0: torch.Tensor) -> torch.Tensor:
        """helper function to get the scalar hamiltonian value through forward pass of MLP

        Parameters
        ----------
        q0: torch.Tensor
            q0/position value in shape (1, 1)
        p0: torch.Tensor
            p0/momenta value in shape (1, 1)

        Returns
        -------
        torch.Tensor
            scalar hamiltonian values in shape of (1, 1)
        
        Examples
        --------
        >>> model = SRNN()
        >>> q0 = torch.Tensor([1.0]).unsqueuze(1)
        >>> p0 = torch.Tensor([2.0]).unsqueuze(1)
        >>> scalar_h = model.get_hamiltonian(q0, p0) # shape(1, 1)
        """
        x = torch.cat([q0, p0], dim=1)
        H = self.h_net(x)
        return H
    
    def forward(self, z: torch.Tensor):
        """forward method which returns complete predicted trajectory using leapfrog integrator
        
        Parameters
        ---------
        z: torch.Tensor
            initial input value of shape (batch, 2)

        Returns
        -------
        torch.Tensor
            predicted trajectory of shape (batch, T, 2)
        
        Examples
        --------
        >>> model = SRNN()
        >>> initial_states = torch.randn(5, 2, requires_grad=True)
        >>> pred_traj = srnn(initial_states) # shape (5, 10, 2)
        """
        q0 = z[:, 0:1]
        p0 = z[:, 1:2]
        predicted_traj = leapfrog(q0, p0, self.get_hamiltonian, self.dt, self.T, is_hamiltonian=True)
        predicted_traj = predicted_traj.permute(1, 0, 2)
        return predicted_traj


class SRNNModel(TorchModel):
    """SRNN wrapper model which inherits TorchModel.

    Parameters
    ----------
    d_input: int, default 2
        input state variables (q, p)
    d_hidden: Tuple[int, ...], default (128, 128)
        hidden layer dimensions for the MultilayerPerceptron that returns scalar hamiltonian value
    activation_fn: str, default 'tanh'
        activation function to use in layers except output layer
    dt: float, default 0.1
        delta t value
    T: int, default 10
        total timesteps value

    References
    ----------
    .. [1] Chen, Zhengdao and Zhang, Jianyu and Arjovsky, Martin and Bottou, Leon (2019)/
        "Symplectic recurrent neural networks"
        arXiv preprint arXiv:1909.13334
        https://arxiv.org/pdf/1909.13334
    """



    def __init__(self, d_input:int =2,
                 d_hidden:Tuple[int, ...] = (128, 128),
                 activation_fn: str = 'tanh',
                 dt:float = 0.1,
                 T: int = 10,
                 **kwargs ) -> None:
        model = SRNN(d_input=d_input,
                     d_hidden= d_hidden,
                     activation_fn=activation_fn,
                     dt=dt,
                     T=T)
        super().__init__(model, loss=L2Loss(), **kwargs)