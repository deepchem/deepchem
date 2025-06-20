import torch
import torch.nn as nn
from torch.autograd import grad
from typing import Tuple
from deepchem.models.torch_models.layers import MultilayerPerceptron
from deepchem.models.torch_models.torch_model import TorchModel
from deepchem.models.losses import L2Loss


class HNN(nn.Module):
    """
    Hamiltonian Neural Network (HNN) base class.
    Examples
    --------
    >>> import torch
    >>> hnn = HNN(d_input=2, d_hidden=(32, 32), activation_fn='tanh')
    >>> z = torch.randn(10, 2)  # batch of 10 phase space states
    >>> H = hnn(z)  # Hamiltonian values

    References
    ----------
    .. [1] Greydanus, Samuel, Misko Dzamba, and Jason Yosinski. 
           "Hamiltonian neural networks." Advances in neural information processing systems 32 (2019).
    """

    def __init__(self,
                 d_input: int = 2,
                 d_hidden: Tuple[int, ...] = (32, 32),
                 activation_fn: str = 'tanh') -> None:
        """
        Initialize the Hamiltonian Neural Network.
        
        Parameters
        ----------
        d_input : int, default 2
            Input dimension [q, p]
        d_hidden : Tuple[int, ...], default (32, 32)
            Hidden layer dimensions for the multilayer perceptron
        activation_fn : str, default 'tanh'
            Activation function for the MLP. Options include 'tanh', 'relu', 'sigmoid', etc.
        """
        super().__init__()
        self.net = MultilayerPerceptron(d_input=d_input,
                                        d_hidden=d_hidden,
                                        d_output=1,
                                        activation_fn=activation_fn)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute the Hamiltonian.
        
        Parameters
        ----------
        z : torch.Tensor
            Phase space coordinates of shape (batch_size, d_input)
            where z = [q, p] with q being positions and p being momentum
            
        Returns
        -------
        torch.Tensor
            Hamiltonian values of shape (batch_size,)
        
        Note
        -------
        The paper suggest to return the scalar output
        """
        H = self.net(z).squeeze(-1)
        return H


class HNNModel(TorchModel):
    """
    DeepChem wrapper for Hamiltonian Neural Network.
        
    Examples
    --------
    >>> import torch
    >>> model = HNNModel(d_input=2, d_hidden=(64, 64))
    >>> z = torch.randn(100, 2)  
    >>> dz_dt = model.symplectic_gradient(z)
    
    Notes
    -----
    This model uses deepchem's L2 loss
    """

    def __init__(self,
                 d_input: int = 2,
                 d_hidden: Tuple[int, ...] = (32, 32),
                 activation_fn: str = 'tanh',
                 **kwargs) -> None:
        """
        Initialize HNN model wrapper.
        
        Parameters
        ----------
        d_input : int, default 2
            Input dimension [q, p]
        d_hidden : Tuple[int, ...], default (32, 32)
            Hidden layer dimensions for the multilayer perceptron
        activation_fn : str, default 'tanh'
            Activation function for the MLP
        **kwargs
            Additional arguments passed to TorchModel parent class
        """
        model = HNN(d_input=d_input,
                    d_hidden=d_hidden,
                    activation_fn=activation_fn)
        super().__init__(model, loss=L2Loss(), **kwargs)

    def symplectic_gradient(self,
                            z: torch.Tensor,
                            training: bool = True) -> torch.Tensor:
        """
        Compute symplectic gradients following Hamilton's equations.
        
        
        Parameters
        ----------
        z : torch.Tensor
            Phase space coordinates of shape (batch_size, d_input)
            where z = [q, p] with q being positions and p being momentum.
        training : bool, default True
            Whether to create computation graph for gradient computation.
            Set to False during inference for efficiency.
            
        Returns
        -------
        torch.Tensor
            Time derivatives [dq/dt, dp/dt] of shape (batch_size, d_input)
            
        """
        z = z.clone().detach().requires_grad_(True)
        H = self.model(z)

        grad_H = grad(H.sum(), z, create_graph=training)[0]

        dim = z.shape[-1] // 2
        dH_dq = grad_H[..., :dim]
        dH_dp = grad_H[..., dim:]

        dq_dt = dH_dp
        dp_dt = -dH_dq

        return torch.cat([dq_dt, dp_dt], dim=-1)
