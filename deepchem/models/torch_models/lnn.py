"""
LNN (Lagrangian Neural Network) Model for DeepChem.

This file defines:
1. A PyTorch network (LagrangianNetwork) that learns an approximate L(q, qdot).
2. A custom forward() that returns predicted accelerations (ddq).
3. A TorchModel wrapper (LagrangianNNModel) integrating with DeepChem's training API.

References:
- Cranmer et al. (2020): "Lagrangian Neural Networks" (arXiv:2003.04630).
- DeepChem design patterns in similar PyTorch files (like, gnn.py, pinns_model.py).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad

# DeepChem imports
from deepchem.models import TorchModel

__all__ = [
    "LagrangianNetwork",
    "lagrangian_loss_fn",
    "LagrangianNNModel",
]


class LagrangianNetwork(nn.Module):
    """
    PyTorch module to approximate the Lagrangian L(q, qdot), then produce predicted
    accelerations ddq via Euler–Lagrange logic in the forward pass.

    Parameters
    ----------
    input_dim : int
        Dimension of the input vector [q, qdot]. For a single pendulum, input_dim=2.
    hidden_dim : int
        Number of hidden units in each of the two feedforward layers.

    Notes
    -----
    This network sets `requires_grad=True` on the input tensor to compute partial
    derivatives w.r.t. q, qdot. The final output is ddq (predicted accelerations).

    Example
    -------
    >>> net = LagrangianNetwork(input_dim=2, hidden_dim=64)
    >>> x = torch.tensor([[0.5, -0.2]], dtype=torch.float32)
    >>> ddq = net(x)
    >>> ddq.shape
    torch.Size([1, 1])
    """

    def __init__(self, input_dim=2, hidden_dim=64):
        super(LagrangianNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)  # scalar L

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute L(q, qdot) and derive ddq via partial derivatives.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, input_dim), for example, [q, qdot].

        Returns
        -------
        ddq : torch.Tensor
            Predicted accelerations, shape (batch_size, 1).
        """
        # 1) Make a clone that requires grad for partial derivatives
        x = x.clone().detach()
        x.requires_grad_(True)

        # 2) Separate q, qdot (for 1D dof: input_dim=2)
        q = x[:, :1]      # shape (batch_size, 1)
        qdot = x[:, 1:]   # shape (batch_size, 1)

        # 3) MLP to get L(q, qdot)
        combined = torch.cat([q, qdot], dim=1)  # shape (batch_size, 2)
        h = torch.relu(self.fc1(combined))
        h = torch.relu(self.fc2(h))
        L_approx = self.out(h).squeeze(-1)      # shape (batch_size,)

        # 4) Summation for computing partial derivatives
        L_sum = L_approx.sum()

        # 5) partial derivatives wrt q, qdot
        dLdqdot = grad(L_sum, qdot, create_graph=True, retain_graph=True)[0]
        dLdq    = grad(L_sum, q,    create_graph=True, retain_graph=True)[0]

        # 6) Example "ddq = (d/dt)(dL/dqdot) - dL/dq"; here simplified to ddq = - dLdq
        #    another possibility would be do a second derivative pass for (d/dt)(dLdqdot).
        ddq = -dLdq  # shape (batch_size, 1)

        return ddq


def lagrangian_loss_fn(outputs, labels, weights):
    """
    Minimal 3-argument loss function for LagrangianNNModel. We unroll any lists
    so that outputs, labels are direct tensors, then do MSE.

    Parameters
    ----------
    outputs : list or torch.Tensor
        Typically a list of length 1 containing the ddq predictions from forward().
    labels : list or torch.Tensor
        Typically a list of length 1 containing the ground truth accelerations.
    weights : list or None
        Not used here, but must be present for TorchModel's signature.

    Returns
    -------
    loss : torch.Tensor
        Mean squared error between predicted ddq and true ddq.
    """
    if isinstance(outputs, list):
        outputs = outputs[0]
    if isinstance(labels, list):
        labels = labels[0]

    return F.mse_loss(outputs, labels)


class LagrangianNNModel(TorchModel):
    """
    DeepChem model wrapper around LagrangianNetwork. Inherits from TorchModel,
    hooking in the custom Euler–Lagrange-based forward pass and an MSE loss.

    Parameters
    ----------
    input_dim : int, optional (default 2)
        Dimension of the input [q, qdot].
    hidden_dim : int, optional (default 64)
        Hidden size of the feedforward layers.
    model_dir : str, optional
        If provided, directory path for model saving/loading.
    **kwargs
        Additional TorchModel arguments (learning_rate, optimizer, etc.).

    Examples
    --------
    >>> import numpy as np
    >>> from deepchem.data import NumpyDataset
    >>> from deepchem.metrics import Metric
    >>> from sklearn.metrics import mean_squared_error
    >>> # Fake data: 2D input => [theta, omega], 1D label => ddtheta
    >>> X = np.random.rand(10, 2).astype(np.float32)
    >>> y = np.random.rand(10, 1).astype(np.float32)
    >>> ds = NumpyDataset(X, y)
    >>> model = LagrangianNNModel(input_dim=2, hidden_dim=16, learning_rate=1e-3)
    >>> model.fit(ds, nb_epoch=5)
    >>> scores = model.evaluate(ds, [Metric(mean_squared_error)])
    >>> print("MSE:", scores["mean_squared_error"])
    """

    def __init__(self, input_dim=2, hidden_dim=64, model_dir=None, **kwargs):
        net = LagrangianNetwork(input_dim=input_dim, hidden_dim=hidden_dim)
        super(LagrangianNNModel, self).__init__(
            model=net,
            loss=lagrangian_loss_fn,  # 3-arg signature
            model_dir=model_dir,
            **kwargs
        )
