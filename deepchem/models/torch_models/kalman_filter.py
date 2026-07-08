import torch
import torch.nn as nn
import numpy as np
from deepchem.models.torch_models.torch_model import TorchModel
from typing import Union, List, Optional, Callable
from deepchem.utils.typing import OneOrMany

class KalmanModule(nn.Module):
    """
    PyTorch module for a Linear Kalman Filter.
    
    The system is defined by:
    x_t = F * x_{t-1} + w_t,  w_t ~ N(0, Q)
    z_t = H * x_t + v_t,      v_t ~ N(0, R)
    """
    def __init__(self, state_dim: int, observation_dim: int, device: torch.device):
        super(KalmanModule, self).__init__()
        self.state_dim = state_dim
        self.observation_dim = observation_dim
        self.device = device

        # Learnable parameters
        self.F = nn.Parameter(torch.eye(state_dim, device=device)) # State transition
        self.H = nn.Parameter(torch.randn(observation_dim, state_dim, device=device)) # Observation matrix
        
        # Covariances (parameterized as Cholesky factors or log-diagonal for stability)
        # Here i use simple diagonal log-variance for simplicity in this initial version
        self.Q_log_diag = nn.Parameter(torch.zeros(state_dim, device=device)) 
        self.R_log_diag = nn.Parameter(torch.zeros(observation_dim, device=device))

        # Initial state
        self.x0 = nn.Parameter(torch.zeros(state_dim, device=device))
        self.P0_log_diag = nn.Parameter(torch.zeros(state_dim, device=device))

    def forward(self, observations):
        """
        Parameters
        ----------
        observations: torch.Tensor
            Shape (batch_size, seq_len, observation_dim)
        
        Returns
        -------
        filtered_states: torch.Tensor
            Shape (batch_size, seq_len, state_dim)
        nll_loss: torch.Tensor
            Scalar Negative Log Likelihood
        """
        batch_size, seq_len, _ = observations.shape
        
        # Initialize state
        x_curr = self.x0.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1) # (B, S, 1)
        P_curr = torch.diag_embed(torch.exp(self.P0_log_diag)).unsqueeze(0).expand(batch_size, -1, -1) # (B, S, S)
        
        Q = torch.diag_embed(torch.exp(self.Q_log_diag))
        R = torch.diag_embed(torch.exp(self.R_log_diag))
        
        filtered_states = []
        log_likelihood = 0.0
        
        for t in range(seq_len):
            z_t = observations[:, t, :].unsqueeze(-1) # (B, O, 1)
            
            # --- Predict ---
            x_pred = torch.matmul(self.F, x_curr)
            P_pred = torch.matmul(torch.matmul(self.F, P_curr), self.F.t()) + Q
            
            # --- Update ---
            # Residual
            y = z_t - torch.matmul(self.H, x_pred) # (B, O, 1)
            
            # Residual covariance S = H P H^T + R
            S = torch.matmul(torch.matmul(self.H, P_pred), self.H.t()) + R # (B, O, O)
            
            # Inverse of S (add jitter for stability)
            S_inv = torch.inverse(S + 1e-6 * torch.eye(self.observation_dim, device=self.device))
            
            # Kalman Gain K = P H^T S^-1
            K = torch.matmul(torch.matmul(P_pred, self.H.t()), S_inv) # (B, S, O)
            
            # Update state
            x_curr = x_pred + torch.matmul(K, y)
            
            # Update covariance P = (I - KH) P_pred
            I = torch.eye(self.state_dim, device=self.device)
            P_curr = torch.matmul((I - torch.matmul(K, self.H)), P_pred)
            
            filtered_states.append(x_curr.squeeze(-1))
            
            # Compute Likelihood using S and y
            # L = -0.5 * (y^T S^-1 y + log|S| + const)
            # minimize NLL = -L
            # NLL = 0.5 * (y^T S^-1 y + log|S|)
            
            y_flat = y.squeeze(-1) # (B, O)
            term1 = torch.bmm(y.transpose(1, 2), torch.matmul(S_inv, y)).squeeze() # (B,)
            term2 = torch.logdet(S) # (B,)
            
            log_likelihood = log_likelihood - 0.5 * (term1 + term2).mean()

        filtered_states = torch.stack(filtered_states, dim=1) # (B, T, S)
        
        return filtered_states, -log_likelihood

class KalmanFilter(TorchModel):
    """
    A DeepChem model that learns a linear Kalman Filter from data.
    
    The model takes sequences of observations and trains the system parameters
    (transition matrix, observation matrix, noise covariances) to maximize
    the likelihood of the observations.
    """
    def __init__(self,
                 state_dim: int,
                 observation_dim: int,
                 learning_rate: float = 0.01,
                 **kwargs):
        """
        Parameters
        ----------
        state_dim: int
            Dimension of the hidden state vector.
        observation_dim: int
            Dimension of the observation vector.
        learning_rate: float
            Learning rate for the optimizer.
        """
        self.state_dim = state_dim
        self.observation_dim = observation_dim
        
        # Determine device
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
            
        module = KalmanModule(state_dim, observation_dim, device)
        
        # Define loss function to pull the only loss output
        def nll_loss(outputs, labels, weights):
            return outputs[0]
        
        super(KalmanFilter, self).__init__(
            module,
            loss=nll_loss,
            output_types=['prediction', 'loss'],
            learning_rate=learning_rate,
            device=device,
            **kwargs
        )
