import torch
import torch.nn as nn


class SymbolicNet(nn.Module):
    """
    Linear + quadratic symbolic regression model.

    y = w1*x + w2*x^2 + b
    """

    def __init__(self, n_features: int):
        super().__init__()

        self.n_features = n_features

        # Linear weights
        self.linear = nn.Linear(n_features, 1, bias=False)

        # Quadratic weights
        self.quadratic = nn.Linear(n_features, 1, bias=False)

        # Bias
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        linear_part = self.linear(x)
        quad_part = self.quadratic(x ** 2)
        return linear_part + quad_part + self.bias

    def get_equation(self):
        return {
            "linear": self.linear.weight.detach().cpu().numpy().flatten(),
            "quadratic": self.quadratic.weight.detach().cpu().numpy().flatten(),
            "bias": self.bias.item()
        }

