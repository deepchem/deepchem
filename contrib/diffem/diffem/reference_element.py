"""
Reference Element — P1 Triangle.

Linear shape functions on the reference triangle (0,0), (1,0), (0,1):
    N0 = 1 - xi - eta,  N1 = xi,  N2 = eta

Gradients are constant, so 1-point centroid quadrature is exact for the
Poisson stiffness integral.
"""

import torch


# in FEM we need to calculate the field in the element, not in the nnodes.
# for doing this we need the shape functions and the quadrature.
class P1Triangle:
    def __init__(self):
        # since the shape functions are linear make no sense to consider
        # second order derivatives, I just need one point: I am
        # keeping everything easy because this is just an example.
        # (1/3, 1/3) is the geometric centroid of the triangle and
        # the weight is 0.5 since I am integrating on a unit triangle and
        # its weight can be thinked as its area, so 0.5
        self.quad_points = torch.tensor([[1 / 3, 1 / 3]], dtype=torch.float32)
        self.quad_weights = torch.tensor([0.5], dtype=torch.float32)

    # (eps, eta) is just a local coordinate system that help me to write the
    # coordinates of the nodes of the triangle as (0,0), (1,0), (0,1). Also, s
    # since xi has shape (N_points, 2) I have eps = xi[:, 0] and eta = xi[:, 1]
    def basis(self, xi):
        # stacking the shape functions for the triangle in a tensor
        return torch.stack([1.0 - xi[:, 0] - xi[:, 1], xi[:, 0], xi[:, 1]], dim=1)

    def grad_basis(self, xi):
        # costant gradients for the 3 nodes are stored:
        g = torch.tensor(
            [
                [-1.0, -1.0],  # Node 0. N0=1-psi-eta, dN0/dxi=-1, dN0/deta=-1, N1=N2=0 so are worthless
                [1.0, 0.0],    # Node 1. N1=psi, dN1/dxi=1, dN1/deta=0 , N0=N2=0 so are worthless
                [0.0, 1.0],    # Node 2. N2=eta, dN2/dxi=0, dN2/deta=1, N0=N1=0 so are worthelss
            ],
            dtype=torch.float32,
            device=xi.device,
        )
        return g.unsqueeze(0).expand(xi.shape[0], -1, -1)

    def to(self, device):
        for attr_name in vars(self):
            val = getattr(self, attr_name)
            if isinstance(val, torch.Tensor):
                setattr(self, attr_name, val.to(device))
        return self
