"""
DeepChem Integration — DifferentiableFEMModel.

Wraps the inverse problem as a ``dc.models.TorchModel`` subclass.
Learnable parameters: per-element log-conductivity (ensures k > 0).
Forward pass: assemble + solve.  Loss: MSE vs observed temperatures.
Use ``model.fit(dataset)`` and ``model.predict(dataset)`` via standard DeepChem.
"""

import torch
import torch.nn as nn
import deepchem as dc

from .assembler import DifferentiableAssembler
from .solver import DifferentiableFEMSolver


# class that inherits nn.Module
class FEMForwardModule(nn.Module):
    def __init__(self, mesh, ref_element, bc_dict):
        super().__init__()
        # stoing mesh, reference element and boundary conditions as regular attributes (they are not learnable):
        self.mesh = mesh
        self.ref = ref_element
        self.bc_dict = bc_dict

        # learnable parameter: a 1D tensor of lenght n_elements initialized to all zeroes
        self.log_k = nn.Parameter(
            torch.zeros(mesh.n_elements)
        )  # I dont optimize k directly because the optimizer could push for negative values

    def forward(self, x):
        device = self.log_k.device  # wherever the parameter lives
        self.mesh.to(device)
        self.ref.to(device)
        self.ref.quad_points = self.ref.quad_points.to(device)
        self.ref.quad_weights = self.ref.quad_weights.to(device)

        # x is dummy input from DataLoader (node coords); mesh is stored
        k = torch.exp(
            self.log_k
        )  # transform the learnable loarithimic conductivity into actual conductivity
        asm = DifferentiableAssembler(self.mesh, self.ref)
        slv = DifferentiableFEMSolver(self.mesh, asm)
        return slv.solve(self.bc_dict, k=k)


# class that ineirhts dc.models.TorchModel
class DifferentiableFEMModel(dc.models.TorchModel):
    def __init__(self, mesh, ref_element, bc_dict, **kwargs):
        module = FEMForwardModule(mesh, ref_element, bc_dict)
        super().__init__(model=module, loss=dc.models.losses.L2Loss(), **kwargs)
