"""
Differentiable Finite Element Method in DeepChem.

An end-to-end differentiable 2D FEM solver in PyTorch, integrated with
DeepChem's TorchModel and NumpyDataset abstractions.
"""

from .mesh import FEMMesh, mesh_to_dataset
from .mesh_generator import generate_unit_square_mesh
from .reference_element import P1Triangle
from .assembler import DifferentiableAssembler
from .solver import DifferentiableFEMSolver
from .model import FEMForwardModule, DifferentiableFEMModel

__all__ = [
    "FEMMesh",
    "mesh_to_dataset",
    "generate_unit_square_mesh",
    "P1Triangle",
    "DifferentiableAssembler",
    "DifferentiableFEMSolver",
    "FEMForwardModule",
    "DifferentiableFEMModel",
]
