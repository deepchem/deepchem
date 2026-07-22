"""
Mesh data structure and DeepChem dataset helper.

``FEMMesh`` stores node coordinates and element connectivity as PyTorch tensors.
- **nodes** is float32 and can carry requires_grad for shape optimisation.
- **elements** is int64 (indexing only, never differentiated).
- **boundary_nodes** maps tag names to index lists for declarative BC specification.
"""

import torch
import numpy as np
import deepchem as dc


class FEMMesh:
    def __init__(self, nodes, elements, boundary_nodes=None):
        # put the parameters as standard PyTorch tensors:
        self.nodes = torch.as_tensor(nodes, dtype=torch.float32)  # discretization points
        self.elements = torch.as_tensor(elements, dtype=torch.int64)  # in FEM we create elements...
        # ...(square, crosses, triangles, trapezoidal, ecc...) from the discretization points
        self.boundary_nodes = boundary_nodes or {}
        # since nodes and elements have shape (n_nodes, spatial_dim) and (n_elements, nodes_per_element):
        self.dim = self.nodes.shape[1]
        self.n_nodes = self.nodes.shape[0]
        self.n_elements = self.elements.shape[0]

    def to(self, device):
        # loads mesh datas to GPU
        self.nodes = self.nodes.to(device)
        self.elements = self.elements.to(device)
        return self

    def __repr__(self):
        return f"FEMMesh(nodes={self.n_nodes}, elems={self.n_elements}, dim={self.dim})"


# helper to create a DeepChem dataset from a mesh:
def mesh_to_dataset(mesh, solution=None):
    X = mesh.nodes.detach().cpu().numpy()
    # y is the ouput field (if known, othervise zeros)
    y = (
        solution.detach().cpu().numpy().reshape(-1, 1)
        if solution is not None
        else np.zeros((mesh.n_nodes, 1), dtype=np.float32)
    )

    # we store the element connectivity in the dataset metadata or a custom attribute
    # since standard NumpyDataset is (X,y,w), we can subclass or just Attach it.
    ds = dc.data.NumpyDataset(X=X, y=y)
    ds.elements = mesh.elements.cpu().numpy()  # Custom attribute
    return ds
