"""
Differentiable Finite Element Method (FEM) solver for the 2D Poisson equation.

    -∇²u = f   in Ω ⊂ R²
     u = g      on ∂Ω  (Dirichlet boundary conditions)

Implementation details
----------------------
- Linear (P1) triangular elements — one degree of freedom per node
- Galerkin method: assembles global stiffness matrix K and load vector F
- Dirichlet BCs applied by row-zeroing + identity diagonal substitution
- Solve: u = K⁻¹ F  via torch.linalg.solve  (fully differentiable)
- Differentiable w.r.t. node coordinates AND source term f

This enables two use cases:
  1. Forward solve  — given f and g, compute u
  2. Inverse solve  — given noisy observations of u, recover f or material
                      parameters via gradient descent through the solver

Why this beats existing implementations
----------------------------------------
Ayman Khan's GSoC proto (https://github.com/amugoodbad229/deepchem/tree/gsoc-fem-demo)
implements FEM as standalone scripts. This repo goes further:
  - MeshDataset: structured dataset abstraction (maps to DeepChem Dataset)
  - MeshFeaturizer: generates meshes from PDE problem specifications
  - Full inverse problem inside a training loop (not just a demo)
  - Ready to drop into DeepChem TorchModel with zero algorithmic changes

References
----------
[1] Blechschmidt & Ernst, "Three ways to solve PDEs with neural networks",
    GAMM-Mitteilungen, 2021. https://arxiv.org/abs/2307.02494
[2] Hughes, T.J.R., "The Finite Element Method", Prentice-Hall, 1987.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import numpy as np

# core FEM assembly and solve

class FEMSolver(nn.Module):
    """Differentiable FEM solver for the 2D Poisson equation.

    Assembles the global stiffness matrix K and load vector F element-by-
    element using the Galerkin method with linear (P1) triangular basis
    functions, applies Dirichlet boundary conditions, and solves KU = F.

    The entire pipeline is differentiable via PyTorch autograd:
    - Gradients flow through torch.linalg.solve to K and F
    - Gradients flow from F back to the source term
    - Gradients flow from K back to node coordinates

    This makes the solver usable inside gradient-based optimisation loops
    for inverse problems (parameter identification).

    Parameters
    ----------
    None — stateless module, all problem data passed in forward().

    Example
    -------
    >>> import torch
    >>> from fem_solver import FEMSolver, make_unit_square_mesh
    >>> nodes, elements, boundary_mask = make_unit_square_mesh(nx=4, ny=4)
    >>> boundary_values = torch.zeros(nodes.shape[0])
    >>> source = torch.ones(nodes.shape[0])
    >>> solver = FEMSolver()
    >>> u = solver(nodes, elements, boundary_mask, boundary_values, source)
    >>> print(u.shape)
    torch.Size([25])
    """

    def forward(
        self,
        nodes: torch.Tensor,
        elements: torch.Tensor,
        boundary_mask: torch.Tensor,
        boundary_values: torch.Tensor,
        source: torch.Tensor,
    ) -> torch.Tensor:
        """Solve the Poisson equation on the given triangular mesh.

        Parameters
        ----------
        nodes : torch.Tensor, shape (N, 2)
            (x, y) coordinates of mesh nodes.
        elements : torch.Tensor, shape (E, 3), dtype=torch.long
            Node indices of each triangular element (CCW orientation).
        boundary_mask : torch.Tensor, shape (N,), dtype=torch.bool
            True at nodes where Dirichlet BCs are applied.
        boundary_values : torch.Tensor, shape (N,)
            Known solution values at boundary nodes.
        source : torch.Tensor, shape (N,)
            Source term f(x, y) evaluated at each node.

        Returns
        -------
        u : torch.Tensor, shape (N,)
            FEM solution at all mesh nodes.
        """
        N = nodes.shape[0]
        device = nodes.device

        K = torch.zeros(N, N, dtype=torch.float32, device=device)
        F = torch.zeros(N, dtype=torch.float32, device=device)

        # assemble global stiffness matrix and load vector 
        for e in elements:
            i, j, k = e[0].item(), e[1].item(), e[2].item()
            coords = nodes[[i, j, k]]          # (3, 2)

            x = coords[:, 0]
            y = coords[:, 1]

            # Signed area via cross product (positive = CCW)
            area = 0.5 * (
                (x[1] - x[0]) * (y[2] - y[0]) -
                (x[2] - x[0]) * (y[1] - y[0])
            )

            # Shape function gradients (constant over P1 element)
            #   φ_i(x,y) = (a_i + b_i*x + c_i*y) / (2*area)
            b = torch.stack([y[1] - y[2], y[2] - y[0], y[0] - y[1]])
            c = torch.stack([x[2] - x[1], x[0] - x[2], x[1] - x[0]])

            # Element stiffness: K_e[p,q] = ∫∇φ_p·∇φ_q dΩ
            #   = (b_p*b_q + c_p*c_q) / (4*|area|)
            K_local = (torch.outer(b, b) + torch.outer(c, c)) / (
                4.0 * torch.abs(area)
            )

            # Scatter into global K
            idx = [i, j, k]
            for p in range(3):
                for q in range(3):
                    K[idx[p], idx[q]] = K[idx[p], idx[q]] + K_local[p, q]

            # Element load vector: centroid quadrature
            #   F_e[p] ≈ f_avg * |area| / 3
            f_avg = (source[i] + source[j] + source[k]) / 3.0
            contrib = f_avg * torch.abs(area) / 3.0
            for ni in [i, j, k]:
                F[ni] = F[ni] + contrib

        # Apply Dirichlet boundary conditions
        # Method: row zeroing + identity diagonal
        #   K[i,:] = 0,  K[i,i] = 1,  F[i] = g_i  for boundary node i
        for i in range(N):
            if boundary_mask[i]:
                K[i, :] = 0.0
                K[i, i] = 1.0
                F[i] = boundary_values[i]

        # --- Solve: differentiable via autograd through linalg.solve ---
        u = torch.linalg.solve(K, F)
        return u


# mesh generation utilities

def make_unit_square_mesh(
    nx: int = 4,
    ny: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate a uniform triangular mesh on the unit square [0,1]².

    Divides the square into an nx×ny grid of rectangles, each split into
    two triangles by a diagonal cut, yielding 2*nx*ny elements total.

    Parameters
    ----------
    nx : int
        Number of divisions along x. Total nodes = (nx+1)*(ny+1).
    ny : int
        Number of divisions along y.

    Returns
    -------
    nodes : torch.Tensor, shape ((nx+1)*(ny+1), 2)
        Node (x, y) coordinates.
    elements : torch.Tensor, shape (2*nx*ny, 3), dtype=torch.long
        Triangle connectivity (CCW node indices).
    boundary_mask : torch.Tensor, shape ((nx+1)*(ny+1),), dtype=torch.bool
        True at nodes on the boundary of the unit square.

    Example
    -------
    >>> nodes, elements, boundary_mask = make_unit_square_mesh(nx=3, ny=3)
    >>> nodes.shape
    torch.Size([16, 2])
    >>> elements.shape
    torch.Size([18, 3])
    """
    xs = np.linspace(0.0, 1.0, nx + 1, dtype=np.float32)
    ys = np.linspace(0.0, 1.0, ny + 1, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    nodes_np = np.column_stack([xx.ravel(), yy.ravel()])  # (N, 2)

    elems = []
    for j in range(ny):
        for i in range(nx):
            n00 = j * (nx + 1) + i
            n10 = j * (nx + 1) + i + 1
            n01 = (j + 1) * (nx + 1) + i
            n11 = (j + 1) * (nx + 1) + i + 1
            elems.append([n00, n10, n01])   # lower-left triangle
            elems.append([n10, n11, n01])   # upper-right triangle

    nodes = torch.tensor(nodes_np, dtype=torch.float32)
    elements = torch.tensor(elems, dtype=torch.long)

    x = nodes[:, 0]
    y = nodes[:, 1]
    boundary_mask = (x == 0) | (x == 1) | (y == 0) | (y == 1)

    return nodes, elements, boundary_mask


def make_boundary_values(
    nodes: torch.Tensor,
    boundary_mask: torch.Tensor,
    fn=None,
) -> torch.Tensor:
    """Evaluate boundary condition function g(x, y) at boundary nodes.

    Parameters
    ----------
    nodes : torch.Tensor, shape (N, 2)
    boundary_mask : torch.Tensor, shape (N,), dtype=torch.bool
    fn : callable or None
        g(x, y) -> scalar. If None, returns zero BCs.

    Returns
    -------
    boundary_values : torch.Tensor, shape (N,)
        g evaluated at all nodes (interior nodes set to 0).
    """
    boundary_values = torch.zeros(nodes.shape[0], dtype=torch.float32)
    if fn is not None:
        for i in range(nodes.shape[0]):
            if boundary_mask[i]:
                boundary_values[i] = fn(
                    nodes[i, 0].item(), nodes[i, 1].item())
    return boundary_values