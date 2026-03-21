"""
Tests for fem_solver.py

Run with:
    pytest tests/test_fem.py -v
    pytest tests/test_fem.py -v -m slow   # includes integration tests
"""

import pytest
import numpy as np
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from fem_solver import FEMSolver, make_unit_square_mesh, make_boundary_values

#helpers

def _minimal_mesh():
    """2-element unit square mesh — fast for unit tests."""
    nodes = torch.tensor(
        [[0., 0.], [1., 0.], [0., 1.], [1., 1.]], dtype=torch.float32)
    elements = torch.tensor([[0, 1, 2], [1, 3, 2]], dtype=torch.long)
    boundary_mask = torch.tensor([True, True, True, False])
    boundary_values = torch.zeros(4)
    source = torch.ones(4)
    return nodes, elements, boundary_mask, boundary_values, source


# shape and basic correctness

def test_output_shape_minimal_mesh():
    """Solution vector must have shape (N,)."""
    nodes, elements, bm, bv, src = _minimal_mesh()
    solver = FEMSolver()
    u = solver(nodes, elements, bm, bv, src)
    assert u.shape == (4,), f"Expected (4,), got {u.shape}"


def test_output_shape_unit_square_mesh():
    """Output shape must match number of nodes for larger mesh."""
    nodes, elements, bm = make_unit_square_mesh(nx=6, ny=6)
    bv = torch.zeros(nodes.shape[0])
    src = torch.ones(nodes.shape[0])
    solver = FEMSolver()
    u = solver(nodes, elements, bm, bv, src)
    assert u.shape == (nodes.shape[0],)


#boundary conditions

def test_dirichlet_bcs_enforced_zero():
    """Homogeneous Dirichlet BCs must be exactly zero at boundary."""
    nodes, elements, bm = make_unit_square_mesh(nx=4, ny=4)
    bv = torch.zeros(nodes.shape[0])
    src = torch.ones(nodes.shape[0])
    solver = FEMSolver()
    u = solver(nodes, elements, bm, bv, src)
    assert torch.allclose(u[bm], torch.zeros(bm.sum()), atol=1e-5), (
        f"Non-zero values at boundary: {u[bm]}"
    )


def test_dirichlet_bcs_enforced_nonzero():
    """Non-homogeneous Dirichlet BCs must be satisfied exactly."""
    nodes, elements, bm = make_unit_square_mesh(nx=4, ny=4)
    # g(x, y) = x (linear BC)
    bv = make_boundary_values(nodes, bm, fn=lambda x, y: x)
    src = torch.zeros(nodes.shape[0])
    solver = FEMSolver()
    u = solver(nodes, elements, bm, bv, src)
    assert torch.allclose(u[bm], bv[bm], atol=1e-5), (
        f"BC mismatch: max error = {(u[bm] - bv[bm]).abs().max():.2e}"
    )


def test_zero_source_zero_bc_gives_zero_solution():
    """With f=0 and g=0, solution must be identically zero."""
    nodes, elements, bm = make_unit_square_mesh(nx=4, ny=4)
    bv = torch.zeros(nodes.shape[0])
    src = torch.zeros(nodes.shape[0])
    solver = FEMSolver()
    u = solver(nodes, elements, bm, bv, src)
    assert torch.allclose(u, torch.zeros_like(u), atol=1e-6), (
        f"Expected zero solution, got max |u| = {u.abs().max():.2e}"
    )


#differentiablity

def test_gradients_flow_through_source():
    """Loss.backward() must produce non-zero gradients w.r.t. source."""
    nodes, elements, bm = make_unit_square_mesh(nx=4, ny=4)
    bv = torch.zeros(nodes.shape[0])
    source = torch.ones(nodes.shape[0], requires_grad=True)

    solver = FEMSolver()
    u = solver(nodes, elements, bm, bv, source)
    loss = u.sum()
    loss.backward()

    assert source.grad is not None, "No gradient w.r.t. source"
    assert not torch.all(source.grad == 0), "All-zero gradient w.r.t. source"


def test_gradients_flow_through_nodes():
    """Loss.backward() must produce non-zero gradients w.r.t. node coords."""
    nodes_np, elements, bm = make_unit_square_mesh(nx=3, ny=3)
    nodes = nodes_np.clone().detach().requires_grad_(True)
    bv = torch.zeros(nodes.shape[0])
    source = torch.ones(nodes.shape[0])

    solver = FEMSolver()
    u = solver(nodes, elements, bm, bv, source)
    loss = u.sum()
    loss.backward()

    assert nodes.grad is not None, "No gradient w.r.t. node coordinates"
    assert not torch.all(nodes.grad == 0), (
        "All-zero gradient w.r.t. node coordinates"
    )


def test_gradient_not_nan():
    """Gradients must be finite (no NaN from degenerate elements)."""
    nodes, elements, bm = make_unit_square_mesh(nx=4, ny=4)
    bv = torch.zeros(nodes.shape[0])
    source = torch.ones(nodes.shape[0], requires_grad=True)

    solver = FEMSolver()
    u = solver(nodes, elements, bm, bv, source)
    u.sum().backward()

    assert torch.isfinite(source.grad).all(), (
        f"NaN or Inf in source gradient"
    )


def test_mesh_node_count():
    """make_unit_square_mesh must produce (nx+1)*(ny+1) nodes."""
    for nx, ny in [(2, 2), (4, 4), (6, 8)]:
        nodes, elements, bm = make_unit_square_mesh(nx=nx, ny=ny)
        assert nodes.shape == ((nx + 1) * (ny + 1), 2)
        assert elements.shape == (2 * nx * ny, 3)


def test_mesh_boundary_mask_unit_square():
    """All boundary nodes must be on the edge of [0,1]²."""
    nodes, _, bm = make_unit_square_mesh(nx=4, ny=4)
    boundary_nodes = nodes[bm]
    on_edge = (
        (boundary_nodes[:, 0] == 0) | (boundary_nodes[:, 0] == 1) |
        (boundary_nodes[:, 1] == 0) | (boundary_nodes[:, 1] == 1)
    )
    assert on_edge.all(), "Boundary mask includes interior nodes"


def test_mesh_node_coordinates_in_unit_square():
    """All nodes must lie within [0,1]²."""
    nodes, _, _ = make_unit_square_mesh(nx=5, ny=5)
    assert (nodes >= 0).all() and (nodes <= 1).all()

#inverse problem

@pytest.mark.slow
def test_inverse_problem_source_recovery():
    """
    Inverse problem: recover unknown source f from noisy observations of u.

    Setup:
      - True source: f(x,y) = sin(πx)sin(πy) on 8x8 mesh
      - Forward solve to get u_true
      - Add Gaussian noise to get u_obs
      - Optimize source_est via gradient descent to minimise ||u(f_est) - u_obs||²
      - Assert loss decreases and final source is closer to truth than init
    """
    torch.manual_seed(42)
    nx, ny = 8, 8
    nodes, elements, bm = make_unit_square_mesh(nx=nx, ny=ny)
    bv = torch.zeros(nodes.shape[0])

    # True source: f = sin(πx)sin(πy)
    x = nodes[:, 0]
    y = nodes[:, 1]
    true_source = torch.sin(np.pi * x) * torch.sin(np.pi * y)

    # Forward solve to get ground-truth solution
    solver = FEMSolver()
    with torch.no_grad():
        u_true = solver(nodes, elements, bm, bv, true_source)

    # Add 1% Gaussian noise
    noise_level = 0.01 * u_true.abs().max()
    u_obs = u_true + noise_level * torch.randn_like(u_true)

    # Learnable source (initialised to zero)
    source_est = torch.zeros(nodes.shape[0], requires_grad=True)
    optimizer = torch.optim.Adam([source_est], lr=0.05)

    losses = []
    for step in range(100):
        optimizer.zero_grad()
        u_pred = solver(nodes, elements, bm, bv, source_est)
        loss = ((u_pred - u_obs) ** 2).mean()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    # Loss must decrease
    assert losses[-1] < losses[0], (
        f"Loss did not decrease: {losses[0]:.6f} -> {losses[-1]:.6f}"
    )

    # Final estimate must be closer to truth than zero initialisation
    init_error = (true_source ** 2).mean().item()
    final_error = ((source_est.detach() - true_source) ** 2).mean().item()
    assert final_error < init_error, (
        f"Source estimate did not improve: "
        f"init_error={init_error:.4f}, final_error={final_error:.4f}"
    )


@pytest.mark.slow
def test_mesh_refinement_reduces_error():
    """
    Finer mesh must give lower error on a problem with known exact solution.

    Exact solution: u(x,y) = sin(πx)sin(πy)
    Source term:    f(x,y) = 2π²sin(πx)sin(πy)  (from -∇²u = f)

    FEM error must decrease as mesh is refined (h-convergence).
    """
    solver = FEMSolver()
    errors = []

    for nx in [4, 8, 16]:
        nodes, elements, bm = make_unit_square_mesh(nx=nx, ny=nx)
        x = nodes[:, 0]
        y = nodes[:, 1]

        # Exact source for this problem
        source = 2 * (np.pi ** 2) * torch.sin(np.pi * x) * torch.sin(np.pi * y)
        bv = torch.zeros(nodes.shape[0])

        with torch.no_grad():
            u_fem = solver(nodes, elements, bm, bv, source)

        u_exact = torch.sin(np.pi * x) * torch.sin(np.pi * y)
        # Ignore boundary where u_exact=0 by construction
        interior = ~bm
        error = (u_fem[interior] - u_exact[interior]).abs().mean().item()
        errors.append(error)

    # Error must strictly decrease with refinement
    assert errors[0] > errors[1] > errors[2], (
        f"Mesh refinement did not reduce error: {errors}"
    )