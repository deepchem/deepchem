"""Basic tests for the diffem package."""

import torch
import numpy as np

from diffem import (
    FEMMesh,
    generate_unit_square_mesh,
    P1Triangle,
    DifferentiableAssembler,
    DifferentiableFEMSolver,
)


def test_mesh_creation():
    mesh = generate_unit_square_mesh(4, 4)
    assert mesh.n_nodes == 25
    assert mesh.n_elements == 32
    assert mesh.dim == 2
    print("test_mesh_creation PASSED")


def test_partition_of_unity():
    """Partition-of-unity sanity check."""
    ref_elem = P1Triangle()
    pt = torch.tensor([[0.2, 0.3]])  # dummy point inside the triangle (e.g., psi=0.2, eta=0.3)
    # calculations of the shape function and check of partition of unity rule: N0 + N1 + N2 = 1
    N = ref_elem.basis(pt)
    assert abs(N.sum().item() - 1.0) < 1e-6
    print(f"N(0.2, 0.3) = {N.tolist()} — sum = {N.sum().item():.6f} (partition of unity holds)")
    print("test_partition_of_unity PASSED")


def test_laplace_exact():
    """Laplace u=x should be recovered exactly by linear FEM."""
    mesh = generate_unit_square_mesh(4, 4)
    ref = P1Triangle()
    asm = DifferentiableAssembler(mesh, ref)
    slv = DifferentiableFEMSolver(mesh, asm)

    bc = {}
    for n in mesh.boundary_nodes["left"]:   bc[n] = 0.0
    for n in mesh.boundary_nodes["right"]:  bc[n] = 1.0

    u = slv.solve(bc)
    u_exact = mesh.nodes[:, 0:1]
    linf = torch.max(torch.abs(u - u_exact)).item()
    assert linf < 1e-5, f"Linf error too large: {linf}"
    print(f"test_laplace_exact PASSED (Linf = {linf:.2e})")


def test_boundary_nodes():
    """Check boundary node tagging."""
    mesh = generate_unit_square_mesh(4, 4)
    # be sure that the left boundary is [0, 1, 2, 3, 4].
    assert mesh.boundary_nodes["left"] == [0, 1, 2, 3, 4]
    print(f"boundary left: {mesh.boundary_nodes['left']}")
    print("test_boundary_nodes PASSED")


if __name__ == "__main__":
    test_mesh_creation()
    test_partition_of_unity()
    test_laplace_exact()
    test_boundary_nodes()
    print("\nAll tests passed!")
