"""
Forward Solve — Laplace on a 10x10 Mesh.

-Delta u = 0, u=0 left, u=1 right.  Exact solution: u(x,y) = x.
"""

import torch
import matplotlib.pyplot as plt

from diffem import generate_unit_square_mesh, P1Triangle, DifferentiableAssembler, DifferentiableFEMSolver


def main():
    mesh = generate_unit_square_mesh(10, 10)  # create a 10x10 triangluar mash on the uniti square
    ref = P1Triangle()  # create the linear triangle with its shape functions, gradients and centroid quadrature rule
    asm = DifferentiableAssembler(mesh, ref)  # create the assembler for build K and load F
    slv = DifferentiableFEMSolver(mesh, asm)  # create the solver

    # boundary conditions
    bc = {}
    for n in mesh.boundary_nodes["left"]:   bc[n] = 0.0  # left wall is fixed at 0.0
    for n in mesh.boundary_nodes["right"]:  bc[n] = 1.0  # right wall is fixed at 1.0
    # bc would have 22 entries because 11 points on the left and 11 points on right

    u = slv.solve(bc)  # runs the full pipleine
    u_exact = mesh.nodes[:, 0:1]  # analytical solution

    l2 = torch.sqrt(torch.mean((u - u_exact) ** 2)).item()  # L2 error computation
    linf = torch.max(torch.abs(u - u_exact)).item()  # L^oo error computation
    print(f"nodes: {mesh.n_nodes}, elements: {mesh.n_elements}")
    print(f"L2 error:   {l2:.2e}")
    print(f"Linf error: {linf:.2e}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # visualization side by side comparison
    nds = mesh.nodes.detach().numpy()
    els = mesh.elements.numpy()

    # iterate twice on the the points on the FEM and on the points of the
    # solution and draw a  filled colotr plot on the triangluar mesh
    for ax, data, title in zip(
        axes,
        [u.detach().numpy().ravel(), u_exact.numpy().ravel()],
        ["FEM Solution", "Exact u = x"],
    ):
        tc = ax.tripcolor(nds[:, 0], nds[:, 1], els, data, shading="gouraud", cmap="viridis")
        ax.triplot(nds[:, 0], nds[:, 1], els, "k-", lw=0.3, alpha=0.3)
        ax.set_title(title)
        ax.set_aspect("equal")
        plt.colorbar(tc, ax=ax)
    plt.tight_layout()
    plt.savefig("forward_solve.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
