"""
DeepChem Integration Demo.

Demonstrates the DifferentiableFEMModel on a simple test problem where
the exact solution is known: -Delta u = 0, u=0 left, u=1 right => u(x,y) = x.
"""

import numpy as np

from diffem import generate_unit_square_mesh, P1Triangle, mesh_to_dataset, DifferentiableFEMModel


def main():
    # demonstrate DeepChem integration on a simple test problem where
    # the exact solution is knowm
    mesh_dc = generate_unit_square_mesh(4, 4)
    ref = P1Triangle()

    # boundary conditions
    bc_dc = {}
    for n in mesh_dc.boundary_nodes["left"]:   bc_dc[n] = 0.0
    for n in mesh_dc.boundary_nodes["right"]:  bc_dc[n] = 1.0

    ds = mesh_to_dataset(mesh_dc)
    model = DifferentiableFEMModel(
        mesh_dc, ref, bc_dc, learning_rate=0.01, batch_size=mesh_dc.n_nodes
    )  # model inizialization
    # forward pass (assemble stiffness matrix, applies bc, solve the linear system
    # and returns the predict solution at the nodes
    u_pred = model.predict(ds)
    print(f"predict() output shape: {u_pred.shape}")
    print(
        f"max error vs exact (u=x): "
        f"{np.max(np.abs(u_pred.ravel() - mesh_dc.nodes[:, 0].cpu().numpy())):.2e}"
    )


if __name__ == "__main__":
    main()
