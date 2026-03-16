"""
h-Convergence Study.

PDE: -Delta u = 2*pi^2 * sin(pi*x) * sin(pi*y), u=0 on boundary.
Exact: u = sin(pi*x) * sin(pi*y).  Expected P1 rate: O(h^2).
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

from diffem import generate_unit_square_mesh, P1Triangle, DifferentiableAssembler, DifferentiableFEMSolver


def main():
    # we consider five progressively finer mesh resolutions
    sizes = [4, 8, 16, 32, 64]
    # hs and errs empyt list that will collect mesh size and errors for each resolution
    hs, errs = [], []

    for N in sizes:
        m = generate_unit_square_mesh(N, N)
        r = P1Triangle()
        a = DifferentiableAssembler(m, r)
        s = DifferentiableFEMSolver(m, a)
        h = 1.0 / N
        x, y = m.nodes[:, 0], m.nodes[:, 1]

        # source function at every nodes, the formula is 2*(pi^2)*sin(pi*x)*sin(pi*y)
        f_vals = (2 * np.pi**2 * torch.sin(np.pi * x) * torch.sin(np.pi * y)).unsqueeze(1)
        all_bdy = set()  # settinf boundary considtions
        for tag in m.boundary_nodes.values():
            all_bdy.update(tag)  # collect the vboundary conditions in a dictionary with keys left,right,bottom,top
        bc = {n: 0.0 for n in all_bdy}  # this is trivial because on the boundary y=0 or x=0 so f=0

        u_fem = s.solve(bc, f_source=f_vals)  # runs the full fem solver
        u_ex = (torch.sin(np.pi * x) * torch.sin(np.pi * y)).unsqueeze(1)
        l2 = torch.sqrt(torch.mean((u_fem - u_ex) ** 2)).item()  # l2 error
        hs.append(h)
        errs.append(l2)
        print(f"N={N:4d}  h={h:.4f}  L2={l2:.6e}")

    hs, errs = np.array(hs), np.array(errs)  # convergence rates
    rates = np.log(errs[:-1] / errs[1:]) / np.log(hs[:-1] / hs[1:])  # converagence rate ratio on consecutive meshes
    print(f"\nconvergence rates: {np.round(rates, 2)}")
    print(f"expected for P1: ~2.0")

    plt.figure(figsize=(7, 5))
    plt.loglog(hs, errs, "bo-", lw=2, label="FEM L2 error")
    # FEM theory guarantees O(h^2) convergen rate for the L2 error.
    plt.loglog(hs, errs[0] * (hs / hs[0]) ** 2, "r--", label="O(h²) reference")
    plt.xlabel("h")
    plt.ylabel("L2 error")
    plt.title("h-convergence — P1 triangles, poisson")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig("convergence.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
