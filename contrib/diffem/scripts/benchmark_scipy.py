"""
Benchmark Against SciPy Sparse Solver.

Compare our PyTorch solve to ``scipy.sparse.linalg.spsolve`` on the same
problem — validates correctness and gives a timing baseline.
"""

import time
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from diffem import generate_unit_square_mesh, P1Triangle, DifferentiableAssembler, DifferentiableFEMSolver


# function that solve the exact provlem as DifferentiableFEMSolver but using numpy instead of pytorch
def scipy_solve(mesh, bc_dict, penalty=1e10):
    ne, nn = mesh.n_elements, mesh.n_nodes
    coords = mesh.nodes.detach().numpy()
    elems = mesh.elements.numpy()
    grad_ref = np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]])

    rows, cols, vals = [], [], []
    for e in range(ne):
        en = elems[e]
        xe = coords[en]
        J = xe.T @ grad_ref
        Jinv = np.linalg.inv(J)
        B = grad_ref @ Jinv
        area = 0.5 * abs(np.linalg.det(J))
        Ke = area * (B @ B.T)
        for a in range(3):
            for b in range(3):
                rows.append(en[a])
                cols.append(en[b])
                vals.append(Ke[a, b])

    K = csr_matrix((vals, (rows, cols)), shape=(nn, nn))
    F = np.zeros(nn)
    for node, val in bc_dict.items():
        K[node, node] += penalty
        F[node] = penalty * val
    return spsolve(K, F)


def main():
    print(f"{'N':>5} {'Nodes':>7} {'Elems':>7} {'Max|diff|':>12} {'Torch ms':>10} {'SciPy ms':>10}")
    print("-" * 58)
    for N in [5, 10, 20, 40]:
        m = generate_unit_square_mesh(N, N)
        r = P1Triangle()
        a = DifferentiableAssembler(m, r)
        s = DifferentiableFEMSolver(m, a)
        bc = {}
        for n in m.boundary_nodes["left"]:   bc[n] = 0.0
        for n in m.boundary_nodes["right"]:  bc[n] = 1.0

        t0 = time.perf_counter()
        ut = s.solve(bc)
        dt_t = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        us = scipy_solve(m, bc)
        dt_s = (time.perf_counter() - t0) * 1000

        diff = np.max(np.abs(ut.detach().numpy().ravel() - us))
        print(f"{N:5d} {m.n_nodes:7d} {m.n_elements:7d} {diff:12.2e} {dt_t:10.1f} {dt_s:10.1f}")


if __name__ == "__main__":
    main()
