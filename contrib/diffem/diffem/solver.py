"""
Solver with Penalty-Method BCs.

Instead of zeroing rows (requires dense conversion and breaks symmetry),
we use the **penalty method**: add beta to K_{ii} and set F_i = beta * g_i
for each Dirichlet node.  With beta = 1e10 the constraint is accurate to
~10 digits while keeping the matrix symmetric and the assembly sparse-friendly.
"""

import torch


# BCs enforcer and linear sistem solver class
class DifferentiableFEMSolver:
    def __init__(self, mesh, assembler, penalty=1e10):
        self.mesh = mesh
        self.assembler = assembler
        self.penalty = penalty

    #  after I builded the K siffnes matrix I have a sistem of equations
    # that have infinite numbers of solutions, for keep it simple I impose the
    # BCs whit the identity/penalty method, specifically the zerorow method
    def _apply_penalty_bcs(self, K_dense, F, bc_dict):
        # We need to clone to keep autograd happy if we modify in-place
        K = K_dense.clone()
        F = F.clone()
        for node, val in bc_dict.items():
            K[node, node] = K[node, node] + self.penalty
            F[node] = self.penalty * val
        return K, F

    def solve(self, bc_dict, f_source=None, k=None):
        nn = self.mesh.n_nodes

        # stiffness matrix setup
        K_sp = self.assembler.assemble_stiffness(k=k)
        K_dense = K_sp.to_dense()

        # force setup
        if f_source is not None:
            F = self.assembler.assemble_load(f_source)
        else:
            F = torch.zeros(
                (nn, 1), dtype=torch.float32, device=self.mesh.nodes.device
            )

        K_bc, F_bc = self._apply_penalty_bcs(K_dense, F, bc_dict)
        # linear Solve
        # u = K_inv * F
        return torch.linalg.solve(K_bc, F_bc)
