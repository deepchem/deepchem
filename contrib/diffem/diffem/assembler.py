"""
Differentiable Stiffness Assembly.

For the Poisson equation -div(k * grad(u)) = f, the element stiffness is:
    K^e = k^e * |Omega^e| * B^T B
where B = grad_hat(N) * J^{-1} and |Omega^e| = 0.5 * |det J|.

Everything is batched via ``torch.bmm`` — no Python loops over elements.
"""

import torch


# poisson equation is -L(u)=f, the following class thelp us for the backpropagation
class DifferentiableAssembler:
    def __init__(self, mesh, ref_element):
        self.mesh = mesh
        self.ref = ref_element

    def assemble_stiffness(self, k=None):
        ne = self.mesh.n_elements
        nn = self.mesh.n_nodes
        coords = self.mesh.nodes[self.mesh.elements]

        grad_ref = self.ref.grad_basis(self.ref.quad_points[0:1]).expand(ne, -1, -1)
        # map geometry tell us how much the triangle get distorced or deformed from its started config for
        # that we need the Jacobian that tell us the deformations and the its determinant that is the change in areas.
        J = torch.bmm(coords.transpose(1, 2), grad_ref)  # this is just J = X^T * dN/dxi
        det_J = torch.det(J)
        J_inv = torch.inverse(J)
        # transform gradients to global coordinates, so I am doing grad_global = grad_ref * J_inv
        grad_phys = torch.bmm(grad_ref, J_inv)

        # compute local stiffness matrix (K_local) for each individual triangle so I am doing
        # K_local = Integral( grad^T * grad ) dV but for omogenous triangles, grad is constant, so Integral = Area * (grad^T * grad)
        # Integral = Area * (grad^T * grad) and Area = 0.5 * |detJ| (Triangle area in 2D)
        area = 0.5 * torch.abs(det_J)
        K_local = torch.bmm(grad_phys, grad_phys.transpose(1, 2))

        scale = area.clone()
        if k is not None:
            scale = scale * k
        K_local = K_local * scale.view(-1, 1, 1)

        elems = self.mesh.elements
        rows = elems.unsqueeze(2).expand(-1, -1, 3).reshape(-1)  # (N, 3, 3)  [[0,0,0], [1,1,1]...]
        cols = elems.unsqueeze(1).expand(-1, 3, -1).reshape(-1)  # (N, 3, 3)  [[0,1,2], [0,1,2]...]
        K_global = torch.sparse_coo_tensor(
            torch.stack([rows, cols]), K_local.reshape(-1), size=(nn, nn)
        )
        return K_global

    def assemble_load(self, f_vals):
        ne = self.mesh.n_elements
        coords = self.mesh.nodes[self.mesh.elements]
        grad_ref = self.ref.grad_basis(self.ref.quad_points[0:1]).expand(ne, -1, -1)
        J = torch.bmm(coords.transpose(1, 2), grad_ref)
        area = 0.5 * torch.abs(torch.det(J))

        F = torch.zeros_like(f_vals)
        for loc in range(3):
            nids = self.mesh.elements[:, loc]
            contrib = (area / 3.0) * f_vals[nids].squeeze(-1)
            F.scatter_add_(0, nids.unsqueeze(1), contrib.unsqueeze(1))
        return F
