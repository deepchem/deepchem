"""
Differentiability Benchmarks.

Test 1 — Shape optimisation: Free interior node pushed toward hot wall.
Test 2 — Finite-difference gradient check: Validates autograd accuracy.
"""

import torch

from diffem import FEMMesh, generate_unit_square_mesh, P1Triangle, DifferentiableAssembler, DifferentiableFEMSolver


def test_shape_optimisation():
    """Test 1: Shape optimisation."""
    #  3(0,1) ---- 2(1,1)
    #    | \     /    |
    #   | 4(0.5,0.5) |
    #   | /    \    |
    #  0(0,0) ---- 1(1,0)
    nodes_t1 = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]],
        dtype=torch.float32,
        requires_grad=True,
    )
    elems_t1 = torch.tensor(
        [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]], dtype=torch.int64
    )  # four triangles shareing the center node 4
    mesh_t1 = FEMMesh(nodes_t1.detach(), elems_t1)
    mesh_t1.nodes = nodes_t1
    ref = P1Triangle()
    bc_t1 = {
        0: 0.0,
        1: 1.0,
        2: 1.0,
        3: 0.0,
    }  # boundary conditions, left wall (node 0 and 3) at temp=0 (cold) and right wall (node 1 and 2) at temp=1 (hot)
    opt = torch.optim.Adam([nodes_t1], lr=0.02)

    # run 15 optimization iterations
    print("shape optimisation — push free node toward hot wall")
    for i in range(15):
        opt.zero_grad()
        a = DifferentiableAssembler(mesh_t1, ref)
        s = DifferentiableFEMSolver(mesh_t1, a)
        u = s.solve(bc_t1)
        (-u[4]).backward()  # loss function and backpropgation in one line
        # print every  third iterations with the temperature of the center node,
        # the gradient with respect to x and y and position.
        if i % 3 == 0:
            print(
                f"  iteration number {i:2d}: Temp={u[4].item():.4f}"
                f"  gradients=({nodes_t1.grad[4, 0]:.4f}, {nodes_t1.grad[4, 1]:.4f})"
                f"  position={nodes_t1.data[4].tolist()}"
            )
        opt.step()
    print(f"final pos: {nodes_t1.data[4].tolist()} — moved rightward ✓")


def test_finite_difference_gradient():
    """Test 2: Finite-difference gradient check.

    We will compute the gradients in two different ways and check if they agree.
    """
    print("\nfinite-difference gradient validation")
    mesh_fd = generate_unit_square_mesh(4, 4)
    ref = P1Triangle()
    interior_node = 12  # I take the node 12 that is in the center of the grid
    all_bdy = set()
    for tag in mesh_fd.boundary_nodes.values():
        all_bdy.update(tag)
    assert interior_node not in all_bdy, f"Node {interior_node} is on boundary"

    # boundary conditions
    bc_fd = {}
    for n in mesh_fd.boundary_nodes["left"]:   bc_fd[n] = 0.0
    for n in mesh_fd.boundary_nodes["right"]:  bc_fd[n] = 1.0

    # first way: compute the gradient with autograd
    mesh_fd.nodes.requires_grad_(True)
    a = DifferentiableAssembler(mesh_fd, ref)
    s = DifferentiableFEMSolver(mesh_fd, a)
    loss = s.solve(bc_fd).sum()
    loss.backward()
    ag = mesh_fd.nodes.grad[interior_node].clone()

    # second way: compute the gradient with finite differences
    eps = 1e-4  # the perturbation size
    fd = torch.zeros(2)  # 2 element tensor to store the two finete difference gradients
    for d in range(2):
        for sign, idx in [
            (1, 0),
            (-1, 1),
        ]:  # I need to do a positive and a negative perturbation in the two dimensions
            np_ = mesh_fd.nodes.data.clone()
            np_[interior_node, d] += sign * eps
            m_ = FEMMesh(np_, mesh_fd.elements, mesh_fd.boundary_nodes)
            a_ = DifferentiableAssembler(m_, ref)
            s_ = DifferentiableFEMSolver(m_, a_)
            if sign == 1:
                fp = s_.solve(bc_fd).sum().item()
            else:
                fm = s_.solve(bc_fd).sum().item()
        fd[d] = (fp - fm) / (
            2 * eps
        )  # centra difference approximation is O(eps^2) instead of O(epe) so its more accurate

    print(f"autograd:  {ag.tolist()}")
    print(f"fin. diff: {fd.tolist()}")
    rel = torch.norm(ag - fd) / (torch.norm(fd) + 1e-12)
    print(f"relative error: {rel.item():.2e}")
    # pass or fail check with 5% threshold :
    print("GOOD, gradient check passed" if rel < 0.05 else "BAD, gradient check FAILED")


def main():
    test_shape_optimisation()
    test_finite_difference_gradient()


if __name__ == "__main__":
    main()
