import torch
from torch.autograd import grad
from typing import Callable



def leapfrog(p: torch.Tensor,
             q: torch.Tensor,
             Func: Callable,
             dt: float,
             N: int,
             is_hamiltonian: bool = True) -> torch.Tensor:
    """
    Leapfrog integrator for simulating Hamiltonian or general dynamics.

    Parameters
    ----------
    p : torch.Tensor
        Initial momentum tensor, shape (1,)
    q : torch.Tensor
        Initial position tensor, shape (1,)
    Func : callable
        If is_hamiltonian=True: Function that returns Hamiltonian H(z) where z=[p,q]
        If is_hamiltonian=False: FuncCallabletion that returns derivatives [dp/dt, dq/dt]
    dt : float
        Time step size for integration
    N : int
        Number of steps to simulate

    Returns
    -------
    trajectories : torch.Tensor
        Trajectories of shape (T, 2) which contains (p, q) pairs at each time step

    Examples
    --------
    >>> import torch
    >>> def hamiltonian_fn(z):
    ...     p, q = z
    ...     return p**2 + q**2
    >>> p0 = torch.tensor([-0.0429839], requires_grad=True)
    >>> q0 = torch.tensor([0.44432778], requires_grad=True)
    >>> dt = 0.1
    >>> N = 100
    >>> trajectories = leapfrog(p0, q0, hamiltonian_fn, dt, N, is_hamiltonian=True)
    >>> trajectories.shape
    torch.Size([100, 2])
    """

    trajectories = torch.empty((N, 2))

    p = p.clone().requires_grad_(True)
    q = q.clone().requires_grad_(True)

    if is_hamiltonian:
        hamilt = Func(torch.cat([p, q], dim=-1))
        dpdt = -grad(hamilt.sum(), q)[0]

        for i in range(N):

            p_half = p + (dt / 2) * dpdt

            trajectories[i, :1] = p.detach()
            trajectories[i, 1:] = q.detach()

            hamil = Func(torch.cat([p_half, q], dim=-1))
            dqdt = grad(hamil.sum(), p_half)[0]
            q_next = q + dt * dqdt

            hamil = Func(torch.cat([p_half, q_next], dim=-1))
            dpdt = -grad(hamil.sum(), q_next)[0]
            p_next = p_half + (dt / 2) * dpdt

            p, q = p_next, q_next

    else:
        time_drvt = Func((p, q))
        dpdt = time_drvt[0]

        for i in range(N):

            p_half = p + dpdt * (dt / 2)

            trajectories[i, :1] = p
            trajectories[i, 1:] = q

            time_drvt = Func((p_half, q))
            dqdt = time_drvt[1]
            q_next = q + dqdt * dt

            time_drvt = Func((p_half, q_next))
            dpdt = time_drvt[0]
            p_next = p_half + dpdt * (dt / 2)

            p, q = p_next, q_next

    return trajectories
