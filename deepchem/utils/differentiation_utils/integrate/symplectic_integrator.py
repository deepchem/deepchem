import torch
from torch.autograd import grad
from typing import Callable


def leapfrog(q0: torch.Tensor,
             p0: torch.Tensor,
             fcn: Callable[..., torch.Tensor],
             dt: float,
             T: int,
             is_hamiltonian: bool = True,
             device='cpu',
             enable_graph=True) -> torch.Tensor:
    """
    Leapfrog integrator for simulating Hamiltonian or general dynamics.

    A second-order symplectic integrator that alternates between half-step
    momentum updates and full-step position updates, which conserves the
    energy in physical system

    Parameters
    ----------
    q0 : torch.Tensor
        Initial position tensor, shape (1,)
    p0 : torch.Tensor
        Initial momentum tensor, shape (1,)
    fcn : callable
        If is_hamiltonian=True: Function that returns Hamiltonian H(z) where z=[p,q]
        If is_hamiltonian=False: FuncCallabletion that returns derivatives [dp/dt, dq/dt]
    dt : float
        Time step size for integration
    T : int
        Number of time-steps to simulate
    is_hamiltonian : bool
        boolean flag for indicating either hamiltonian or non-hamiltonian function
    device : str
        The device to compute leapfrog method, Default value is "cpu"
    enable_graph : bool
        flag to enable/disable gradients

    Returns
    -------
    trajectories : torch.Tensor
        Trajectories of shape (T, 2) which contains (p, q) pairs at each time step

    Examples
    --------
    >>> import torch
    >>> def hamiltonian_fn(q, p):
    ...     return q**2 + p**2
    >>> q0 = torch.rand(1).unsqueeze(0)
    >>> p0 = torch.rand(1).unsqueeze(0)
    >>> dt = 0.1
    >>> T = 100
    >>> trajectories = leapfrog(p0, q0, hamiltonian_fn, dt, T, is_hamiltonian=True)
    >>> trajectories.shape
    torch.Size([100, 1, 2])

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Leapfrog_integration
    .. [2] https://www.johndcook.com/blog/2020/07/13/leapfrog-integrator/
    .. [3] https://cvarin.github.io/CSci-Survival-Guide/leapfrog.html
    .. [4] https://github.com/zhengdao-chen/SRNN

    """

    trajectories = torch.empty((T, q0.shape[0], q0.shape[1] * 2),
                               requires_grad=False).to(device)

    q = q0
    p = p0
    q.requires_grad_()
    p.requires_grad_()

    # checking if "fcn" follows hamiltonian "T + V" format otherwise continue with else block for non-hamiltonian
    if is_hamiltonian:
        hamilt = fcn(q, p)
        dpdt = -grad(hamilt.sum(), q, create_graph=enable_graph)[0]

        for i in range(T):

            p_half = p + (dt / 2) * dpdt

            trajectories[i, :, :q0.shape[1]] = q
            trajectories[i, :, q0.shape[1]:] = p

            hamil = fcn(q, p_half)
            dqdt = grad(hamil.sum(), p_half, create_graph=enable_graph)[0]
            q_next = q + dt * dqdt

            hamil = fcn(q_next, p_half)
            dpdt = -grad(hamil.sum(), q_next, create_graph=enable_graph)[0]
            p_next = p_half + (dt / 2) * dpdt

            q, p = q_next, p_next
    else:
        time_drvt = fcn(q, p)
        dpdt = time_drvt[0]

        for i in range(T):

            p_half = p + dpdt * (dt / 2)

            trajectories[i, :, :q0.shape[0]] = q
            trajectories[i, :, q0.shape[0]:] = p

            time_drvt = fcn(q, p_half)
            dqdt = time_drvt[1]
            q_next = q + dqdt * dt

            time_drvt = fcn(q_next, p_half)
            dpdt = time_drvt[0]
            p_next = p_half + dpdt * (dt / 2)

            q, p = q_next, p_next

    return trajectories
