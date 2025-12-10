# Adapted from https://github.com/diffqc/dqc/blob/master/dqc/api/properties.py
import torch
from typing import Optional
from deepchem.utils.dft_utils import BaseQCCalc
from deepchem.utils import memoize_method


@memoize_method
def dipole_moment(qc: BaseQCCalc) -> torch.Tensor:
    """Returns the total dipole moment of the system, sum of negative derivative
    of energy w.r.t. electric field and ionic dipole moment.
    The dipole is pointing from negative to positive charge.

    Examples
    --------
    >>> import torch
    >>> from deepchem.utils.dft_utils import dipole_moment, Mol, HF
    >>> dtype = torch.float64
    >>> moldesc = "O 0 0 0.2156; H 0 1.4749 -0.8625; H 0 -1.4749 -0.8625"  # in Bohr
    >>> efield = torch.zeros(3, dtype=dtype).requires_grad_()  # efield must be specified
    >>> mol = Mol(moldesc=moldesc, basis="3-21G", dtype=dtype, efield=(efield,))
    >>> qc = HF(mol).run()
    >>> dm = dipole_moment(qc)
    >>> dm
    tensor([-1.9132e-17, -1.1380e-15, -9.3935e-01], dtype=torch.float64,
           grad_fn=<AddBackward0>)

    Parameters
    ----------
    qc: BaseQCCalc
        The qc calc object that has been simulated.

    Returns
    -------
    torch.Tensor
        Dipole moment in atomic units with shape (ndim,).

    """
    energy = qc.energy()
    system = qc.get_system()
    efield = system.efield

    assert isinstance(efield, tuple)
    assert len(efield) > 0, "Constant electric field must be provided."
    assert isinstance(efield[0], torch.Tensor)
    assert efield[0].requires_grad, "Electric field needs to be differentiable."

    # Contribution from electron
    ele_dipole = -_jac(energy, efield[0])

    # Contribution from ions
    atompos = system.atompos
    atomzs = system.atomzs.to(atompos.dtype)
    ion_dipole = torch.einsum("ad,a->d", atompos, atomzs)

    return ele_dipole + ion_dipole


def _jac(a: torch.Tensor,
         b: torch.Tensor,
         create_graph: Optional[bool] = None,
         retain_graph: Optional[bool] = True) -> torch.Tensor:
    """Calculates the jacobian of a w.r.t. b.

    Examples
    --------
    >>> import torch
    >>> from deepchem.utils.dft_utils.api.properties import _jac
    >>> a = torch.tensor([2.0], requires_grad=True)
    >>> b = a**4
    >>> grad_b = _jac(b, a, create_graph=True)
    >>> grad_b
    tensor([[32.]], grad_fn=<ViewBackward0>)
    >>> hess_b = _jac(grad_b, a)
    >>> hess_b
    tensor([[[48.]]], grad_fn=<ViewBackward0>)

    Parameters
    ----------
    a: torch.Tensor
        Numerator/Dependent tensor.
    b: torch.Tensor
        Denominator/Independent tensor.
    create_graph: Optional[bool] (default None)
        Needed for higher order derivatives.
    retain_graph: Optional[bool] (default True)
        Allows multiple gradient computations from same graph.

    Returns
    -------
    torch.Tensor
        Jacobian of a w.r.t. b.

    """
    if create_graph is None:
        create_graph = torch.is_grad_enabled()
    assert create_graph is not None

    aflat = a.reshape(-1)
    anumel = a.numel()
    bnumel = b.numel()
    res = torch.empty((anumel, bnumel), dtype=a.dtype, device=a.device)
    for i in range(anumel):
        res[i] = torch.autograd.grad(aflat[i],
                                     b,
                                     create_graph=create_graph,
                                     retain_graph=retain_graph)[0].reshape(-1)
    res = res.reshape((*a.shape, bnumel))
    return res
