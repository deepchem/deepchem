import pytest
try:
    import torch
except:
    pass
import numpy as np


@pytest.mark.torch
def test_lattice():
    """Test lattice object.
    Comparing it's output with it's original implementation in dqc.
    """
    from deepchem.utils.dft_utils import Lattice
    a = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    lattice = Lattice(a)

    assert torch.allclose(lattice.lattice_vectors(), a)
    assert torch.allclose(lattice.recip_vectors(),
                          torch.inverse(a.transpose(-2, -1)) * (2 * np.pi))
    assert torch.allclose(lattice.volume(), torch.det(a))

    assert torch.allclose(
        lattice.get_lattice_ls(1.0),
        torch.tensor([[0., 0., -1.], [0., -1., 0.], [-1., 0., 0.], [0., 0., 0.],
                      [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]))
    assert torch.allclose(
        lattice.get_gvgrids(6.0)[0],
        torch.tensor([[0.0000, 0.0000, -6.2832], [0.0000, -6.2832, 0.0000],
                      [-6.2832, 0.0000, 0.0000], [0.0000, 0.0000, 0.0000],
                      [6.2832, 0.0000, 0.0000], [0.0000, 6.2832, 0.0000],
                      [0.0000, 0.0000, 6.2832]]))
    assert torch.allclose(
        lattice.get_gvgrids(6.0)[1],
        torch.tensor([1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000]))
    assert lattice.estimate_ewald_eta(1e-5) == 1.8


def test_spin_param():
    """Test SpinParam object."""
    from deepchem.utils.dft_utils import SpinParam
    dens_u = torch.ones(1)
    dens_d = torch.zeros(1)
    sp = SpinParam(u=dens_u, d=dens_d)

    assert torch.allclose(sp.u, dens_u)
    assert torch.allclose(sp.d, dens_d)
    assert torch.allclose(sp.sum(), torch.tensor([1.]))
    assert torch.allclose(sp.reduce(torch.multiply), torch.tensor([0.]))


def test_val_grad():
    """Test ValGrad object."""
    from deepchem.utils.dft_utils import ValGrad
    dens = torch.ones(1)
    grad = torch.zeros(1)
    lapl = torch.ones(1)
    kin = torch.ones(1)
    vg = ValGrad(value=dens, grad=grad, lapl=lapl, kin=kin)

    assert torch.allclose(vg.value, dens)
    assert torch.allclose(vg.grad, grad)
    assert torch.allclose(vg.lapl, lapl)
    assert torch.allclose(vg.kin, kin)

    vg2 = vg + vg
    assert torch.allclose(vg2.value, torch.tensor([2.]))
    assert torch.allclose(vg2.grad, torch.tensor([0.]))
    assert torch.allclose(vg2.lapl, torch.tensor([2.]))
    assert torch.allclose(vg2.kin, torch.tensor([2.]))

    vg5 = vg * 5
    assert torch.allclose(vg5.value, torch.tensor([5.]))
    assert torch.allclose(vg5.grad, torch.tensor([0.]))
    assert torch.allclose(vg5.lapl, torch.tensor([5.]))
    assert torch.allclose(vg5.kin, torch.tensor([5.]))
