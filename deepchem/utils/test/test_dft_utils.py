import pytest
try:
    import torch
except:
    pass
import numpy as np
from typing import Union


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


@pytest.mark.torch
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


@pytest.mark.torch
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


@pytest.mark.torch
def test_base_xc():
    """Test BaseXC."""
    from deepchem.utils.dft_utils import ValGrad, SpinParam
    from deepchem.utils.dft_utils import BaseXC

    class MyXC(BaseXC):

        @property
        def family(self) -> int:
            return 1

        def get_edensityxc(
                self, densinfo: Union[ValGrad,
                                      SpinParam[ValGrad]]) -> torch.Tensor:
            if isinstance(densinfo, ValGrad):
                return densinfo.value.pow(2)
            else:
                return densinfo.u.value.pow(2) + densinfo.d.value.pow(2)

        def get_vxc(
            self, densinfo: Union[ValGrad, SpinParam[ValGrad]]
        ) -> Union[ValGrad, SpinParam[ValGrad]]:
            if isinstance(densinfo, ValGrad):
                return ValGrad(value=2 * densinfo.value)
            else:
                return SpinParam(u=ValGrad(value=2 * densinfo.u.value),
                                 d=ValGrad(value=2 * densinfo.d.value))

    xc = MyXC()
    densinfo_v = ValGrad(value=torch.tensor([1., 2., 3.], requires_grad=True))
    assert torch.allclose(xc.get_edensityxc(densinfo_v),
                          torch.tensor([1., 4., 9.]))
    assert xc.get_vxc(densinfo_v) == ValGrad(value=torch.tensor([2., 4., 6.]))

    densinfo_s = SpinParam(
        u=ValGrad(value=torch.tensor([1., 2., 3.], requires_grad=True)),
        d=ValGrad(value=torch.tensor([4., 5., 6.], requires_grad=True)))
    assert torch.allclose(xc.get_edensityxc(densinfo_s),
                          torch.tensor([17., 29., 45.]))
    assert xc.get_vxc(densinfo_s) == SpinParam(
        u=ValGrad(value=torch.tensor([2., 4., 6.])),
        d=ValGrad(value=torch.tensor([8., 10., 12.])))


@pytest.mark.torch
def test_add_base_xc():
    from deepchem.utils.dft_utils import ValGrad, SpinParam
    from deepchem.utils.dft_utils import BaseXC, AddBaseXC

    class MyXC(BaseXC):

        @property
        def family(self) -> int:
            return 1

        def get_edensityxc(
                self, densinfo: Union[ValGrad,
                                      SpinParam[ValGrad]]) -> torch.Tensor:
            if isinstance(densinfo, ValGrad):
                return densinfo.value.pow(2)
            else:
                return densinfo.u.value.pow(2) + densinfo.d.value.pow(2)

        def get_vxc(
            self, densinfo: Union[ValGrad, SpinParam[ValGrad]]
        ) -> Union[ValGrad, SpinParam[ValGrad]]:
            if isinstance(densinfo, ValGrad):
                return ValGrad(value=2 * densinfo.value)
            else:
                return SpinParam(u=ValGrad(value=2 * densinfo.u.value),
                                 d=ValGrad(value=2 * densinfo.d.value))

    xc = MyXC()
    densinfo_v = ValGrad(value=torch.tensor([1., 2., 3.], requires_grad=True))
    assert torch.allclose(xc.get_edensityxc(densinfo_v),
                          torch.tensor([1., 4., 9.]))
    assert xc.get_vxc(densinfo_v) == ValGrad(value=torch.tensor([2., 4., 6.]))

    densinfo = SpinParam(
        u=ValGrad(value=torch.tensor([1., 2., 3.], requires_grad=True)),
        d=ValGrad(value=torch.tensor([4., 5., 6.], requires_grad=True)))
    assert torch.allclose(xc.get_edensityxc(densinfo),
                          torch.tensor([17., 29., 45.]))
    assert xc.get_vxc(densinfo) == SpinParam(
        u=ValGrad(value=torch.tensor([2., 4., 6.])),
        d=ValGrad(value=torch.tensor([8., 10., 12.])))

    xc2 = AddBaseXC(xc, xc)
    assert torch.allclose(xc2.get_edensityxc(densinfo),
                          torch.tensor([34., 58., 90.]))
    assert xc2.get_vxc(densinfo) == SpinParam(
        u=ValGrad(value=torch.tensor([4., 8., 12.])),
        d=ValGrad(value=torch.tensor([16., 20., 24.])))

    xc3 = xc + xc
    assert torch.allclose(xc3.get_edensityxc(densinfo),
                          torch.tensor([34., 58., 90.]))
    assert xc3.get_vxc(densinfo) == SpinParam(
        u=ValGrad(value=torch.tensor([4., 8., 12.])),
        d=ValGrad(value=torch.tensor([16., 20., 24.])))


@pytest.mark.torch
def test_mul_base_xc():
    from deepchem.utils.dft_utils import ValGrad, SpinParam
    from deepchem.utils.dft_utils import BaseXC, MulBaseXC
    class MyXC(BaseXC):
        @property
        def family(self) -> int:
            return 1
        def get_edensityxc(self, densinfo: Union[ValGrad, SpinParam[ValGrad]]) -> torch.Tensor:
            if isinstance(densinfo, ValGrad):
                return densinfo.value.pow(2)
            else:
                return densinfo.u.value.pow(2) + densinfo.d.value.pow(2)
        def get_vxc(self, densinfo: Union[ValGrad, SpinParam[ValGrad]]) -> Union[ValGrad, SpinParam[ValGrad]]:
            if isinstance(densinfo, ValGrad):
                return ValGrad(value=2*densinfo.value)
            else:
                return SpinParam(u=ValGrad(value=2*densinfo.u.value),
                                 d=ValGrad(value=2*densinfo.d.value))
    xc = MyXC()
    densinfo_v = ValGrad(value=torch.tensor([1., 2., 3.], requires_grad=True))
    assert torch.allclose(xc.get_edensityxc(densinfo_v), torch.tensor([1., 4., 9.]))
    assert xc.get_vxc(densinfo_v) == ValGrad(value=torch.tensor([2., 4., 6.]))

    densinfo_s = SpinParam(u=ValGrad(value=torch.tensor([1., 2., 3.], requires_grad=True)),
                           d=ValGrad(value=torch.tensor([4., 5., 6.], requires_grad=True)))
    assert torch.allclose(xc.get_edensityxc(densinfo_s), torch.tensor([17., 29., 45.]))
    assert xc.get_vxc(densinfo_s) == SpinParam(u=ValGrad(value=torch.tensor([2., 4., 6.])),
                                               d=ValGrad(value=torch.tensor([ 8., 10., 12.])))

    xc2 = MulBaseXC(xc, 2.)
    assert torch.allclose(xc2.get_edensityxc(densinfo_s), torch.tensor([34., 58., 90.]))
    assert xc2.get_vxc(densinfo_s) == SpinParam(u=ValGrad(value=torch.tensor([ 4.,  8., 12.])),
                                                d=ValGrad(value=torch.tensor([16., 20., 24.])))

    xc3 = xc * 2.
    assert torch.allclose(xc3.get_edensityxc(densinfo_s),
                          torch.tensor([34., 58., 90.]))
    assert xc3.get_vxc(densinfo_s) == SpinParam(
        u=ValGrad(value=torch.tensor([4., 8., 12.])),
        d=ValGrad(value=torch.tensor([16., 20., 24.])))


@pytest.mark.torch
def test_parse_moldesc():
    """Tests Moldesc Parser."""
    from deepchem.utils.dft_utils import parse_moldesc
    system = {
        'type': 'mol',
        'kwargs': {
            'moldesc': 'H 0.86625 0 0; F -0.86625 0 0',
            'basis': '6-311++G(3df,3pd)'
        }
    }
    atomzs, atomposs = parse_moldesc(system["kwargs"]["moldesc"])
    assert torch.allclose(atomzs, torch.tensor([1., 9.], dtype=torch.float64))
    assert torch.allclose(
        atomposs,
        torch.tensor(
            [[0.86625, 0.00000, 0.00000], [-0.86625, 0.00000, 0.00000]],
            dtype=torch.float64))
    system2 = (['H', 'F'], torch.tensor([[0.86625, 0, 0], [-0.86625, 0, 0]]))
    atomzs2, atomposs2 = parse_moldesc(system2)
    assert torch.allclose(atomzs2, torch.tensor([1., 9.], dtype=torch.float64))
    assert torch.allclose(
        atomposs2,
        torch.tensor(
            [[0.86625, 0.00000, 0.00000], [-0.86625, 0.00000, 0.00000]],
            dtype=torch.float64))


@pytest.mark.torch
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


@pytest.mark.torch
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
