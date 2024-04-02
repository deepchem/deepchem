import pytest
try:
    import torch
except:
    pass
import numpy as np
from typing import Union


@pytest.mark.torch
def test_base_orb_params():
    from deepchem.utils.dft_utils import BaseOrbParams

    class MyOrbParams(BaseOrbParams):

        @staticmethod
        def params2orb(params, coeffs, with_penalty):
            return params, coeffs

        @staticmethod
        def orb2params(orb):
            return orb, torch.tensor([0], dtype=orb.dtype, device=orb.device)

    params = torch.randn(3, 4, 5)
    coeffs = torch.randn(3, 4, 5)
    with_penalty = 0.1
    orb, penalty = MyOrbParams.params2orb(params, coeffs, with_penalty)
    params2, coeffs2 = MyOrbParams.orb2params(orb)
    assert torch.allclose(params, params2)


@pytest.mark.torch
def test_qr_orb_params():
    from deepchem.utils.dft_utils import QROrbParams
    params = torch.randn(3, 3)
    coeffs = torch.randn(4, 3)
    with_penalty = 0.1
    orb, penalty = QROrbParams.params2orb(params, coeffs, with_penalty)
    params2, coeffs2 = QROrbParams.orb2params(orb)
    assert torch.allclose(orb, params2)


@pytest.mark.torch
def test_mat_exp_orb_params():
    from deepchem.utils.dft_utils import MatExpOrbParams
    params = torch.randn(3, 3)
    coeffs = torch.randn(4, 3)
    orb = MatExpOrbParams.params2orb(params, coeffs)[0]
    params2, coeffs2 = MatExpOrbParams.orb2params(orb)
    assert coeffs2.shape == orb.shape


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
def test_addbase_xc():
    """Test AddBaseXC."""
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

    xc1 = MyXC()
    xc = xc1 + xc1
    densinfo_v = ValGrad(value=torch.tensor([1., 2., 3.], requires_grad=True))
    assert torch.allclose(xc.get_edensityxc(densinfo_v),
                          torch.tensor([2., 8., 18.]))
    assert xc.get_vxc(densinfo_v) == ValGrad(value=torch.tensor([4., 8., 12.]))

    densinfo_s = SpinParam(
        u=ValGrad(value=torch.tensor([1., 2., 3.], requires_grad=True)),
        d=ValGrad(value=torch.tensor([4., 5., 6.], requires_grad=True)))
    assert torch.allclose(xc.get_edensityxc(densinfo_s),
                          torch.tensor([34., 58., 90.]))
    assert xc.get_vxc(densinfo_s) == SpinParam(
        u=ValGrad(value=torch.tensor([4., 8., 12.])),
        d=ValGrad(value=torch.tensor([16., 20., 24.])))

    xc2 = AddBaseXC(xc1, xc1)
    densinfo_v = ValGrad(value=torch.tensor([1., 2., 3.], requires_grad=True))
    assert torch.allclose(xc2.get_edensityxc(densinfo_v),
                          torch.tensor([2., 8., 18.]))
    assert xc2.get_vxc(densinfo_v) == ValGrad(value=torch.tensor([4., 8., 12.]))

    densinfo_s = SpinParam(
        u=ValGrad(value=torch.tensor([1., 2., 3.], requires_grad=True)),
        d=ValGrad(value=torch.tensor([4., 5., 6.], requires_grad=True)))
    assert torch.allclose(xc2.get_edensityxc(densinfo_s),
                          torch.tensor([34., 58., 90.]))
    assert xc2.get_vxc(densinfo_s) == SpinParam(
        u=ValGrad(value=torch.tensor([4., 8., 12.])),
        d=ValGrad(value=torch.tensor([16., 20., 24.])))


@pytest.mark.torch
def test_mulbase_xc():
    """Test AddBaseXC."""
    from deepchem.utils.dft_utils import ValGrad, SpinParam
    from deepchem.utils.dft_utils import BaseXC, MulBaseXC

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

    xc1 = MyXC()
    xc = xc1 * 2
    densinfo_v = ValGrad(value=torch.tensor([1., 2., 3.], requires_grad=True))
    assert torch.allclose(xc.get_edensityxc(densinfo_v),
                          torch.tensor([2., 8., 18.]))
    assert xc.get_vxc(densinfo_v) == ValGrad(value=torch.tensor([4., 8., 12.]))

    densinfo_s = SpinParam(
        u=ValGrad(value=torch.tensor([1., 2., 3.], requires_grad=True)),
        d=ValGrad(value=torch.tensor([4., 5., 6.], requires_grad=True)))
    assert torch.allclose(xc.get_edensityxc(densinfo_s),
                          torch.tensor([34., 58., 90.]))
    assert xc.get_vxc(densinfo_s) == SpinParam(
        u=ValGrad(value=torch.tensor([4., 8., 12.])),
        d=ValGrad(value=torch.tensor([16., 20., 24.])))

    xc2 = MulBaseXC(xc1, 2)
    densinfo_v = ValGrad(value=torch.tensor([1., 2., 3.], requires_grad=True))
    assert torch.allclose(xc2.get_edensityxc(densinfo_v),
                          torch.tensor([2., 8., 18.]))
    assert xc2.get_vxc(densinfo_v) == ValGrad(value=torch.tensor([4., 8., 12.]))

    densinfo_s = SpinParam(
        u=ValGrad(value=torch.tensor([1., 2., 3.], requires_grad=True)),
        d=ValGrad(value=torch.tensor([4., 5., 6.], requires_grad=True)))
    assert torch.allclose(xc2.get_edensityxc(densinfo_s),
                          torch.tensor([34., 58., 90.]))
    assert xc2.get_vxc(densinfo_s) == SpinParam(
        u=ValGrad(value=torch.tensor([4., 8., 12.])),
        d=ValGrad(value=torch.tensor([16., 20., 24.])))


@pytest.mark.torch
def test_base_grid():
    import torch
    from deepchem.utils.dft_utils import BaseGrid

    class Grid(BaseGrid):

        def __init__(self):
            super(Grid, self).__init__()
            self.ngrid = 10
            self.ndim = 3
            self.dvolume = torch.ones(self.ngrid)
            self.rgrid = torch.ones((self.ngrid, self.ndim))

        def get_dvolume(self):
            return self.dvolume

        def get_rgrid(self):
            return self.rgrid

    grid = Grid()
    assert torch.allclose(grid.get_dvolume(), torch.ones(10))
    assert torch.allclose(grid.get_rgrid(), torch.ones((10, 3)))


@pytest.mark.torch
def test_base_df():
    """Test BaseDF. Checks that it doesn't raise errors."""
    from deepchem.utils.dft_utils import BaseDF

    class MyDF(BaseDF):

        def __init__(self):
            super(MyDF, self).__init__()

        def get_j2c(self):
            return torch.ones((3, 3))

        def get_j3c(self):
            return torch.ones((3, 3, 3))

    df = MyDF()
    assert torch.allclose(df.get_j2c(), torch.ones((3, 3)))
    assert torch.allclose(df.get_j3c(), torch.ones((3, 3, 3)))


@pytest.mark.torch
def test_base_hamilton():
    """Test BaseHamilton. Checks that it doesn't raise errors."""
    from deepchem.utils.dft_utils import BaseHamilton

    class MyHamilton(BaseHamilton):

        def __init__(self):
            self._nao = 2
            self._kpts = torch.tensor([[0.0, 0.0, 0.0]])
            self._df = None

        @property
        def nao(self):
            return self._nao

        @property
        def kpts(self):
            return self._kpts

        @property
        def df(self):
            return self._df

        def build(self):
            return self

        def get_nuclattr(self):
            return torch.ones((1, 1, self.nao, self.nao))

    ham = MyHamilton()
    hamilton = ham.build()
    assert torch.allclose(hamilton.get_nuclattr(),
                          torch.tensor([[[[1., 1.], [1., 1.]]]]))


@pytest.mark.torch
def test_cgto_basis():
    from deepchem.utils.dft_utils import CGTOBasis
    alphas = torch.ones(1)
    coeffs = torch.ones(1)
    cgto = CGTOBasis(angmom=0, alphas=alphas, coeffs=coeffs)
    cgto.wfnormalize_()

    if cgto.normalized is True:
        assert True
    else:
        assert False


@pytest.mark.torch
def test_atom_cgto_basis():
    from deepchem.utils.dft_utils import AtomCGTOBasis, CGTOBasis
    alphas = torch.ones(1)
    coeffs = torch.ones(1)
    cgto = CGTOBasis(angmom=0, alphas=alphas, coeffs=coeffs)
    atomcgto = AtomCGTOBasis(atomz=1, bases=[cgto], pos=[0.0, 0.0, 0.0])
    assert atomcgto.bases[0] == cgto


@pytest.mark.torch
def test_base_system():
    """Test BaseSystem. Checks that it doesn't raise errors."""
    from deepchem.utils.dft_utils import BaseSystem, BaseHamilton, BaseGrid

    class MySystem(BaseSystem):

        def __init__(self):
            self.hamiltonian = BaseHamilton()
            self.grid = BaseGrid()

        def get_hamiltonian(self):
            return self.hamiltonian

        def get_grid(self):
            return self.grid

        def requires_grid(self):
            return True

    system = MySystem()
    assert system.requires_grid()


@pytest.mark.torch
def test_radial_grid():
    from deepchem.utils.dft_utils import RadialGrid
    grid = RadialGrid(4, grid_integrator="chebyshev", grid_transform="logm3")
    assert grid.get_rgrid().shape == torch.Size([4, 1])
    assert grid.get_dvolume().shape == torch.Size([4])


@pytest.mark.torch
def test_get_xw_integration():
    from deepchem.utils.dft_utils import get_xw_integration
    x, w = get_xw_integration(4, "chebyshev")
    assert x.shape == (4,)
    assert w.shape == torch.Size([4])


@pytest.mark.torch
def test_sliced_radial_grid():
    from deepchem.utils.dft_utils import RadialGrid, SlicedRadialGrid
    grid = RadialGrid(4)
    sliced_grid = SlicedRadialGrid(grid, 2)
    assert sliced_grid.get_rgrid().shape == torch.Size([1])


@pytest.mark.torch
def test_de2_transform():
    from deepchem.utils.dft_utils import DE2Transformation
    x = torch.linspace(-1, 1, 100)
    r = DE2Transformation().x2r(x)
    assert r.shape == torch.Size([100])
    drdx = DE2Transformation().get_drdx(x)
    assert drdx.shape == torch.Size([100])


@pytest.mark.torch
def test_logm3_transform():
    from deepchem.utils.dft_utils import LogM3Transformation
    x = torch.linspace(-1, 1, 100)
    r = LogM3Transformation().x2r(x)
    assert r.shape == torch.Size([100])
    drdx = LogM3Transformation().get_drdx(x)
    assert drdx.shape == torch.Size([100])


@pytest.mark.torch
def test_treutlerm4_transform():
    from deepchem.utils.dft_utils import TreutlerM4Transformation
    x = torch.linspace(-1, 1, 100)
    r = TreutlerM4Transformation().x2r(x)
    assert r.shape == torch.Size([100])
    drdx = TreutlerM4Transformation().get_drdx(x)
    assert drdx.shape == torch.Size([100])


@pytest.mark.torch
def test_get_grid_transform():
    from deepchem.utils.dft_utils import get_grid_transform
    transform = get_grid_transform("logm3")
    transform.x2r(torch.tensor([0.5])) == torch.tensor([2.])
