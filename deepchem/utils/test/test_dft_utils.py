import pytest
try:
    import torch
except:
    print("torch not available")
try:
    import pylibxc
except:
    print("pylibxc not available")

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


@pytest.mark.torch
def test_SCF_QCCalc():
    from deepchem.utils.dft_utils import SCF_QCCalc, BaseSCFEngine, SpinParam

    # Define the engine
    class engine(BaseSCFEngine):

        def polarised():
            return False

        def dm2energy(
                self, dm: Union[torch.Tensor,
                                SpinParam[torch.Tensor]]) -> torch.Tensor:
            if isinstance(dm, SpinParam):
                return dm.u + dm.d * 1.1
            return dm * 1.1

    myEngine = engine()
    a = SCF_QCCalc(myEngine)
    assert torch.allclose(a.dm2energy(torch.tensor([1.1])),
                          torch.tensor([1.2100]))


@pytest.mark.torch
def test_BaseSCFEngine():
    from deepchem.utils.dft_utils import BaseSCFEngine, SpinParam

    class engine(BaseSCFEngine):

        def polarised():
            return False

        def dm2energy(
                self, dm: Union[torch.Tensor,
                                SpinParam[torch.Tensor]]) -> torch.Tensor:
            if isinstance(dm, SpinParam):
                return dm.u + dm.d * 1.1
            return dm * 1.1

    myEngine = engine()
    assert myEngine.dm2energy(torch.tensor(1.2)) == torch.tensor(1.32)


@pytest.mark.torch
def test_hf_engine():
    """Tests HFEngine and methods of its parent class BaseSCFEngine."""
    from deepchem.utils.dft_utils import (BaseHamilton, BaseSystem, BaseGrid,
                                          SpinParam, HFEngine)
    from deepchem.utils.differentiation_utils import LinearOperator

    class MyLinOp(LinearOperator):

        def __init__(self, shape):
            super(MyLinOp, self).__init__(shape)
            self.param = torch.rand(shape)

        def _getparamnames(self, prefix=""):
            return [prefix + "param"]

        def _mv(self, x):
            return torch.matmul(self.param, x)

        def _rmv(self, x):
            return torch.matmul(self.param.transpose(-2, -1).conj(), x)

        def _mm(self, x):
            return torch.matmul(self.param, x)

        def _rmm(self, x):
            return torch.matmul(self.param.transpose(-2, -1).conj(), x)

        def _fullmatrix(self):
            return self.param

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

        def get_e_elrep(self, dmtot):
            return 2 * dmtot

        def get_e_exchange(self, dm):
            return 3 * dm

        def get_e_hcore(self, dm):
            return 4.0 * dm

        def get_elrep(self, dmtot):
            return MyLinOp((self.nao + 1, self.nao + 1))

        def get_exchange(self, dm):
            return MyLinOp((self.nao + 1, self.nao + 1))

        def get_kinnucl(self):
            linop = MyLinOp((self.nao + 1, self.nao + 1))
            return linop

        def ao_orb2dm(self, orb: torch.Tensor,
                      orb_weight: torch.Tensor) -> torch.Tensor:
            return orb * orb_weight

    ham = MyHamilton()

    class MySystem(BaseSystem):

        def __init__(self):
            self.hamiltonian = ham
            self.grid = BaseGrid()

        def get_hamiltonian(self):
            return self.hamiltonian

        def get_grid(self):
            return self.grid

        def requires_grid(self):
            return True

        def get_orbweight(
            self,
            polarized: bool = False
        ) -> Union[torch.Tensor, SpinParam[torch.Tensor]]:
            return SpinParam(torch.tensor([1.0]), torch.tensor([2.0]))

        def get_nuclei_energy(self):
            return 10.0

    system = MySystem()
    engine = HFEngine(system)
    engine.set_eigen_options(eigen_options={"method": "exacteig"})

    assert engine.dm2energy(torch.tensor([2])) == torch.tensor([28.0])
    assert engine.dm2scp(torch.tensor([2])).shape == torch.Size([3, 3])
    assert engine.scp2dm(torch.rand((2, 2, 2))).u.shape == torch.Size([2, 1])


@pytest.mark.torch
def test_CalcLDALibXCUnpol():
    from deepchem.utils.dft_utils import CalcLDALibXCUnpol
    libxcfcn = pylibxc.LibXCFunctional("lda_x", "unpolarized")
    rho = torch.tensor([0.1, 0.2, 0.3])
    res = CalcLDALibXCUnpol.apply(rho, 0, libxcfcn)[0]
    assert torch.allclose(
        res, torch.tensor([[-0.0343, -0.0864, -0.1483]], dtype=torch.float64),
        0.001)


@pytest.mark.torch
def test_CalcLDALibXCPol():
    from deepchem.utils.dft_utils import CalcLDALibXCPol
    libxcfcn = pylibxc.LibXCFunctional("lda_x", "polarized")
    rho_u = torch.tensor([0.1, 0.2, 0.3])
    rho_d = torch.tensor([0.1, 0.2, 0.3])
    res = CalcLDALibXCPol.apply(rho_u, rho_d, 0, libxcfcn)[0]
    assert torch.allclose(
        res, torch.tensor([[-0.0864, -0.2177, -0.3738]], dtype=torch.float64),
        0.001)


@pytest.mark.torch
def test_CalcGGALibXCUnpol():
    from deepchem.utils.dft_utils import CalcGGALibXCUnpol
    libxcfcn = pylibxc.LibXCFunctional("gga_c_pbe", "unpolarized")
    rho = torch.tensor([0.1, 0.2, 0.3])
    sigma = torch.tensor([0.1, 0.2, 0.3])
    res = CalcGGALibXCUnpol.apply(rho, sigma, 0, libxcfcn)[0]
    assert torch.allclose(
        res, torch.tensor([[-0.0016, -0.0070, -0.0137]], dtype=torch.float64),
        0.0001, 0.0001)


@pytest.mark.torch
def test_CalcGGALibXCPol():
    from deepchem.utils.dft_utils import CalcGGALibXCPol
    libxcfcn = pylibxc.LibXCFunctional("gga_c_pbe", "polarized")
    rho_u = torch.tensor([0.1, 0.2, 0.3])
    rho_d = torch.tensor([0.1, 0.2, 0.3])
    sigma_uu = torch.tensor([0.1, 0.2, 0.3])
    sigma_ud = torch.tensor([0.1, 0.2, 0.3])
    sigma_dd = torch.tensor([0.1, 0.2, 0.3])
    res = CalcGGALibXCPol.apply(rho_u, rho_d, sigma_uu, sigma_ud, sigma_dd, 0,
                                libxcfcn)[0]
    assert torch.allclose(
        res, torch.tensor([[-0.0047, -0.0175, -0.0322]], dtype=torch.float64),
        0.0001, 0.0001)


@pytest.mark.torch
def test_CalcMGGALibXCUnpol():
    from deepchem.utils.dft_utils import CalcMGGALibXCUnpol
    libxcfcn = pylibxc.LibXCFunctional("mgga_c_m06_l", "unpolarized")
    rho = torch.tensor([0.1, 0.2, 0.3])
    sigma = torch.tensor([0.1, 0.2, 0.3])
    lapl = torch.tensor([0.1, 0.2, 0.3])
    kin = torch.tensor([0.1, 0.2, 0.3])
    res = CalcMGGALibXCUnpol.apply(rho, sigma, lapl, kin, 0, libxcfcn)[0]
    assert torch.allclose(
        res, torch.tensor([[-0.0032, -0.0066, -0.0087]], dtype=torch.float64),
        0.0001, 0.0001)


@pytest.mark.torch
def test_CalcMGGALibXCPol():
    from deepchem.utils.dft_utils import CalcMGGALibXCPol
    libxcfcn = pylibxc.LibXCFunctional("mgga_c_m06_l", "polarized")
    rho_u = torch.tensor([0.1, 0.2, 0.3])
    rho_d = torch.tensor([0.1, 0.2, 0.3])
    sigma_uu = torch.tensor([0.1, 0.2, 0.3])
    sigma_ud = torch.tensor([0.1, 0.2, 0.3])
    sigma_dd = torch.tensor([0.1, 0.2, 0.3])
    lapl_u = torch.tensor([0.1, 0.2, 0.3])
    lapl_d = torch.tensor([0.1, 0.2, 0.3])
    kin_u = torch.tensor([0.1, 0.2, 0.3])
    kin_d = torch.tensor([0.1, 0.2, 0.3])
    res = CalcMGGALibXCPol.apply(rho_u, rho_d, sigma_uu, sigma_ud, sigma_dd,
                                 lapl_u, lapl_d, kin_u, kin_d, 0, libxcfcn)[0]
    assert torch.allclose(
        res, torch.tensor([[-0.0065, -0.0115, -0.0162]], dtype=torch.float64),
        0.0001, 0.0001)


@pytest.mark.torch
def test_get_libxc_res():
    from deepchem.utils.dft_utils.xc.libxc_wrapper import _get_libxc_res
    libxcfcn = pylibxc.LibXCFunctional("lda_x", "unpolarized")
    rho = torch.tensor([0.1, 0.2, 0.3])
    res = _get_libxc_res({"rho": rho}, 0, libxcfcn, 2, False)
    assert torch.allclose(
        res[0], torch.tensor([[-0.0343, -0.0864, -0.1483]],
                             dtype=torch.float64), 0.0001, 0.0001)


@pytest.mark.torch
def test_pack_input():
    from deepchem.utils.dft_utils.xc.libxc_wrapper import _pack_input
    rho = torch.tensor([[1, 2], [3, 4]])
    sigma = torch.tensor([[1, 2], [3, 4]])
    assert np.allclose(_pack_input(rho, sigma),
                       np.array([[[1, 1], [3, 3]], [[2, 2], [4, 4]]]))


@pytest.mark.torch
def test_unpack_input():
    from deepchem.utils.dft_utils.xc.libxc_wrapper import _unpack_input
    inp = np.array([[1, 3], [2, 4]])
    assert np.allclose(tuple(_unpack_input(inp))[0], np.array([1, 2]))
    assert np.allclose(tuple(_unpack_input(inp))[1], np.array([3, 4]))


@pytest.mark.torch
def test_get_dos():
    from deepchem.utils.dft_utils.xc.libxc_wrapper import _get_dos
    assert _get_dos(0) == (True, False, False, False, False)


@pytest.mark.torch
def test_extract_returns():
    from deepchem.utils.dft_utils.xc.libxc_wrapper import _extract_returns
    ret = {"zk": np.array([1, 2, 3])}
    assert torch.allclose(
        _extract_returns(ret, 0, 1)[0], torch.tensor([1, 2, 3]))


@pytest.mark.torch
def test_get_grad_inps():
    from deepchem.utils.dft_utils.xc.libxc_wrapper import _get_grad_inps
    grad_res = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]
    inps = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]
    derivs = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]
    needs_input_grad = [True, True]
    deriv_idxs = [[0], [1]]
    result = _get_grad_inps(grad_res, inps, derivs, needs_input_grad,
                            deriv_idxs)
    assert torch.allclose(result[0], torch.tensor([1, 4, 9]))
    assert torch.allclose(result[1], torch.tensor([4, 10, 18]))


@pytest.mark.torch
def test_LibXCLDA():
    from deepchem.utils.dft_utils import ValGrad, LibXCLDA
    # create a LDA wrapper for libxc
    lda = LibXCLDA("lda_x")
    # create a density information
    densinfo = ValGrad(value=torch.rand(2, 3, 4), grad=torch.rand(2, 3, 4, 3))
    # get the exchange-correlation potential
    potinfo = lda.get_vxc(densinfo)
    assert potinfo.value.shape == torch.Size([2, 3, 4])
    edens = lda.get_edensityxc(densinfo)
    assert edens.shape == torch.Size([2, 3, 4])


@pytest.mark.torch
def test_LibXCGGA():
    from deepchem.utils.dft_utils import ValGrad, LibXCGGA
    # create a GGA wrapper for libxc
    gga = LibXCGGA("gga_c_pbe")
    # create a density information
    n = 2
    rho_u = torch.rand((n,), dtype=torch.float64).requires_grad_()
    grad_u = torch.rand((3, n), dtype=torch.float64).requires_grad_()
    densinfo = ValGrad(value=rho_u, grad=grad_u)
    # get the exchange-correlation potential
    potinfo = gga.get_vxc(densinfo)
    assert potinfo.value.shape == torch.Size([2])


@pytest.mark.torch
def test_LibXCMGGA():
    from deepchem.utils.dft_utils import ValGrad, LibXCMGGA
    # create a MGGA wrapper for libxc
    mgga = LibXCMGGA("mgga_x_scan")
    # create a density information
    n = 2
    rho_u = torch.rand((n,), dtype=torch.float64).requires_grad_()
    grad_u = torch.rand((3, n), dtype=torch.float64).requires_grad_()
    lapl_u = torch.rand((n,), dtype=torch.float64).requires_grad_()
    kin_u = torch.rand((n,), dtype=torch.float64).requires_grad_()
    densinfo = ValGrad(value=rho_u, grad=grad_u, lapl=lapl_u, kin=kin_u)
    # get the exchange-correlation potential
    potinfo = mgga.get_vxc(densinfo)
    assert potinfo.value.shape == torch.Size([2])


@pytest.mark.torch
def test_prepare_libxc_input():
    from deepchem.utils.dft_utils import ValGrad, SpinParam
    from deepchem.utils.dft_utils.xc.libxc import _prepare_libxc_input
    # create a density information
    n = 2
    rho_u = torch.rand((n,), dtype=torch.float64).requires_grad_()
    grad_u = torch.rand((3, n), dtype=torch.float64).requires_grad_()
    lapl_u = torch.rand((n,), dtype=torch.float64).requires_grad_()
    kin_u = torch.rand((n,), dtype=torch.float64).requires_grad_()
    rho_d = torch.rand((n,), dtype=torch.float64).requires_grad_()
    grad_d = torch.rand((3, n), dtype=torch.float64).requires_grad_()
    lapl_d = torch.rand((n,), dtype=torch.float64).requires_grad_()
    kin_d = torch.rand((n,), dtype=torch.float64).requires_grad_()
    densinfo = SpinParam(u=ValGrad(value=rho_u,
                                   grad=grad_u,
                                   lapl=lapl_u,
                                   kin=kin_u),
                         d=ValGrad(value=rho_d,
                                   grad=grad_d,
                                   lapl=lapl_d,
                                   kin=kin_d))
    # prepare the input for libxc
    inputs = _prepare_libxc_input(densinfo, 4)
    assert len(inputs) == 9


@pytest.mark.torch
def test_postproc_libxc_voutput():
    from deepchem.utils.dft_utils import ValGrad
    from deepchem.utils.dft_utils.xc.libxc import _postproc_libxc_voutput
    # create a density information
    n = 2
    rho_u = torch.rand((n,), dtype=torch.float64).requires_grad_()
    grad_u = torch.rand((3, n), dtype=torch.float64).requires_grad_()
    lapl_u = torch.rand((n,), dtype=torch.float64).requires_grad_()
    kin_u = torch.rand((n,), dtype=torch.float64).requires_grad_()
    densinfo = ValGrad(value=rho_u, grad=grad_u, lapl=lapl_u, kin=kin_u)
    # postprocess the output from libxc
    potinfo = _postproc_libxc_voutput(
        densinfo,
        torch.rand((n,), dtype=torch.float64).requires_grad_())
    assert potinfo.value.shape == torch.Size([2])


@pytest.mark.torch
def test_get_libxc():
    from deepchem.utils.dft_utils import ValGrad, get_libxc
    xc = get_libxc("gga_c_pbe")
    n = 2
    rho_u = torch.rand((n,), dtype=torch.float64).requires_grad_()
    grad_u = torch.rand((3, n), dtype=torch.float64).requires_grad_()
    densinfo = ValGrad(value=rho_u, grad=grad_u)
    # get the exchange-correlation potential
    potinfo = xc.get_vxc(densinfo)
    assert potinfo.value.shape == torch.Size([2])


@pytest.mark.torch
def test_get_xc():
    from deepchem.utils.dft_utils import ValGrad, get_xc
    xc = get_xc("lda_x + gga_c_pbe")
    n = 2
    rho_u = torch.rand((n,), dtype=torch.float64).requires_grad_()
    grad_u = torch.rand((3, n), dtype=torch.float64).requires_grad_()
    densinfo = ValGrad(value=rho_u, grad=grad_u)
    # get the exchange-correlation potential
    potinfo = xc.get_vxc(densinfo)
    assert potinfo.value.shape == torch.Size([2])


@pytest.mark.torch
def test_libcintwrapper():
    from deepchem.utils.dft_utils import AtomCGTOBasis, LibcintWrapper, loadbasis
    dtype = torch.double
    d = 1.0
    pos_requires_grad = True
    pos1 = torch.tensor([0.1 * d, 0.0 * d, 0.2 * d],
                        dtype=dtype,
                        requires_grad=pos_requires_grad)
    pos2 = torch.tensor([0.0 * d, 1.0 * d, -0.4 * d],
                        dtype=dtype,
                        requires_grad=pos_requires_grad)
    pos3 = torch.tensor([0.2 * d, -1.4 * d, -0.9 * d],
                        dtype=dtype,
                        requires_grad=pos_requires_grad)
    poss = [pos1, pos2, pos3]
    atomzs = [1, 1, 1]

    allbases = [
        loadbasis("%d:%s" % (max(atomz, 1), "3-21G"),
                  dtype=dtype,
                  requires_grad=False) for atomz in atomzs
    ]

    atombases = [
        AtomCGTOBasis(atomz=atomzs[i], bases=allbases[i], pos=poss[i]) \
        for i in range(len(allbases))
    ]
    wrap = LibcintWrapper(atombases, True, None)
    assert wrap.ao_idxs() == (0, 6)


@pytest.mark.torch
def test_SubsetLibcintWrapper():
    from deepchem.utils.dft_utils import AtomCGTOBasis, LibcintWrapper, loadbasis
    dtype = torch.double
    d = 1.0
    pos_requires_grad = True
    pos1 = torch.tensor([0.1 * d, 0.0 * d, 0.2 * d],
                        dtype=dtype,
                        requires_grad=pos_requires_grad)
    pos2 = torch.tensor([0.0 * d, 1.0 * d, -0.4 * d],
                        dtype=dtype,
                        requires_grad=pos_requires_grad)
    pos3 = torch.tensor([0.2 * d, -1.4 * d, -0.9 * d],
                        dtype=dtype,
                        requires_grad=pos_requires_grad)
    poss = [pos1, pos2, pos3]
    atomzs = [1, 1, 1]
    allbases = [
        loadbasis("%d:%s" % (max(atomz, 1), "3-21G"),
                  dtype=dtype,
                  requires_grad=False) for atomz in atomzs
    ]
    atombases = [
        AtomCGTOBasis(atomz=atomzs[i], bases=allbases[i], pos=poss[i])
        for i in range(len(allbases))
    ]
    wrap = LibcintWrapper(atombases, True, None)
    subset = wrap[1:3]
    assert subset.ao_idxs() == (1, 3)


@pytest.mark.torch
def test_molintor():
    from deepchem.utils.dft_utils import AtomCGTOBasis, LibcintWrapper, loadbasis, \
        int1e, int2e, int2c2e, int3c2e, overlap, kinetic, nuclattr, elrep, coul2c, coul3c
    dtype = torch.double
    d = 1.0
    pos_requires_grad = True
    pos1 = torch.tensor([0.1 * d, 0.0 * d, 0.2 * d],
                        dtype=dtype,
                        requires_grad=pos_requires_grad)
    pos2 = torch.tensor([0.0 * d, 1.0 * d, -0.4 * d],
                        dtype=dtype,
                        requires_grad=pos_requires_grad)
    pos3 = torch.tensor([0.2 * d, -1.4 * d, -0.9 * d],
                        dtype=dtype,
                        requires_grad=pos_requires_grad)
    poss = [pos1, pos2, pos3]
    atomzs = [1, 1, 1]

    allbases = [
        loadbasis("%d:%s" % (max(atomz, 1), "3-21G"),
                  dtype=dtype,
                  requires_grad=False) for atomz in atomzs
    ]

    atombases = [
        AtomCGTOBasis(atomz=atomzs[i], bases=allbases[i], pos=poss[i]) \
        for i in range(len(allbases))
    ]
    env = LibcintWrapper(atombases, True, None)

    assert int1e("r0", env).shape == torch.Size([3, 6, 6])
    assert int1e("r0r0", env).shape == torch.Size([9, 6, 6])
    assert int1e("r0r0r0", env).shape == torch.Size([27, 6, 6])
    assert int2e("ar12b", env).shape == torch.Size([6, 6, 6, 6])
    assert int2c2e("ipip1", env).shape == torch.Size([3, 3, 6, 6])
    assert int3c2e("ar12", env).shape == torch.Size([6, 6, 6])
    assert overlap(env).shape == torch.Size([6, 6])
    assert kinetic(env).shape == torch.Size([6, 6])
    assert nuclattr(env).shape == torch.Size([6, 6])
    assert elrep(env).shape == torch.Size([6, 6, 6, 6])
    assert coul2c(env).shape == torch.Size([6, 6])
    assert coul3c(env).shape == torch.Size([6, 6, 6])


@pytest.mark.torch
def test_intor_name_manager():
    from deepchem.utils.dft_utils.hamilton.intor.namemgr import IntorNameManager
    mgr = IntorNameManager("int1e", "r0")
    assert mgr.fullname == "int1e_r0"
    assert mgr.get_intgl_name(True) == "int1e_r0_sph"
    assert mgr.get_ft_intgl_name(True) == "GTO_ft_r0_sph"
    assert mgr.get_intgl_symmetry([0, 1, 2, 0]).code == "s1"
    assert mgr.get_intgl_components_shape() == (3,)


@pytest.mark.torch
def test_base_symmetry():
    from deepchem.utils.dft_utils.hamilton.intor.symmetry import BaseSymmetry

    class SemNew(BaseSymmetry):

        def get_reduced_shape(self, orig_shape):
            return orig_shape

        @property
        def code(self) -> str:
            return "sn"

        def reconstruct_array(self, arr, orig_shape):
            return arr

    sym = SemNew()
    assert sym.get_reduced_shape((2, 3, 4)) == (2, 3, 4)
    assert sym.code == 'sn'
    assert sym.reconstruct_array(torch.rand((2, 3, 4)),
                                 (2, 3, 4)).shape == torch.Size([2, 3, 4])


@pytest.mark.torch
def test_s1_symmetry():
    from deepchem.utils.dft_utils.hamilton.intor.symmetry import S1Symmetry
    sym = S1Symmetry()
    assert sym.get_reduced_shape((2, 3, 4)) == (2, 3, 4)
    assert sym.code == 's1'
    assert sym.reconstruct_array(np.random.rand(2, 3, 4),
                                 (2, 3, 4)).shape == torch.Size([2, 3, 4])


@pytest.mark.torch
def test_s4_symmetry():
    from deepchem.utils.dft_utils.hamilton.intor.symmetry import S4Symmetry
    sym = S4Symmetry()
    assert sym.get_reduced_shape((3, 3, 4, 4)) == (6, 10)
    assert sym.code == 's4'
    assert sym.reconstruct_array(np.random.rand(2, 3, 4, 4),
                                 (3, 3, 4, 4)).shape == (3, 3, 4, 4)


@pytest.mark.torch
def test_np2ctypes():
    """Just checks that it doesn't raise errors."""
    from deepchem.utils.dft_utils.hamilton.intor.utils import np2ctypes
    arr = np.random.rand(2, 3, 4)
    np2ctypes(arr)


@pytest.mark.torch
def test_int2ctypes():
    """Just checks that it doesn't raise errors."""
    from deepchem.utils.dft_utils.hamilton.intor.utils import int2ctypes
    arr = 51
    int2ctypes(arr)


@pytest.mark.torch
def test_memoize_method():
    from deepchem.utils import memoize_method

    class A:

        @memoize_method
        def foo(self):
            print("foo")
            return 1

    a = A()
    assert a.foo() == 1


@pytest.mark.torch
def test_load_basis():
    from deepchem.utils.dft_utils import loadbasis
    H = loadbasis("1:3-21G")
    assert H[0].alphas.shape == torch.Size([2])


@pytest.mark.torch
def test_ks_engine():
    """Tests KSEngine and KS Class."""
    from deepchem.utils.dft_utils import (BaseHamilton, BaseSystem, BaseGrid,
                                          SpinParam, KSEngine)
    from deepchem.utils.differentiation_utils import LinearOperator

    class MyLinOp(LinearOperator):

        def __init__(self, shape):
            super(MyLinOp, self).__init__(shape)
            self.param = torch.rand(shape)

        def _getparamnames(self, prefix=""):
            return [prefix + "param"]

        def _mv(self, x):
            return torch.matmul(self.param, x)

        def _rmv(self, x):
            return torch.matmul(self.param.transpose(-2, -1).conj(), x)

        def _mm(self, x):
            return torch.matmul(self.param, x)

        def _rmm(self, x):
            return torch.matmul(self.param.transpose(-2, -1).conj(), x)

        def _fullmatrix(self):
            return self.param

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

        def get_e_elrep(self, dmtot):
            return 2 * dmtot

        def get_e_exchange(self, dm):
            return 3 * dm

        def get_e_hcore(self, dm):
            return 4.0 * dm

        def get_elrep(self, dmtot):
            return MyLinOp((self.nao + 1, self.nao + 1))

        def get_exchange(self, dm):
            return MyLinOp((self.nao + 1, self.nao + 1))

        def get_kinnucl(self):
            linop = MyLinOp((self.nao + 1, self.nao + 1))
            return linop

        def ao_orb2dm(self, orb: torch.Tensor,
                      orb_weight: torch.Tensor) -> torch.Tensor:
            return orb * orb_weight

    ham = MyHamilton()

    class MySystem(BaseSystem):

        def __init__(self):
            self.hamiltonian = ham
            self.grid = BaseGrid()

        def get_hamiltonian(self):
            return self.hamiltonian

        def get_grid(self):
            return self.grid

        def requires_grid(self):
            return True

        def get_orbweight(
            self,
            polarized: bool = False
        ) -> Union[torch.Tensor, SpinParam[torch.Tensor]]:
            return SpinParam(torch.tensor([1.0]), torch.tensor([2.0]))

        def get_nuclei_energy(self):
            return 10.0

    system = MySystem()
    engine = KSEngine(system, None)
    engine.set_eigen_options(eigen_options={"method": "exacteig"})

    assert engine.dm2energy(torch.tensor([2])) == torch.tensor([22.0])
    assert engine.dm2scp(torch.tensor([2])).shape == torch.Size([3, 3])
    assert engine.scp2dm(torch.rand((2, 2, 2))).u.shape == torch.Size([2, 1])


@pytest.mark.torch
def test_read_float():
    from deepchem.utils.dft_utils.api.loadbasis import _read_float
    assert _read_float("1.0D+00") == 1.0


@pytest.mark.torch
def test_get_basis_file():
    from deepchem.utils.dft_utils.api.loadbasis import _get_basis_file
    fname = _get_basis_file("1:3-21G")
    path = fname.split("/")[-1]
    assert path == "01.gaussian94"


@pytest.mark.torch
def test_normalize_basisname():
    from deepchem.utils.dft_utils.api.loadbasis import _normalize_basisname
    assert _normalize_basisname("6-311++G**") == '6-311ppgss'


@pytest.mark.torch
def test_expand_angmoms():
    from deepchem.utils.dft_utils.api.loadbasis import _expand_angmoms
    assert _expand_angmoms("SP", 2) == [0, 1]


@pytest.mark.torch
def test_DensityFitInfo():
    from deepchem.utils.dft_utils.data.datastruct import DensityFitInfo, AtomCGTOBasis, CGTOBasis
    method = "df"
    auxbasis = [
        AtomCGTOBasis(atomz=1,
                      bases=[
                          CGTOBasis(angmom=0,
                                    alphas=torch.ones(1),
                                    coeffs=torch.ones(1))
                      ],
                      pos=[[0.0, 0.0, 0.0]])
    ]
    df = DensityFitInfo(method=method, auxbasis=auxbasis)
    assert df == DensityFitInfo(method='df',
                                auxbasis=[
                                    AtomCGTOBasis(
                                        atomz=1,
                                        bases=[
                                            CGTOBasis(angmom=0,
                                                      alphas=torch.tensor([1.]),
                                                      coeffs=torch.tensor([1.]),
                                                      normalized=False)
                                        ],
                                        pos=torch.tensor([[0., 0., 0.]]))
                                ])


@pytest.mark.torch
def test_is_z_float():
    from deepchem.utils.dft_utils.data.datastruct import is_z_float
    assert is_z_float(0.1)
    assert not is_z_float(1)
    assert is_z_float(torch.tensor(1.0))


@pytest.mark.torch
def test_OrbitalOrthogonalizer():
    from deepchem.utils.dft_utils.hamilton.orbconverter import OrbitalOrthogonalizer
    ovlp = torch.tensor([[1.0, 0.5], [0.5, 1.0]])
    orthozer = OrbitalOrthogonalizer(ovlp)
    assert orthozer.nao() == 2
    mat = torch.tensor([[1.0, 0.5], [0.5, 1.0]])
    assert torch.allclose(orthozer.convert2(mat),
                          torch.tensor([[1.0000, 0.0000], [0.0000, 1.0000]]))


@pytest.mark.torch
def test_LebedevLoader():
    from deepchem.utils.dft_utils.grid.lebedev_grid import LebedevLoader
    grid = LebedevLoader.load(3)
    assert grid.shape == (6, 3)


@pytest.mark.torch
def test_LebedevGrid():
    from deepchem.utils.dft_utils.grid.radial_grid import RadialGrid
    from deepchem.utils.dft_utils.grid.lebedev_grid import LebedevGrid
    grid = RadialGrid(100, grid_integrator="chebyshev", grid_transform="logm3")
    l_grid = LebedevGrid(grid, 3)
    assert l_grid.get_rgrid().shape == torch.Size([600, 3])
    assert grid.get_rgrid().shape == torch.Size([100, 1])


@pytest.mark.torch
def test_BeckeGrid():
    from deepchem.utils.dft_utils import LebedevGrid, BeckeGrid, RadialGrid
    grid = RadialGrid(100, grid_integrator="chebyshev", grid_transform="logm3")
    atomgrid = [LebedevGrid(grid, 3), LebedevGrid(grid, 3)]
    atompos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
                           dtype=torch.float64)
    grid = BeckeGrid(atomgrid, atompos)
    assert grid.get_rgrid().shape == torch.Size([1200, 3])
    assert grid.get_dvolume().shape == torch.Size([1200])


@pytest.mark.torch
def test_PBCBeckeGrid():
    from deepchem.utils.dft_utils import LebedevGrid, PBCBeckeGrid, RadialGrid, Lattice
    grid = RadialGrid(100, grid_integrator="chebyshev", grid_transform="logm3")
    atomgrid = [LebedevGrid(grid, 3), LebedevGrid(grid, 3)]
    atompos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
                           dtype=torch.float64)
    a = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
                     dtype=torch.float64)
    lattice = Lattice(a)
    grid = PBCBeckeGrid(atomgrid, atompos, lattice)
    assert grid.get_rgrid().shape == torch.Size([720, 3])
    assert grid.get_dvolume().shape == torch.Size([720])


@pytest.mark.torch
def test_construct_rgrids():
    from deepchem.utils.dft_utils import LebedevGrid, RadialGrid
    from deepchem.utils.dft_utils.grid.multiatoms_grid import _construct_rgrids
    grid = RadialGrid(100, grid_integrator="chebyshev", grid_transform="logm3")
    atomgrid = [LebedevGrid(grid, 3), LebedevGrid(grid, 3)]
    atompos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
                           dtype=torch.float64)
    allpos_lst, rgrid, dvol_atoms = _construct_rgrids(atomgrid, atompos)
    assert len(allpos_lst) == 2


@pytest.mark.torch
def test_get_atom_weights():
    from deepchem.utils.dft_utils import LebedevGrid, RadialGrid
    from deepchem.utils.dft_utils.grid.multiatoms_grid import _get_atom_weights, _construct_rgrids
    grid = RadialGrid(100, grid_integrator="chebyshev", grid_transform="logm3")
    atomgrid = [LebedevGrid(grid, 3), LebedevGrid(grid, 3)]
    atompos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
                           dtype=torch.float64)
    rgrids, _, _ = _construct_rgrids(atomgrid, atompos)
    w = _get_atom_weights(rgrids, atompos)
    assert w.shape == torch.Size([1200])


@pytest.mark.torch
def test_BaseTruncationRules():
    from deepchem.utils.dft_utils import BaseTruncationRules

    class MyTrunc(BaseTruncationRules):

        def to_truncate(self, atm: int) -> bool:
            return False

    trunc = MyTrunc()
    assert not trunc.to_truncate(1)


@pytest.mark.torch
def test_NoTrunc():
    from deepchem.utils.dft_utils import NoTrunc
    rule = NoTrunc()
    assert not rule.to_truncate(1)


@pytest.mark.torch
def test_DasguptaTrunc():
    from deepchem.utils.dft_utils import DasguptaTrunc
    rule = DasguptaTrunc(75)
    assert rule.to_truncate(1)


@pytest.mark.torch
def test_NWChemTrunc():
    from deepchem.utils.dft_utils import NWChemTrunc
    rule = NWChemTrunc([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5], 13,
                       [3, 5, 7, 9, 13], torch.float64, torch.device('cpu'))
    assert rule.to_truncate(10)


@pytest.mark.torch
def test_get_nr():
    from deepchem.utils.dft_utils.grid.truncation_rules import _get_nr
    assert _get_nr(5, 1) == 5


@pytest.mark.torch
def test_evl():
    from deepchem.utils.dft_utils import AtomCGTOBasis, LibcintWrapper, loadbasis, RadialGrid, evl
    dtype = torch.double
    d = 1.0
    pos_requires_grad = True
    pos1 = torch.tensor([0.1 * d, 0.0 * d, 0.2 * d],
                        dtype=dtype,
                        requires_grad=pos_requires_grad)
    pos2 = torch.tensor([0.0 * d, 1.0 * d, -0.4 * d],
                        dtype=dtype,
                        requires_grad=pos_requires_grad)
    pos3 = torch.tensor([0.2 * d, -1.4 * d, -0.9 * d],
                        dtype=dtype,
                        requires_grad=pos_requires_grad)
    poss = [pos1, pos2, pos3]
    atomzs = [1, 1, 1]
    allbases = [
        loadbasis("%d:%s" % (max(atomz, 1), "3-21G"),
                  dtype=dtype,
                  requires_grad=False) for atomz in atomzs
    ]
    atombases = [
        AtomCGTOBasis(atomz=atomzs[i], bases=allbases[i], pos=poss[i])
        for i in range(len(allbases))
    ]
    wrap = LibcintWrapper(atombases, True, None)
    grid = RadialGrid(100, grid_integrator="chebyshev", grid_transform="logm3")
    grad = evl("", wrap, grid.get_rgrid())
    assert grad.shape == torch.Size([6, 100])


@pytest.mark.torch
def test_pbc_evl():
    from deepchem.utils.dft_utils import AtomCGTOBasis, LibcintWrapper, loadbasis, Lattice, RadialGrid, pbc_evl
    dtype = torch.float64
    d = 1.0
    pos_requires_grad = True
    pos1 = torch.tensor([0.1 * d, 0.0 * d, 0.2 * d],
                        dtype=dtype,
                        requires_grad=pos_requires_grad)
    pos2 = torch.tensor([0.0 * d, 1.0 * d, -0.4 * d],
                        dtype=dtype,
                        requires_grad=pos_requires_grad)
    pos3 = torch.tensor([0.2 * d, -1.4 * d, -0.9 * d],
                        dtype=dtype,
                        requires_grad=pos_requires_grad)
    poss = [pos1, pos2, pos3]
    atomzs = [1, 1, 1]
    allbases = [
        loadbasis("%d:%s" % (max(atomz, 1), "3-21G"),
                  dtype=dtype,
                  requires_grad=False) for atomz in atomzs
    ]
    atombases = [
        AtomCGTOBasis(atomz=atomzs[i], bases=allbases[i], pos=poss[i])
        for i in range(len(allbases))
    ]
    a = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], dtype=dtype)
    lattice = Lattice(a)
    wrap = LibcintWrapper(atombases, True, lattice)
    grid = RadialGrid(100, grid_integrator="chebyshev", grid_transform="logm3")
    grad = pbc_evl("", wrap, grid.get_rgrid())
    assert grad.shape == torch.Size([1, 6, 100])


@pytest.mark.torch
def test_eval_gto():
    from deepchem.utils.dft_utils import AtomCGTOBasis, LibcintWrapper, loadbasis, RadialGrid, eval_gto
    dtype = torch.double
    d = 1.0
    pos_requires_grad = True
    pos1 = torch.tensor([0.1 * d, 0.0 * d, 0.2 * d],
                        dtype=dtype,
                        requires_grad=pos_requires_grad)
    pos2 = torch.tensor([0.0 * d, 1.0 * d, -0.4 * d],
                        dtype=dtype,
                        requires_grad=pos_requires_grad)
    pos3 = torch.tensor([0.2 * d, -1.4 * d, -0.9 * d],
                        dtype=dtype,
                        requires_grad=pos_requires_grad)
    poss = [pos1, pos2, pos3]
    atomzs = [1, 1, 1]
    allbases = [
        loadbasis("%d:%s" % (max(atomz, 1), "3-21G"),
                  dtype=dtype,
                  requires_grad=False) for atomz in atomzs
    ]
    atombases = [
        AtomCGTOBasis(atomz=atomzs[i], bases=allbases[i], pos=poss[i])
        for i in range(len(allbases))
    ]
    wrap = LibcintWrapper(atombases, True, None)
    grid = RadialGrid(100, grid_integrator="chebyshev", grid_transform="logm3")
    grad = eval_gto(wrap, grid.get_rgrid())
    assert grad.shape == torch.Size([6, 100])


@pytest.mark.torch
def test_eval_gradgto():
    from deepchem.utils.dft_utils import AtomCGTOBasis, LibcintWrapper, loadbasis, RadialGrid, eval_gradgto
    dtype = torch.double
    d = 1.0
    pos_requires_grad = True
    pos1 = torch.tensor([0.1 * d, 0.0 * d, 0.2 * d],
                        dtype=dtype,
                        requires_grad=pos_requires_grad)
    pos2 = torch.tensor([0.0 * d, 1.0 * d, -0.4 * d],
                        dtype=dtype,
                        requires_grad=pos_requires_grad)
    pos3 = torch.tensor([0.2 * d, -1.4 * d, -0.9 * d],
                        dtype=dtype,
                        requires_grad=pos_requires_grad)
    poss = [pos1, pos2, pos3]
    atomzs = [1, 1, 1]
    allbases = [
        loadbasis("%d:%s" % (max(atomz, 1), "3-21G"),
                  dtype=dtype,
                  requires_grad=False) for atomz in atomzs
    ]
    atombases = [
        AtomCGTOBasis(atomz=atomzs[i], bases=allbases[i], pos=poss[i])
        for i in range(len(allbases))
    ]
    wrap = LibcintWrapper(atombases, True, None)
    grid = RadialGrid(100, grid_integrator="chebyshev", grid_transform="logm3")
    grad = eval_gradgto(wrap, grid.get_rgrid())
    assert grad.shape == torch.Size([3, 6, 100])


@pytest.mark.torch
def test_eval_laplgto():
    from deepchem.utils.dft_utils import AtomCGTOBasis, LibcintWrapper, loadbasis, RadialGrid, eval_laplgto
    dtype = torch.double
    d = 1.0
    pos_requires_grad = True
    pos1 = torch.tensor([0.1 * d, 0.0 * d, 0.2 * d],
                        dtype=dtype,
                        requires_grad=pos_requires_grad)
    pos2 = torch.tensor([0.0 * d, 1.0 * d, -0.4 * d],
                        dtype=dtype,
                        requires_grad=pos_requires_grad)
    pos3 = torch.tensor([0.2 * d, -1.4 * d, -0.9 * d],
                        dtype=dtype,
                        requires_grad=pos_requires_grad)
    poss = [pos1, pos2, pos3]
    atomzs = [1, 1, 1]
    allbases = [
        loadbasis("%d:%s" % (max(atomz, 1), "3-21G"),
                  dtype=dtype,
                  requires_grad=False) for atomz in atomzs
    ]
    atombases = [
        AtomCGTOBasis(atomz=atomzs[i], bases=allbases[i], pos=poss[i])
        for i in range(len(allbases))
    ]
    wrap = LibcintWrapper(atombases, True, None)
    grid = RadialGrid(100, grid_integrator="chebyshev", grid_transform="logm3")
    grad = eval_laplgto(wrap, grid.get_rgrid())
    assert grad.shape == torch.Size([6, 100])


@pytest.mark.torch
def test_pbc_eval_gto():
    from deepchem.utils.dft_utils import AtomCGTOBasis, LibcintWrapper, loadbasis, Lattice, RadialGrid, pbc_eval_gto
    dtype = torch.float64
    d = 1.0
    pos_requires_grad = True
    pos1 = torch.tensor([0.1 * d, 0.0 * d, 0.2 * d],
                        dtype=dtype,
                        requires_grad=pos_requires_grad)
    pos2 = torch.tensor([0.0 * d, 1.0 * d, -0.4 * d],
                        dtype=dtype,
                        requires_grad=pos_requires_grad)
    pos3 = torch.tensor([0.2 * d, -1.4 * d, -0.9 * d],
                        dtype=dtype,
                        requires_grad=pos_requires_grad)
    poss = [pos1, pos2, pos3]
    atomzs = [1, 1, 1]
    allbases = [
        loadbasis("%d:%s" % (max(atomz, 1), "3-21G"),
                  dtype=dtype,
                  requires_grad=False) for atomz in atomzs
    ]
    atombases = [
        AtomCGTOBasis(atomz=atomzs[i], bases=allbases[i], pos=poss[i])
        for i in range(len(allbases))
    ]
    a = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], dtype=dtype)
    lattice = Lattice(a)
    wrap = LibcintWrapper(atombases, True, lattice)
    grid = RadialGrid(100, grid_integrator="chebyshev", grid_transform="logm3")
    grad = pbc_eval_gto(wrap, grid.get_rgrid())
    assert grad.shape == torch.Size([1, 6, 100])


@pytest.mark.torch
def test_pbc_eval_gradgto():
    from deepchem.utils.dft_utils import AtomCGTOBasis, LibcintWrapper, loadbasis, Lattice, RadialGrid, pbc_eval_gradgto
    dtype = torch.float64
    d = 1.0
    pos_requires_grad = True
    pos1 = torch.tensor([0.1 * d, 0.0 * d, 0.2 * d],
                        dtype=dtype,
                        requires_grad=pos_requires_grad)
    pos2 = torch.tensor([0.0 * d, 1.0 * d, -0.4 * d],
                        dtype=dtype,
                        requires_grad=pos_requires_grad)
    pos3 = torch.tensor([0.2 * d, -1.4 * d, -0.9 * d],
                        dtype=dtype,
                        requires_grad=pos_requires_grad)
    poss = [pos1, pos2, pos3]
    atomzs = [1, 1, 1]
    allbases = [
        loadbasis("%d:%s" % (max(atomz, 1), "3-21G"),
                  dtype=dtype,
                  requires_grad=False) for atomz in atomzs
    ]
    atombases = [
        AtomCGTOBasis(atomz=atomzs[i], bases=allbases[i], pos=poss[i])
        for i in range(len(allbases))
    ]
    a = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], dtype=dtype)
    lattice = Lattice(a)
    wrap = LibcintWrapper(atombases, True, lattice)
    grid = RadialGrid(100, grid_integrator="chebyshev", grid_transform="logm3")
    grad = pbc_eval_gradgto(wrap, grid.get_rgrid())
    assert grad.shape == torch.Size([3, 1, 6, 100])


@pytest.mark.torch
def test_pbc_eval_laplgto():
    from deepchem.utils.dft_utils import AtomCGTOBasis, LibcintWrapper, loadbasis, Lattice, RadialGrid, pbc_eval_laplgto
    dtype = torch.float64
    d = 1.0
    pos_requires_grad = True
    pos1 = torch.tensor([0.1 * d, 0.0 * d, 0.2 * d],
                        dtype=dtype,
                        requires_grad=pos_requires_grad)
    pos2 = torch.tensor([0.0 * d, 1.0 * d, -0.4 * d],
                        dtype=dtype,
                        requires_grad=pos_requires_grad)
    pos3 = torch.tensor([0.2 * d, -1.4 * d, -0.9 * d],
                        dtype=dtype,
                        requires_grad=pos_requires_grad)
    poss = [pos1, pos2, pos3]
    atomzs = [1, 1, 1]
    allbases = [
        loadbasis("%d:%s" % (max(atomz, 1), "3-21G"),
                  dtype=dtype,
                  requires_grad=False) for atomz in atomzs
    ]
    atombases = [
        AtomCGTOBasis(atomz=atomzs[i], bases=allbases[i], pos=poss[i])
        for i in range(len(allbases))
    ]
    a = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], dtype=dtype)
    lattice = Lattice(a)
    wrap = LibcintWrapper(atombases, True, lattice)
    grid = RadialGrid(100, grid_integrator="chebyshev", grid_transform="logm3")
    grad = pbc_eval_laplgto(wrap, grid.get_rgrid())
    assert grad.shape == torch.Size([1, 6, 100])


@pytest.mark.torch
def test_gto_evaluator():
    from deepchem.utils.dft_utils import AtomCGTOBasis, LibcintWrapper, loadbasis, RadialGrid, gto_evaluator
    dtype = torch.double
    d = 1.0
    pos_requires_grad = True
    pos1 = torch.tensor([0.1 * d, 0.0 * d, 0.2 * d],
                        dtype=dtype,
                        requires_grad=pos_requires_grad)
    pos2 = torch.tensor([0.0 * d, 1.0 * d, -0.4 * d],
                        dtype=dtype,
                        requires_grad=pos_requires_grad)
    pos3 = torch.tensor([0.2 * d, -1.4 * d, -0.9 * d],
                        dtype=dtype,
                        requires_grad=pos_requires_grad)
    poss = [pos1, pos2, pos3]
    atomzs = [1, 1, 1]
    allbases = [
        loadbasis("%d:%s" % (max(atomz, 1), "3-21G"),
                  dtype=dtype,
                  requires_grad=False) for atomz in atomzs
    ]
    atombases = [
        AtomCGTOBasis(atomz=atomzs[i], bases=allbases[i], pos=poss[i])
        for i in range(len(allbases))
    ]
    wrap = LibcintWrapper(atombases, True, None)
    grid = RadialGrid(100, grid_integrator="chebyshev", grid_transform="logm3")
    grad = gto_evaluator(wrap, "", grid.get_rgrid(), False)
    assert grad.shape == torch.Size([6, 100])


@pytest.mark.torch
def test_get_evalgto_opname():
    from deepchem.utils.dft_utils.hamilton.intor.gtoeval import _get_evalgto_opname
    assert _get_evalgto_opname("", False) == 'GTOval_cart'


@pytest.mark.torch
def test_get_evalgto_compshape():
    from deepchem.utils.dft_utils.hamilton.intor.gtoeval import _get_evalgto_compshape
    assert _get_evalgto_compshape("ip") == (3,)


@pytest.mark.torch
def test_get_evalgto_derivname():
    from deepchem.utils.dft_utils.hamilton.intor.gtoeval import _get_evalgto_derivname
    assert _get_evalgto_derivname("", "a") == 'rr'


@pytest.mark.torch
def test_PBCIntOption():
    from deepchem.utils.dft_utils import PBCIntOption
    pbc = PBCIntOption()
    assert pbc.get_default() == PBCIntOption(precision=1e-08,
                                             kpt_diff_tol=1e-06)


@pytest.mark.torch
def test_get_default_options():
    from deepchem.utils.dft_utils import get_default_options
    assert get_default_options().precision == 1e-08


@pytest.mark.torch
def test_get_default_kpts():
    from deepchem.utils.dft_utils import get_default_kpts
    assert torch.allclose(
        get_default_kpts(torch.tensor([[1, 1, 1]]), torch.float64, 'cpu'),
        torch.tensor([[1., 1., 1.]], dtype=torch.float64))


@pytest.mark.torch
def test_get_grid():
    from deepchem.utils.dft_utils.grid.factory import get_grid
    grid = get_grid(torch.tensor([1]),
                    torch.tensor([[0, 0, 0]], dtype=torch.float64))
    assert grid.get_rgrid().shape == torch.Size([16710, 3])


@pytest.mark.torch
def test_get_predefined_grid():
    from deepchem.utils.dft_utils import get_predefined_grid
    grid = get_predefined_grid(3, torch.tensor([2]),
                               torch.tensor([[0, 0, 0]], dtype=torch.float64))
    assert grid.get_rgrid().shape == torch.Size([8608, 3])


@pytest.mark.torch
def test_dfmol():
    from deepchem.utils.dft_utils import LibcintWrapper, OrbitalOrthogonalizer, DFMol, DensityFitInfo, AtomCGTOBasis, loadbasis
    dtype = torch.double
    d = 1.0
    pos_requires_grad = True
    pos1 = torch.tensor([0.1 * d, 0.0 * d, 0.2 * d],
                        dtype=dtype,
                        requires_grad=pos_requires_grad)
    pos2 = torch.tensor([0.0 * d, 1.0 * d, -0.4 * d],
                        dtype=dtype,
                        requires_grad=pos_requires_grad)
    pos3 = torch.tensor([0.2 * d, -1.4 * d, -0.9 * d],
                        dtype=dtype,
                        requires_grad=pos_requires_grad)
    poss = [pos1, pos2, pos3]
    atomzs = [1, 1, 1]
    allbases = [
        loadbasis("%d:%s" % (max(atomz, 1), "3-21G"),
                  dtype=dtype,
                  requires_grad=False) for atomz in atomzs
    ]
    atombases = [
        AtomCGTOBasis(atomz=atomzs[i], bases=allbases[i], pos=poss[i])
        for i in range(len(allbases))
    ]
    wrapper = LibcintWrapper(atombases, True, None)
    wrapper.ao_idxs()
    dfinfo = DensityFitInfo("coulomb", atombases)
    orthozer = OrbitalOrthogonalizer(torch.eye(6, dtype=dtype))
    dfmol = DFMol(dfinfo, wrapper, orthozer)
    dfmol.build()
    dm = torch.rand(2, 6, 1)
    elrep = dfmol.get_elrep(dm.to(dtype))
    assert elrep.fullmatrix().shape == torch.Size([6, 6])


@pytest.mark.torch
def test_mol():
    from deepchem.utils.dft_utils.system.mol import Mol
    mol = Mol("H 1 0 0; H -1 0 0", "sto-3g", spin=1)
    mol.setup_grid()
    assert torch.allclose(mol.get_orbweight(),
                          torch.tensor([1.5000, 0.5000], dtype=torch.float64))


@pytest.mark.torch
def test_parse_basis():
    from deepchem.utils.dft_utils.system.mol import _parse_basis
    assert len(_parse_basis(torch.tensor([1, 1]), "sto-3g")) == 2


@pytest.mark.torch
def test_get_nelecs_spin():
    from deepchem.utils.dft_utils.system.mol import _get_nelecs_spin
    assert _get_nelecs_spin(torch.tensor(2), None,
                            0) == (torch.tensor(2), torch.tensor(0), False)


@pytest.mark.torch
def test_get_orb_weights():
    from deepchem.utils.dft_utils.system.mol import _get_orb_weights
    assert _get_orb_weights(
        torch.tensor(2), 1, False, torch.float64,
        torch.device('cpu')) == (torch.tensor([1.], dtype=torch.float64),
                                 torch.tensor([1.], dtype=torch.float64),
                                 torch.tensor([0.], dtype=torch.float64))


@pytest.mark.torch
def test_normalize_efield():
    from deepchem.utils.dft_utils.system.mol import _normalize_efield
    assert torch.allclose(
        _normalize_efield(torch.tensor([1, 2, 3]))[0], torch.tensor([1, 2, 3]))


@pytest.mark.torch
def test_preprocess_efield():
    from deepchem.utils.dft_utils.system.mol import _preprocess_efield
    assert torch.allclose(
        _preprocess_efield((torch.tensor([1, 2, 3]),))[0],
        torch.tensor([1, 2, 3]))
