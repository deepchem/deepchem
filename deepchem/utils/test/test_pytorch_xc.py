import torch

from deepchem.utils.dft_utils import ValGrad, SpinParam
from deepchem.utils.dft_utils.xc.pytorch_xc import (
    PyTorchLDA,
    PyTorchGGA,
    PyTorchMGGA,
)

# LDA

def test_pytorch_lda_x_family():
    xc = PyTorchLDA("lda_x")
    assert xc.family == 1


def test_pytorch_lda_x_unpolarized_shape():
    rho = torch.tensor([0.1, 0.2, 0.5], dtype=torch.float64)
    densinfo = ValGrad(value=rho)
    xc = PyTorchLDA("lda_x")

    edens = xc.get_edensityxc(densinfo)

    assert edens.shape == rho.shape
    assert torch.isfinite(edens).all()


def test_pytorch_lda_x_unpolarized_values():
    rho = torch.tensor([0.1, 0.2, 0.5], dtype=torch.float64)
    densinfo = ValGrad(value=rho)
    xc = PyTorchLDA("lda_x")

    edens = xc.get_edensityxc(densinfo)

    c_x = -0.75 * (3.0 / torch.pi)**(1.0 / 3.0)
    expected = c_x * rho.pow(4.0 / 3.0)

    assert torch.allclose(edens, expected, atol=1e-12, rtol=1e-12)


def test_pytorch_lda_x_spinpolarized_shape():
    rho_u = torch.tensor([0.1, 0.3, 0.4], dtype=torch.float64)
    rho_d = torch.tensor([0.2, 0.1, 0.4], dtype=torch.float64)

    densinfo = SpinParam(
        u=ValGrad(value=rho_u),
        d=ValGrad(value=rho_d),
    )
    xc = PyTorchLDA("lda_x")

    edens = xc.get_edensityxc(densinfo)

    assert edens.shape == rho_u.shape
    assert torch.isfinite(edens).all()


def test_pytorch_lda_x_autograd_unpolarized():
    rho = torch.tensor([0.1, 0.2, 0.5], dtype=torch.float64, requires_grad=True)
    densinfo = ValGrad(value=rho)
    xc = PyTorchLDA("lda_x")

    edens = xc.get_edensityxc(densinfo)
    loss = edens.sum()
    loss.backward()

    assert rho.grad is not None
    assert torch.isfinite(rho.grad).all()


def test_pytorch_lda_x_small_density_stable():
    rho = torch.tensor([0.0, 1e-30, 1e-20], dtype=torch.float64)
    densinfo = ValGrad(value=rho)
    xc = PyTorchLDA("lda_x")

    edens = xc.get_edensityxc(densinfo)

    assert torch.isfinite(edens).all()


def test_pytorch_lda_get_vxc_unpolarized():
    rho = torch.tensor([0.1, 0.2, 0.5], dtype=torch.float64)
    densinfo = ValGrad(value=rho)
    xc = PyTorchLDA("lda_x")

    vxc = xc.get_vxc(densinfo)

    assert isinstance(vxc, ValGrad)
    assert vxc.value.shape == rho.shape
    assert torch.isfinite(vxc.value).all()


def test_pytorch_lda_get_vxc_spinpolarized():
    rho_u = torch.tensor([0.1, 0.3, 0.4], dtype=torch.float64)
    rho_d = torch.tensor([0.2, 0.1, 0.4], dtype=torch.float64)

    densinfo = SpinParam(
        u=ValGrad(value=rho_u),
        d=ValGrad(value=rho_d),
    )
    xc = PyTorchLDA("lda_x")

    vxc = xc.get_vxc(densinfo)

    assert isinstance(vxc, SpinParam)
    assert vxc.u.value.shape == rho_u.shape
    assert vxc.d.value.shape == rho_d.shape
    assert torch.isfinite(vxc.u.value).all()
    assert torch.isfinite(vxc.d.value).all()


def test_pytorch_lda_unknown_name():
    try:
        PyTorchLDA("not_a_real_functional")
        assert False, "Expected ValueError for unknown LDA functional"
    except ValueError:
        pass


# GGA


def test_pytorch_gga_x_pbe_family():
    xc = PyTorchGGA("gga_x_pbe")
    assert xc.family == 2


def test_pytorch_gga_x_pbe_unpolarized_shape():
    rho = torch.tensor([0.1, 0.2, 0.5], dtype=torch.float64)
    grad = torch.tensor(
        [[0.01, 0.02, 0.00],
         [0.03, 0.01, 0.02],
         [0.00, 0.02, 0.01]],
        dtype=torch.float64,
    )

    densinfo = ValGrad(value=rho, grad=grad)
    xc = PyTorchGGA("gga_x_pbe")

    edens = xc.get_edensityxc(densinfo)

    assert edens.shape == rho.shape
    assert torch.isfinite(edens).all()


def test_pytorch_gga_x_pbe_get_vxc():
    rho = torch.tensor([0.1, 0.2, 0.5], dtype=torch.float64)
    grad = torch.tensor(
        [[0.01, 0.02, 0.00],
         [0.03, 0.01, 0.02],
         [0.00, 0.02, 0.01]],
        dtype=torch.float64,
    )
    densinfo = ValGrad(value=rho, grad=grad)
    xc = PyTorchGGA("gga_x_pbe")

    vxc = xc.get_vxc(densinfo)

    assert isinstance(vxc, ValGrad)
    assert vxc.value.shape == rho.shape
    assert torch.isfinite(vxc.value).all()


def test_pytorch_gga_x_pbe_spinpolarized_shape():
    rho_u = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
    rho_d = torch.tensor([0.2, 0.1, 0.4], dtype=torch.float64)

    grad_u = torch.tensor(
        [[0.01, 0.00, 0.02],
         [0.02, 0.01, 0.00],
         [0.00, 0.01, 0.03]],
        dtype=torch.float64,
    )
    grad_d = torch.tensor(
        [[0.00, 0.02, 0.01],
         [0.01, 0.00, 0.01],
         [0.02, 0.01, 0.00]],
        dtype=torch.float64,
    )

    densinfo = SpinParam(
        u=ValGrad(value=rho_u, grad=grad_u),
        d=ValGrad(value=rho_d, grad=grad_d),
    )
    xc = PyTorchGGA("gga_x_pbe")

    edens = xc.get_edensityxc(densinfo)

    assert edens.shape == rho_u.shape
    assert torch.isfinite(edens).all()


def test_pytorch_gga_x_pbe_autograd():
    rho = torch.tensor([0.1, 0.2, 0.5], dtype=torch.float64, requires_grad=True)
    grad = torch.tensor(
        [[0.01, 0.02, 0.00],
         [0.03, 0.01, 0.02],
         [0.00, 0.02, 0.01]],
        dtype=torch.float64,
    )

    densinfo = ValGrad(value=rho, grad=grad)
    xc = PyTorchGGA("gga_x_pbe")

    edens = xc.get_edensityxc(densinfo)
    loss = edens.sum()
    loss.backward()

    assert rho.grad is not None
    assert torch.isfinite(rho.grad).all()


def test_pytorch_gga_x_pbe_small_density_stable():
    rho = torch.tensor([0.0, 1e-30, 1e-12], dtype=torch.float64)
    grad = torch.zeros((3, 3), dtype=torch.float64)

    densinfo = ValGrad(value=rho, grad=grad)
    xc = PyTorchGGA("gga_x_pbe")

    edens = xc.get_edensityxc(densinfo)

    assert torch.isfinite(edens).all()


def test_pytorch_gga_unknown_name():
    try:
        PyTorchGGA("not_a_real_functional")
        assert False, "Expected ValueError for unknown GGA functional"
    except ValueError:
        pass


# MGGA

def test_pytorch_mgga_x_tpss_family():
    xc = PyTorchMGGA("mgga_x_tpss")
    assert xc.family == 4


def test_pytorch_mgga_x_tpss_unpolarized_shape():
    rho = torch.tensor([0.1, 0.2, 0.5], dtype=torch.float64)
    grad = torch.tensor(
        [[0.01, 0.02, 0.00],
         [0.03, 0.01, 0.02],
         [0.00, 0.02, 0.01]],
        dtype=torch.float64,
    )
    kin = torch.tensor([0.05, 0.08, 0.12], dtype=torch.float64)

    densinfo = ValGrad(value=rho, grad=grad, kin=kin)
    xc = PyTorchMGGA("mgga_x_tpss")

    edens = xc.get_edensityxc(densinfo)

    assert edens.shape == rho.shape
    assert torch.isfinite(edens).all()


def test_pytorch_mgga_x_tpss_get_vxc():
    rho = torch.tensor([0.1, 0.2, 0.5], dtype=torch.float64)
    grad = torch.tensor(
        [[0.01, 0.02, 0.00],
         [0.03, 0.01, 0.02],
         [0.00, 0.02, 0.01]],
        dtype=torch.float64,
    )
    kin = torch.tensor([0.05, 0.08, 0.12], dtype=torch.float64)

    densinfo = ValGrad(value=rho, grad=grad, kin=kin)
    xc = PyTorchMGGA("mgga_x_tpss")

    vxc = xc.get_vxc(densinfo)

    assert isinstance(vxc, ValGrad)
    assert vxc.value.shape == rho.shape
    assert torch.isfinite(vxc.value).all()


def test_pytorch_mgga_x_tpss_spinpolarized_shape():
    rho_u = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
    rho_d = torch.tensor([0.2, 0.1, 0.4], dtype=torch.float64)

    grad_u = torch.tensor(
        [[0.01, 0.00, 0.02],
         [0.02, 0.01, 0.00],
         [0.00, 0.01, 0.03]],
        dtype=torch.float64,
    )
    grad_d = torch.tensor(
        [[0.00, 0.02, 0.01],
         [0.01, 0.00, 0.01],
         [0.02, 0.01, 0.00]],
        dtype=torch.float64,
    )

    kin_u = torch.tensor([0.04, 0.06, 0.09], dtype=torch.float64)
    kin_d = torch.tensor([0.05, 0.03, 0.08], dtype=torch.float64)

    densinfo = SpinParam(
        u=ValGrad(value=rho_u, grad=grad_u, kin=kin_u),
        d=ValGrad(value=rho_d, grad=grad_d, kin=kin_d),
    )
    xc = PyTorchMGGA("mgga_x_tpss")

    edens = xc.get_edensityxc(densinfo)

    assert edens.shape == rho_u.shape
    assert torch.isfinite(edens).all()


def test_pytorch_mgga_x_tpss_autograd():
    rho = torch.tensor([0.1, 0.2, 0.5], dtype=torch.float64, requires_grad=True)
    grad = torch.tensor(
        [[0.01, 0.02, 0.00],
         [0.03, 0.01, 0.02],
         [0.00, 0.02, 0.01]],
        dtype=torch.float64,
    )
    kin = torch.tensor([0.05, 0.08, 0.12], dtype=torch.float64)

    densinfo = ValGrad(value=rho, grad=grad, kin=kin)
    xc = PyTorchMGGA("mgga_x_tpss")

    edens = xc.get_edensityxc(densinfo)
    loss = edens.sum()
    loss.backward()

    assert rho.grad is not None
    assert torch.isfinite(rho.grad).all()


def test_pytorch_mgga_x_tpss_small_density_stable():
    rho = torch.tensor([0.0, 1e-30, 1e-12], dtype=torch.float64)
    grad = torch.zeros((3, 3), dtype=torch.float64)
    kin = torch.zeros(3, dtype=torch.float64)

    densinfo = ValGrad(value=rho, grad=grad, kin=kin)
    xc = PyTorchMGGA("mgga_x_tpss")

    edens = xc.get_edensityxc(densinfo)

    assert torch.isfinite(edens).all()


def test_pytorch_mgga_unknown_name():
    try:
        PyTorchMGGA("not_a_real_functional")
        assert False, "Expected ValueError for unknown MGGA functional"
    except ValueError:
        pass