from deepchem.utils.dft_utils.api.getxc import get_xc
from deepchem.utils.dft_utils.xc.pytorch_xc import (
    PyTorchLDA,
    PyTorchGGA,
    PyTorchMGGA,
)


def test_get_xc_lda():
    xc = get_xc("lda_x")
    assert isinstance(xc, PyTorchLDA)


def test_get_xc_gga():
    xc = get_xc("gga_x_pbe")
    assert isinstance(xc, PyTorchGGA)


def test_get_xc_mgga():
    xc = get_xc("mgga_x_tpss")
    assert isinstance(xc, PyTorchMGGA)