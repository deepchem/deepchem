"""Test DFT utils"""
import pytest
try:
    import torch
    has_torch = True
except:
    has_torch = False


@pytest.mark.torch
def test_val_grad():
    """Test ValGrad data structure"""
    from deepchem.utils.dft_utils.datastruct import ValGrad

    vg1 = ValGrad(torch.tensor([1, 2, 3]))
    assert torch.allclose(vg1.value, torch.tensor([1, 2, 3]))
    vg1.grad = torch.tensor([4, 5, 6])
    assert torch.allclose(vg1.grad, torch.tensor([4, 5, 6]))
    vg2 = ValGrad(torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6]))
    vg3 = vg1 + vg2
    assert torch.allclose(vg3.value, torch.tensor([2, 4, 6]))
    assert torch.allclose(vg3.grad, torch.tensor([8, 10, 12]))


@pytest.mark.torch
def test_spin_param():
    """Test SpinParam data structure"""
    from deepchem.utils.dft_utils.datastruct import SpinParam

    sp = SpinParam(1, 2)
    assert sp.sum() == 3
    assert sp.reduce(lambda x, y: x * y + 2) == 4


@pytest.mark.torch
def test_base_xc():
    """Test BaseXC."""
    from deepchem.utils.dft_utils.datastruct import ValGrad
    from deepchem.utils.dft_utils.xc import BaseXC

    class MyXC(BaseXC):

        def family(self) -> int:
            return 1

        def get_edensityxc(self, densinfo) -> torch.Tensor:
            return densinfo.value**2

    xc = MyXC()
    densinfo = ValGrad(value=torch.tensor([1., 2., 3.]),
                       grad=torch.tensor([4., 5., 6.]))
    xc2 = xc + xc
    assert torch.all(xc.get_edensityxc(densinfo) == torch.tensor([1., 4., 9.]))
    assert torch.all(
        xc2.get_edensityxc(densinfo) == torch.tensor([2., 8., 18.]))
