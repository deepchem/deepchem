"""Test DFT utils."""

import pytest
try:
    import torch
    has_torch = True
except:
    has_torch = False

from deepchem.utils.dft_utils.xc import BaseXC
from deepchem.utils.dft_utils.datastruct import ValGrad


@pytest.mark.torch
def test_base_xc():

    class MyXC(BaseXC):

        def family(self) -> int:
            return 1

        def get_edensityxc(self, densinfo: ValGrad) -> torch.Tensor:
            return densinfo.value**2

    xc = MyXC()
    densinfo = ValGrad(value=torch.tensor([1., 2., 3.]),
                       grad=torch.tensor([4., 5., 6.]))
    xc2 = xc + xc
    assert torch.all(xc.get_edensityxc(densinfo) == torch.tensor([1., 4., 9.]))
    assert torch.all(
        xc2.get_edensityxc(densinfo) == torch.tensor([2., 8., 18.]))
