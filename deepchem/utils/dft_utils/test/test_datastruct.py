"""
Test for DFT Config Utilities
"""

try:
    import torch
    from deepchem.utils.dft_utils.datastruct import ValGrad, SpinParam
    has_torch = True
except ModuleNotFoundError:
    has_torch = False

import pytest

def test_val_grad():
    """Test ValGrad data structure"""
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
    sp = SpinParam(1, 2)
    assert sp.sum() == 3
    assert sp.reduce(lambda x, y: x * y + 2) == 4
