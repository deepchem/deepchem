import pytest

try:
    import torch
    has_torch = True
except ModuleNotFoundError:
    has_torch = False
    pass


@pytest.mark.torch
def test_fno_block_construction():
    """
    Test if FNOBlock can be constructed in 1D, 2D, and 3D.
    """
    from deepchem.models.torch_models.fno import FNOBlock
    block = FNOBlock(width=32, modes=8, dims=1)
    assert block is not None, "1D FNO Block construction failed"

    block = FNOBlock(width=32, modes=8, dims=2)
    assert block is not None, "2D FNO Block construction failed"

    block = FNOBlock(width=32, modes=8, dims=3)
    assert block is not None, "3D FNO Block construction failed"


@pytest.mark.torch
def test_fno_block_forward():
    """
    Test if FNOBlock can be forward-passed in 2D, without any errors.
    """
    from deepchem.models.torch_models.fno import FNOBlock
    block = FNOBlock(width=32, modes=8, dims=2)
    x = torch.rand(100, 32, 100, 100)
    y = block(x)
    assert y.shape == (100, 32, 100, 100)
