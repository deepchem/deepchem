import pytest
import deepchem as dc
import tempfile

try:
    import torch
    has_torch = True
except ModuleNotFoundError:
    has_torch = False
    pass


@pytest.mark.torch
def test_fno_block_construction():
    from deepchem.models.torch_models.fno import FNOBlock
    block = FNOBlock(width=32, modes=8, dims=1)
    assert block is not None

    block = FNOBlock(width=32, modes=8, dims=2)
    assert block is not None

    block = FNOBlock(width=32, modes=8, dims=3)
    assert block is not None

@pytest.mark.torch
def test_fno_block_forward():
    from deepchem.models.torch_models.fno import FNOBlock
    block = FNOBlock(width=32, modes=8, dims=2)
    x = torch.rand(100, 32, 100, 100)
    y = block(x)
    assert y.shape == (100, 32, 100, 100)

@pytest.mark.torch
def test_fno_block_output_shape():
    from deepchem.models.torch_models.fno import FNOBlock
    block = FNOBlock(width=32, modes=8, dims=2)
    x = torch.rand(100, 32, 100, 100)
    y = block(x)
    assert y.shape == (100, 32, 100, 100)