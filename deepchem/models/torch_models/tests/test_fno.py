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

@pytest.mark.torch
def test_fno_base_construction():
    """
    Test if FNO can be constructed in 1D, 2D, and 3D.
    """
    from deepchem.models.torch_models.fno import FNO
    model = FNO(in_channels=1, out_channels=1, modes=8, width=32, dims=2)
    assert model is not None, "FNOBase construction failed"

@pytest.mark.torch
def test_fno_base_forward():
    """
    Test if FNO can be forward-passed in 2D, without any errors.
    """
    from deepchem.models.torch_models.fno import FNO
    model = FNO(in_channels=1, out_channels=1, modes=8, width=32, dims=2)
    x = torch.rand(100, 1, 100, 100)
    y = model(x)
    assert y.shape == (100, 1, 100, 100)

@pytest.mark.torch
def test_fno_base_fit_normalizers():
    """
    Test if normalizers works as expected.
    """
    from deepchem.models.torch_models.fno import FNO
    model = FNO(in_channels=1, out_channels=1, modes=8, width=32, dims=2)
    x = torch.rand(100, 1, 100, 100)
    y = torch.rand(100, 1, 100, 100)
    model.fit_normalizers(x, y)
    normlized_x = model.input_normalizer.transform(x)
    normalized_y = model.output_normalizer.transform(y)
    denormalized_y = model.output_normalizer.inverse_transform(normalized_y)
    assert torch.allclose(torch.mean(normlized_x), torch.tensor([0.]), atol=1e-6)
    assert torch.allclose(torch.std(normlized_x), torch.tensor([1.]), atol=1e-6)
    assert torch.allclose(torch.mean(normalized_y), torch.tensor([0.]), atol=1e-6)
    assert torch.allclose(torch.std(normalized_y), torch.tensor([1.]), atol=1e-6)
    print(denormalized_y[0, 0, 0, :10])
    print(y[0, 0, 0, :10])
    assert torch.allclose(denormalized_y, y, atol=1e-6)

@pytest.mark.torch
def test_fno_base_meshgrids():
    """
    Test if meshgrids are properly generated
    """
    from deepchem.models.torch_models.fno import FNO
    model = FNO(in_channels=1, out_channels=1, modes=8, width=32, dims=2)
    x = torch.rand(100, 1, 100, 100)
    meshgrid = model._generate_meshgrid(x)
    assert meshgrid.shape == (100, 2, 100, 100)
