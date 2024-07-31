import pytest
try:
    import torch
    from deepchem.models.dft import DM21
except ModuleNotFoundError:
    pass


@pytest.mark.torch
def test_dm21():
    model = DM21()
    input = torch.rand((100, 11))  # 11 Features
    output = model(input)
    assert output.shape == torch.Size((100, 3))
