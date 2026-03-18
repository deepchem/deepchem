import pytest
try:
    import torch
except:
    pass


@pytest.mark.torch
def test_safepow():
    from deepchem.utils.safeops_utils import safepow

    a = torch.tensor([1e-35, 2e-40])
    p = torch.tensor([2., 3])
    result = safepow(a, p)
    assert torch.allclose(torch.tensor([1.0000e-24, 1.0000e-36]), result)
    assert torch.allclose(torch.tensor([0., 0.]), a**p)


@pytest.mark.torch
def test_safenorm():
    from deepchem.utils.safeops_utils import safenorm

    a = torch.tensor([1e-35, 2e-40])
    result = safenorm(a, 0)
    assert torch.allclose(torch.tensor(1.4142e-15), result)
    assert torch.allclose(torch.tensor(1.4142e-15), a.norm())


@pytest.mark.torch
def test_occnumber():
    from deepchem.utils.safeops_utils import occnumber

    result = occnumber(torch.tensor(2.6), 3, torch.double, torch.device('cpu'))
    assert torch.allclose(
        torch.tensor([1.0000, 1.0000, 0.6000], dtype=torch.float64), result)


@pytest.mark.torch
def test_construct_occ_number():
    from deepchem.utils.safeops_utils import _construct_occ_number

    result = _construct_occ_number(2.5, 2, 3, 3, torch.double,
                                   torch.device('cpu'))
    assert torch.allclose(
        torch.tensor([1.0000, 1.0000, 0.5000], dtype=torch.float64), result)


@pytest.mark.torch
def test_get_floor_and_ceil():
    from deepchem.utils.safeops_utils import get_floor_and_ceil

    result = get_floor_and_ceil(2.5)
    assert result[0] == 2
    assert result[1] == 3


@pytest.mark.torch
def test_safe_cdist():
    from deepchem.utils.safeops_utils import safe_cdist

    a = torch.tensor([[1., 2], [3, 4]])
    b = torch.tensor([[1., 2], [3, 4]])

    result_1 = safe_cdist(a, b)
    assert torch.allclose(result_1,
                          torch.tensor([[0.0000, 2.8284], [2.8284, 0.0000]]))

    result_2 = safe_cdist(a, b, add_diag_eps=True)
    assert torch.allclose(
        result_2,
        torch.tensor([[1.4142e-12, 2.8284e+00], [2.8284e+00, 1.4142e-12]]))
