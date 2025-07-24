from deepchem.utils.fno_utils import GaussianNormalizer
import torch
import pytest


@pytest.mark.torch
def test_gaussian_normalizer_norm():
    """
    Test if the GaussianNormalizer transforms data to zero mean and unit variance.
    """
    normalizer = GaussianNormalizer()
    data = torch.randn(100, 10) * 100
    normalizer.fit(data)
    normalized_data = normalizer.transform(data)
    assert torch.allclose(torch.mean(normalized_data),
                          torch.tensor([0.]),
                          atol=1)
    assert torch.allclose(torch.std(normalized_data),
                          torch.tensor([1.]),
                          atol=1)


@pytest.mark.torch
def test_gaussian_normalizer_denorm():
    """
    Test if the GaussianNormalizer transforms data back to the original scale.
    """
    normalizer = GaussianNormalizer()
    data = torch.randn(100, 10) * 100  # scaling data for testing
    normalizer.fit(data)
    normalized_data = normalizer.transform(data)
    denormalized_data = normalizer.inverse_transform(normalized_data)
    assert torch.allclose(data, denormalized_data, atol=1)


@pytest.mark.torch
def test_gaussian_normalizer_dim():
    """
    Test if the GaussianNormalizer can compute statistics over specified dimensions.
    """
    normalizer = GaussianNormalizer(dim=[0, 1])
    data = torch.randn(100, 10, 10) * 100
    normalizer.fit(data)
    normalized_data = normalizer.transform(data)
    assert torch.allclose(torch.mean(normalized_data, dim=[0, 1]),
                          torch.tensor([0.]),
                          atol=1)
    assert torch.allclose(torch.std(normalized_data, dim=[0, 1]),
                          torch.tensor([1.]),
                          atol=1)


@pytest.mark.torch
def test_gaussian_normalizer_to():
    """
    Test if the GaussianNormalizer can be moved to a different device.
    """
    normalizer = GaussianNormalizer()
    data = torch.randn(100, 10)
    normalizer.fit(data)
    if torch.cuda.is_available():
        normalizer.to(torch.device("cuda"))
        assert normalizer.mean.device == torch.device("cuda")
    else:
        normalizer.to(torch.device("cpu"))
        assert normalizer.mean.device == torch.device("cpu")
