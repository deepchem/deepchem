import deepchem as dc
import numpy as np
import pytest
try:
    import torch
    has_torch = True
except ModuleNotFoundError:
    has_torch = False


@pytest.mark.torch
def test_unsorted_segment_sum():

    segment_ids = torch.Tensor([0, 1, 0]).to(torch.int64)
    data = torch.Tensor([[1, 2, 3, 4], [5, 6, 7, 8], [4, 3, 2, 1]])
    num_segments = 2

    if len(segment_ids.shape) != 1:
        raise AssertionError("segment_ids have be a 1-D tensor")

    if data.shape[0] != segment_ids.shape[0]:
        raise AssertionError(
            "segment_ids should be the same size as dimension 0 of input.")

    result = dc.utils.pytorch_utils.unsorted_segment_sum(
        data=data, segment_ids=segment_ids, num_segments=num_segments)

    assert np.allclose(
        np.array(result),
        np.load("deepchem/utils/test/assets/result_segment_sum.npy"),
        atol=1e-04)


@pytest.mark.torch
def test_segment_sum():

    data = torch.Tensor([[1, 2, 3, 4], [4, 3, 2, 1], [5, 6, 7, 8]])
    segment_ids = torch.Tensor([0, 0, 1]).to(torch.int64)

    if len(segment_ids.shape) != 1:
        raise AssertionError("segment_ids have be a 1-D tensor")

    if data.shape[0] != segment_ids.shape[0]:
        raise AssertionError(
            "segment_ids should be the same size as dimension 0 of input.")

    result = dc.utils.pytorch_utils.segment_sum(data=data,
                                                segment_ids=segment_ids)

    assert np.allclose(
        np.array(result),
        np.load("deepchem/utils/test/assets/result_segment_sum.npy"),
        atol=1e-04)


@pytest.mark.torch
def test_chunkify():
    """Test chunkify utility."""
    data = torch.arange(10)
    chunked_list = list(dc.utils.pytorch_utils.chunkify(data, 0, 3))
    assert len(chunked_list) == 4
    assert chunked_list[0][0].tolist() == [0, 1, 2]
    assert chunked_list[3][0].tolist() == [9]
    assert chunked_list[0][1] == 0
    assert chunked_list[3][1] == 9


@pytest.mark.torch
def test_get_memory():
    """Test get_memory utility."""
    data = torch.rand(100, 100, dtype=torch.float64)
    assert dc.utils.pytorch_utils.get_memory(data) == 100 * 100 * 8


@pytest.mark.torch
def test_gaussian_integral():
    """Test the gaussian integral utility."""
    assert dc.utils.pytorch_utils.gaussian_integral(5, 1.0) == 1.0


@pytest.mark.torch
def test_TensorNonTensorSeparator():
    """Test the TensorNonTensorSeparator utility."""
    a = torch.tensor([1., 2, 3])
    b = 4.
    c = torch.tensor([5., 6, 7], requires_grad=True)
    params = [a, b, c]
    separator = dc.utils.pytorch_utils.TensorNonTensorSeparator(params)
    tensor_params = separator.get_tensor_params()
    assert torch.allclose(tensor_params[0],
                          torch.tensor([5., 6., 7.], requires_grad=True))


@pytest.mark.torch
def test_tallqr():
    V = torch.randn(3, 2)
    Q, R = dc.utils.pytorch_utils.tallqr(V)
    assert Q.shape == torch.Size([3, 2])
    assert R.shape == torch.Size([2, 2])
    assert torch.allclose(Q @ R, V)


@pytest.mark.torch
def test_to_fortran_order():
    V = torch.randn(3, 2)
    if V.is_contiguous() is False:
        assert False
    V = dc.utils.pytorch_utils.to_fortran_order(V)
    if V.is_contiguous() is True:
        assert False
    assert V.shape == torch.Size([3, 2])


@pytest.mark.torch
def test_get_np_dtype():
    """Test the get_np_dtype utility."""
    assert dc.utils.pytorch_utils.get_np_dtype(torch.float32) == np.float32
    assert dc.utils.pytorch_utils.get_np_dtype(torch.float64) == np.float64


@pytest.mark.torch
def test_unsorted_segment_max():

    segment_ids = torch.Tensor([0, 1, 0]).to(torch.int64)
    data = torch.Tensor([[1, 2, 3, 4], [5, 6, 7, 8], [4, 3, 2, 1]])
    num_segments = 2

    if len(segment_ids.shape) != 1:
        raise AssertionError("segment_ids have be a 1-D tensor")

    if data.shape[0] != segment_ids.shape[0]:
        raise AssertionError(
            "segment_ids should be the same size as dimension 0 of input.")

    result = dc.utils.pytorch_utils.unsorted_segment_max(
        data=data, segment_ids=segment_ids, num_segments=num_segments)

    assert np.allclose(
        np.array(result),
        np.load("deepchem/utils/test/assets/result_segment_max.npy"),
        atol=1e-04)
