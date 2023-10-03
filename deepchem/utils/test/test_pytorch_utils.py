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
