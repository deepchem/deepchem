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

    # length of segment_ids.shape should be 1
    assert len(segment_ids.shape) == 1

    # Shape of segment_ids should be equal to first dimension of data
    assert segment_ids.shape[-1] == data.shape[0]
    result = dc.utils.pytorch_utils.unsorted_segment_sum(
        data=data, segment_ids=segment_ids, num_segments=num_segments)

    assert np.allclose(
        np.array(result),
        np.load("deepchem/utils/test/assets/result_segment_sum.npy"),
        atol=1e-04)
