import numpy as np
import pytest

from deepchem.utils.data_utils import pad_array


def test_pad_array_basic():
    arr = np.array([1, 2, 3])
    padded = pad_array(arr, 5)
    assert padded.shape[0] == 5
    assert np.allclose(padded[:3], arr)


def test_pad_array_empty():
    arr = np.array([])
    padded = pad_array(arr, 3)
    assert padded.shape[0] == 3


def test_pad_array_negative_shape():
    arr = np.array([1, 2])
    with pytest.raises(Exception):
        pad_array(arr, -1)
