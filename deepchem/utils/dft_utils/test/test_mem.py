"""
Tests for DFT Memory Utilities
"""

try:
    import torch
except:
    raise ModuleNotFoundError

from deepchem.utils.dft_utils.mem import chunkify, get_memory, get_dtype_memsize


def test_chunkify():
    """Test chunkify utility."""
    data = torch.arange(10)
    chunked_list = list(chunkify(data, 0, 3))
    assert len(chunked_list) == 4
    assert chunked_list[0][0].tolist() == [0, 1, 2]
    assert chunked_list[3][0].tolist() == [9]
    assert chunked_list[0][1] == 0
    assert chunked_list[3][1] == 9


def test_get_memory():
    """Test get_memory utility."""
    data = torch.rand(100, 100, dtype=torch.float64)
    assert get_memory(data) == 100 * 100 * 8


def test_get_dtype_memsize():
    """Test get_dtype_memsize utility."""
    data_1 = torch.rand(100, 100, dtype=torch.float64)
    data_2 = torch.rand(100, 100, dtype=torch.float32)
    assert get_dtype_memsize(data_1) == 8
    assert get_dtype_memsize(data_2) == 4
