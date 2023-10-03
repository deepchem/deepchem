import unittest
import os
from deepchem.utils.data_utils import load_sdf_files
import pytest

try:
    import torch
    from deepchem.utils.data_utils import chunkify, get_memory, get_dtype_memsize
    has_torch = True
except:
    has_torch = False


@pytest.mark.torch
def test_chunkify():
    """Test chunkify utility."""
    data = torch.arange(10)
    chunked_list = list(chunkify(data, 0, 3))
    assert len(chunked_list) == 4
    assert chunked_list[0][0].tolist() == [0, 1, 2]
    assert chunked_list[3][0].tolist() == [9]
    assert chunked_list[0][1] == 0
    assert chunked_list[3][1] == 9


@pytest.mark.torch
def test_get_memory():
    """Test get_memory utility."""
    data = torch.rand(100, 100, dtype=torch.float64)
    assert get_memory(data) == 100 * 100 * 8


@pytest.mark.torch
def test_get_dtype_memsize():
    """Test get_dtype_memsize utility."""
    data_1 = torch.rand(100, 100, dtype=torch.float64)
    data_2 = torch.rand(100, 100, dtype=torch.float32)
    assert get_dtype_memsize(data_1) == 8
    assert get_dtype_memsize(data_2) == 4


class TestFileLoading(unittest.TestCase):

    def test_load_sdf_files(self):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        file_path = [os.path.join(current_dir, 'assets', 'gdb9_small.sdf')]
        for df in load_sdf_files(file_path):
            break
        df_shape = (2, 6)
        self.assertEqual(df.shape, df_shape)
        self.assertEqual(df['smiles'][0], '[H]C([H])([H])[H]')
        n_atoms_mol1 = 5
        self.assertEqual(df['mol'][0].GetNumAtoms(), n_atoms_mol1)
        self.assertEqual(len(eval(df['pos_x'][0])), n_atoms_mol1)
        self.assertEqual(len(eval(df['pos_y'][0])), n_atoms_mol1)
        self.assertEqual(len(eval(df['pos_y'][0])), n_atoms_mol1)
