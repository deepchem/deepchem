import unittest
from deepchem.utils.data_utils import load_sdf_files


class TestFileLoading(unittest.TestCase):

  def test_load_sdf_files(self):
    file_path = ['assets/gdb9_small.sdf']
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
