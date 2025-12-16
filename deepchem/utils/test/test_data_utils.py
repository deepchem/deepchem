import unittest
import os
import tempfile
import warnings

import numpy as np

from deepchem.utils.data_utils import load_sdf_files, save_to_disk


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


class TestSaveToDisk(unittest.TestCase):

    def test_save_to_disk_warns_on_existing_file(self):
        # Existing file should trigger a UserWarning when overwrite is False
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "data.npy")
            np.save(file_path, np.zeros(1))  # create the file

            with self.assertWarns(UserWarning):
                save_to_disk(np.ones(1), file_path, overwrite=False)

    def test_save_to_disk_overwrite_true_no_warning(self):
        # overwrite=True should not emit a UserWarning
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "data.npy")
            np.save(file_path, np.zeros(1))

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                save_to_disk(np.ones(1), file_path, overwrite=True)

            self.assertFalse(
                any(isinstance(warn.message, UserWarning) for warn in w))

    def test_save_to_disk_directory_error(self):
        # Passing a directory path should raise IsADirectoryError
        with tempfile.TemporaryDirectory() as tmpdir:
            dir_path = os.path.join(tmpdir, "savedir")
            os.mkdir(dir_path)

            with self.assertRaises(IsADirectoryError):
                save_to_disk(np.ones(1), dir_path)
