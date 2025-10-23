import unittest
import os
import pandas as pd
from deepchem.utils.data_utils import load_sdf_files


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

    def test_load_sdf_files_with_shard_size(self):
        """
        Test `load_sdf_files()` generator with a `shard_size` smaller then length of input sdf file.
        The input sdf file has been infected with a corrupted molecule at index (0-based) 19 to test the
        effects of default argument `clean_mols=True`.
        """
        current_dir = os.path.dirname(os.path.realpath(__file__))
        file_path = [os.path.join(current_dir, 'assets', 'qm9_mini.sdf')]
        list_df = []
        for df in load_sdf_files(file_path, shard_size=5):
            list_df.append(df)
        loaded_df = pd.concat(list_df).reset_index(drop=True)

        required_shape = (
            20, 26
        )  # 1 invalid molecule datapoint to be ignored as `clean_mols=True`
        self.assertEqual(loaded_df.shape, required_shape)

        # assert loaded first molecule data from sdf
        self.assertEqual(loaded_df['smiles'][0], '[H]C([H])([H])[H]')
        n_atoms_mol1 = 5
        self.assertEqual(loaded_df['mol'][0].GetNumAtoms(), n_atoms_mol1)
        self.assertEqual(len(eval(loaded_df['pos_x'][0])), n_atoms_mol1)
        self.assertEqual(len(eval(loaded_df['pos_y'][0])), n_atoms_mol1)
        self.assertEqual(len(eval(loaded_df['pos_y'][0])), n_atoms_mol1)

        tasks = [
            'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'u0', 'u298',
            'h298', 'g298', 'cv'
        ]
        original_tasks_df = pd.read_csv(
            os.path.join(current_dir, 'assets', 'qm9_mini.sdf.csv'))
        y_expected = original_tasks_df[tasks].to_numpy()
        y_loaded = loaded_df[tasks].to_numpy()

        # check if loaded true labels are equal to all corresponding expected true labels for all datapoints (before invalid molecule)
        for i in range(19):
            self.assertTrue(
                all(y_loaded[i, :] == y_expected[i, :]),
                f"Mismatch of labels detected in datapoint with index {i}.")

        # check if the invalid molecule index (0-based) 19 was removed as `clean_mols=True`
        self.assertNotEqual(loaded_df['mol_id'].iloc[19, 0], 19,
                            "Invalid molecule not removed.")

        # check if the details of the invalid molecule is assigned to next molecule
        self.assertNotEqual(
            loaded_df['mol_id'].iloc[19, 1], 'gdb_20_invalid',
            "Invalid molecule details from csv mapped to next molecule.")
        self.assertFalse(
            all(y_loaded[19, :] == y_expected[19, :]),
            "Invalid molecule tasks from csv mapped to next molecule.")

        # check if the tasks of the next molecule (after skipped invalid molecule) match the expected task values
        self.assertTrue(
            all(y_loaded[19, :] == y_expected[20, :]),
            "Mismatch of labels detected in datapoint with index 20 (gdb_21).")
