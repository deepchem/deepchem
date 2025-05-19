import unittest
import os
import tempfile
from openbabel import pybel
from deepchem.utils.data_utils import load_sdf_files, convert_xyz_files_to_sdf


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


def test_no_absurd_nitrogen_charges():
    # Create a temporary output SDF file
    xyz_folder_path = './assets/xyz_test'
    with tempfile.NamedTemporaryFile(suffix=".sdf", delete=False) as tmp_sdf:
        sdf_output_file_path = tmp_sdf.name

    try:
        # Convert the XYZ files to SDF
        convert_xyz_files_to_sdf(xyz_folder_path, sdf_output_file_path)

        # Read the SDF file and validate nitrogen charges
        for mol in pybel.readfile("sdf", sdf_output_file_path):
            print()
            for atom in mol.atoms:
                print(atom, atom.formalcharge)
                assert atom.formalcharge == 0
                if atom.atomicnum == 7:
                    charge = atom.formalcharge
                    assert charge < 4, f"Nitrogen atom has absurd charge {charge} in molecule: {mol.title}"

    finally:
        # Clean up the temporary file
        os.remove(sdf_output_file_path)
