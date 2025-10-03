import unittest
import os
import tempfile
import deepchem as dc
from rdkit import Chem
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


def test_no_absurd_nitrogen_charges():
    # Define the URL of the SDF file (update this with actual URL)
    QM9_URL = "https://deepchemdata.s3.us-west-1.amazonaws.com/datasets/qm9.tar.gz"

    # Create a temporary file with .sdf suffix
    with tempfile.TemporaryDirectory() as tmpdir:

        # Download SDF file using DeepChem utility
        dc.utils.data_utils.download_url(url=QM9_URL, dest_dir=tmpdir)
        dc.utils.data_utils.untargz_file(os.path.join(tmpdir, "qm9.tar.gz"),
                                         tmpdir)

        # Read SDF file with RDKit
        # Disable sanitization to preserve original formal charges and avoid auto-corrections
        suppl = Chem.SDMolSupplier(os.path.join(tmpdir, "qm9.sdf"),
                                   removeHs=False,
                                   sanitize=False)
        for mol in suppl:
            if mol is None:
                continue  # Skip invalid molecules
            for atom in mol.GetAtoms():
                assert atom.GetFormalCharge() == 0  # General check

                if atom.GetAtomicNum() == 7:  # Nitrogen
                    charge = atom.GetFormalCharge()
                    assert charge < 4, f"Nitrogen atom has absurd charge {charge} in molecule."
