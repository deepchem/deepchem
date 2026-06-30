import os
import unittest
import tempfile
import numpy as np
import pandas as pd
import pytest
import deepchem as dc
from rdkit import Chem
from deepchem.utils.data_utils import extract_ase_energy
from deepchem.utils.data_utils import extract_ase_forces
from deepchem.utils.data_utils import extract_ase_labels
from deepchem.utils.data_utils import extract_ase_stress
from deepchem.utils.data_utils import featurize_ase_frame
from deepchem.utils.data_utils import finalize_materials_shard
from deepchem.utils.data_utils import get_ase_calc_results
from deepchem.utils.data_utils import load_sdf_files
from deepchem.utils.data_utils import make_materials_frame_id
from deepchem.utils.data_utils import normalize_materials_inputs
from deepchem.molnet.load_function.qm9_datasets import QM9_URL


def _make_ase_atoms():
    ase = pytest.importorskip("ase")
    return ase.Atoms(numbers=[1, 8],
                     positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])


def _set_single_point_results(atoms, **results):
    pytest.importorskip("ase")
    from ase.calculators.singlepoint import SinglePointCalculator
    atoms.calc = SinglePointCalculator(atoms, **results)


def test_normalize_materials_inputs_string():
    assert normalize_materials_inputs("frames.extxyz") == ["frames.extxyz"]


def test_normalize_materials_inputs_list():
    inputs = ["part1.extxyz", "part2.extxyz"]
    assert normalize_materials_inputs(inputs) == inputs


def test_normalize_materials_inputs_tuple():
    assert normalize_materials_inputs(
        ("part1.extxyz", "part2.extxyz")) == ["part1.extxyz", "part2.extxyz"]


def test_normalize_materials_inputs_invalid_non_iterable():
    with pytest.raises(ValueError, match="MaterialsLoader"):
        normalize_materials_inputs(5)


def test_normalize_materials_inputs_rejects_non_string_member():
    with pytest.raises(ValueError, match="MaterialsLoader"):
        normalize_materials_inputs(path for path in ["frames.extxyz", 5])


def test_make_materials_frame_id():
    assert make_materials_frame_id("frames.extxyz", 3) == "frames.extxyz:3"


def test_get_ase_calc_results_without_calculator():
    atoms = _make_ase_atoms()
    assert get_ase_calc_results(atoms) == {}


def test_get_ase_calc_results_with_results():
    atoms = _make_ase_atoms()
    _set_single_point_results(atoms, energy=1.5)
    assert get_ase_calc_results(atoms) is atoms.calc.results
    assert get_ase_calc_results(atoms)["energy"] == pytest.approx(1.5)


def test_extract_ase_energy_from_info():
    atoms = _make_ase_atoms()
    atoms.info["my_energy"] = 1.5
    assert extract_ase_energy(atoms, "my_energy", "frames.extxyz",
                              0) == pytest.approx(1.5)


def test_extract_ase_energy_from_calc_results():
    atoms = _make_ase_atoms()
    _set_single_point_results(atoms, energy=-2.0)
    assert extract_ase_energy(atoms, "energy", "frames.extxyz",
                              1) == pytest.approx(-2.0)


def test_extract_ase_energy_missing_raises():
    atoms = _make_ase_atoms()
    with pytest.raises(
            ValueError,
            match="task 'energy'.*key 'energy'.*frames.extxyz.*frame 2"):
        extract_ase_energy(atoms, "energy", "frames.extxyz", 2)


def test_extract_ase_forces_from_arrays():
    atoms = _make_ase_atoms()
    expected = np.array([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]], dtype=np.float32)
    atoms.arrays["my_forces"] = expected
    actual = extract_ase_forces(atoms, "my_forces", "frames.extxyz", 0)
    np.testing.assert_allclose(actual, expected)
    assert actual.dtype == np.float32


def test_extract_ase_forces_from_calc_results():
    atoms = _make_ase_atoms()
    expected = np.array([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]])
    _set_single_point_results(atoms, forces=expected)
    actual = extract_ase_forces(atoms, "forces", "frames.extxyz", 0)
    np.testing.assert_allclose(actual, expected)
    assert actual.dtype == np.float32


def test_extract_ase_forces_missing_raises():
    atoms = _make_ase_atoms()
    with pytest.raises(
            ValueError,
            match="task 'forces'.*key 'forces'.*frames.extxyz.*frame 1"):
        extract_ase_forces(atoms, "forces", "frames.extxyz", 1)


def test_extract_ase_forces_invalid_shape_raises():
    atoms = _make_ase_atoms()
    atoms.arrays["forces"] = np.zeros((2, 2))
    with pytest.raises(
            ValueError,
            match="task 'forces'.*key 'forces'.*frames.extxyz.*frame 0"):
        extract_ase_forces(atoms, "forces", "frames.extxyz", 0)


def test_extract_ase_stress_from_info():
    atoms = _make_ase_atoms()
    expected = np.arange(6, dtype=np.float32)
    atoms.info["my_stress"] = expected
    actual = extract_ase_stress(atoms, "my_stress", "frames.extxyz", 0)
    np.testing.assert_allclose(actual, expected)
    assert actual.dtype == np.float32


def test_extract_ase_stress_from_calc_results():
    atoms = _make_ase_atoms()
    expected = np.arange(6, dtype=float)
    _set_single_point_results(atoms, stress=expected)
    actual = extract_ase_stress(atoms, "stress", "frames.extxyz", 0)
    np.testing.assert_allclose(actual, expected)
    assert actual.dtype == np.float32


def test_extract_ase_stress_missing_raises():
    atoms = _make_ase_atoms()
    with pytest.raises(
            ValueError,
            match="task 'stress'.*key 'stress'.*frames.extxyz.*frame 1"):
        extract_ase_stress(atoms, "stress", "frames.extxyz", 1)


def test_extract_ase_stress_invalid_shape_raises():
    atoms = _make_ase_atoms()
    atoms.info["stress"] = np.zeros((3,))
    with pytest.raises(
            ValueError,
            match="task 'stress'.*key 'stress'.*frames.extxyz.*frame 0"):
        extract_ase_stress(atoms, "stress", "frames.extxyz", 0)


def test_extract_ase_labels_preserves_task_order():
    atoms = _make_ase_atoms()
    expected_forces = np.array([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]])
    atoms.info["energy"] = 1.5
    atoms.arrays["forces"] = expected_forces
    labels = extract_ase_labels(atoms, ["forces", "energy"], "energy", "forces",
                                "stress", "frames.extxyz", 0)
    assert labels.shape == (2,)
    np.testing.assert_allclose(labels[0], expected_forces)
    assert labels[1] == pytest.approx(1.5)


def test_extract_ase_labels_unsupported_task_raises():
    atoms = _make_ase_atoms()
    with pytest.raises(ValueError, match="Unsupported materials task 'dipole'"):
        extract_ase_labels(atoms, ["dipole"], "energy", "forces", "stress",
                           "frames.extxyz", 0)


class _DummyFeaturizer:

    def __init__(self, outputs):
        self.outputs = outputs
        self.inputs = None
        self.log_every_n = None

    def featurize(self, inputs, log_every_n):
        self.inputs = inputs
        self.log_every_n = log_every_n
        return self.outputs


def test_featurize_ase_frame_returns_one_feature():
    atoms = object()
    feature = object()
    featurizer = _DummyFeaturizer([feature])
    assert featurize_ase_frame(featurizer, atoms, 25, "frames.xyz",
                               2) is feature
    assert featurizer.inputs == [atoms]
    assert featurizer.log_every_n == 25


def test_featurize_ase_frame_zero_outputs_raises():
    with pytest.raises(ValueError, match="returned 0 outputs"):
        featurize_ase_frame(_DummyFeaturizer([]), object(), 10, "frames.xyz", 0)


def test_featurize_ase_frame_multiple_outputs_raises():
    with pytest.raises(ValueError, match="returned 2 outputs"):
        featurize_ase_frame(_DummyFeaturizer([object(), object()]), object(),
                            10, "frames.xyz", 0)


def test_featurize_ase_frame_empty_array_raises():
    with pytest.raises(ValueError, match="Failed to featurize"):
        featurize_ase_frame(_DummyFeaturizer([np.array([])]), object(), 10,
                            "frames.xyz", 0)


def test_finalize_materials_shard_labeled():
    first_labels = np.empty((2,), dtype=object)
    first_labels[:] = [1.5, np.ones((2, 3), dtype=np.float32)]
    second_labels = np.empty((2,), dtype=object)
    second_labels[:] = [-2.0, np.ones((3, 3), dtype=np.float32)]

    X, y, w, ids = finalize_materials_shard(["graph1", "graph2"],
                                            [first_labels, second_labels],
                                            ["energy", "forces"],
                                            ["frames.xyz:0", "frames.xyz:1"])

    assert X.dtype == object
    assert X.tolist() == ["graph1", "graph2"]
    assert y is not None
    assert y.shape == (2, 2)
    assert y.dtype == object
    assert y[0, 0] == pytest.approx(1.5)
    assert y[1, 1].shape == (3, 3)
    assert w is not None
    np.testing.assert_array_equal(w, np.ones((2, 2), dtype=np.float32))
    assert w.dtype == np.float32
    assert ids.dtype == object
    assert ids.tolist() == ["frames.xyz:0", "frames.xyz:1"]


def test_finalize_materials_shard_unlabeled():
    X, y, w, ids = finalize_materials_shard(["graph"], None, [],
                                            ["frames.xyz:0"])
    assert X.dtype == object
    assert X.tolist() == ["graph"]
    assert y is None
    assert w is None
    assert ids.dtype == object
    assert ids.tolist() == ["frames.xyz:0"]


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


def test_qm9_molecules_charge_neutrality_sanitize_false():

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a temporary file with .sdf suffix
        dc.utils.data_utils.download_url(url=QM9_URL, dest_dir=tmpdir)

        dc.utils.data_utils.untargz_file(os.path.join(tmpdir, "qm9.tar.gz"),
                                         tmpdir)

        qm9_sdf = os.path.join(tmpdir, "qm9.sdf")
        if not os.path.isfile(qm9_sdf):
            qm9_sdf = os.path.join(tmpdir, "qm9", "qm9.sdf")
        assert os.path.isfile(qm9_sdf)

        # Read SDF file with RDKit (no sanitization)
        suppl = Chem.SDMolSupplier(qm9_sdf,
                                   removeHs=False,
                                   sanitize=False,
                                   strictParsing=False)

        for i, mol in enumerate(suppl):
            if mol is None:
                continue
            # qm9 molecules should be neutral
            total_charge = sum(
                atom.GetFormalCharge() for atom in mol.GetAtoms())
            assert total_charge == 0, f"Molecule {i+1} is not neutral."


def test_qm9_molecules_charge_neutrality_sanitize_true():

    # Create a temporary file with .sdf suffix
    with tempfile.TemporaryDirectory() as tmpdir:

        # Download and extract
        dc.utils.data_utils.download_url(url=QM9_URL, dest_dir=tmpdir)
        dc.utils.data_utils.untargz_file(os.path.join(tmpdir, "qm9.tar.gz"),
                                         tmpdir)

        qm9_sdf = os.path.join(tmpdir, "qm9.sdf")
        if not os.path.isfile(qm9_sdf):
            qm9_sdf = os.path.join(tmpdir, "qm9", "qm9.sdf")
        assert os.path.isfile(qm9_sdf)

        # Read SDF file with RDKit (with sanitization)
        suppl = Chem.SDMolSupplier(qm9_sdf,
                                   removeHs=False,
                                   sanitize=True,
                                   strictParsing=False)
        for i, mol in enumerate(suppl):
            if mol is None:
                continue
            # qm9 molecules should be neutral
            total_charge = sum(
                atom.GetFormalCharge() for atom in mol.GetAtoms())
            assert total_charge == 0, f"Molecule {i+1} is not neutral."
