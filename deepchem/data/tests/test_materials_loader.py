import numpy as np
import pytest

import deepchem as dc

ase = pytest.importorskip("ase")
ase_io = pytest.importorskip("ase.io")

Atoms = ase.Atoms
write = ase_io.write


def _write_atoms_file(path, frames, fmt):
    write(path, frames, format=fmt)
    return str(path)


def _make_atoms(numbers, positions, energy=None, forces=None, stress=None):
    atoms = Atoms(numbers=numbers, positions=positions)
    if energy is not None:
        atoms.info["my_energy"] = energy
    if forces is not None:
        atoms.arrays["my_forces"] = np.asarray(forces, dtype=np.float32)
    if stress is not None:
        atoms.info["my_stress"] = np.asarray(stress, dtype=np.float32)
    return atoms


def _make_loader(tasks):
    featurizer = dc.feat.AtomisticRadiusGraphFeaturizer(cutoff=2.5)
    return dc.data.MaterialsLoader(tasks=tasks,
                                   featurizer=featurizer,
                                   energy_key="my_energy",
                                   forces_key="my_forces",
                                   stress_key="my_stress")


def _make_default_loader(tasks):
    featurizer = dc.feat.AtomisticRadiusGraphFeaturizer(cutoff=2.5)
    return dc.data.MaterialsLoader(tasks=tasks, featurizer=featurizer)


def test_materials_loader_energy(tmp_path):
    frames = [
        _make_atoms(numbers=[1, 8],
                    positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                    energy=1.5),
        _make_atoms(numbers=[6, 1, 1],
                    positions=[[0.0, 0.0, 0.0], [0.8, 0.0, 0.0],
                               [-0.8, 0.0, 0.0]],
                    energy=-2.0)
    ]
    input_file = _write_atoms_file(tmp_path / "energy.extxyz", frames, "extxyz")

    dataset = _make_loader(["energy"]).create_dataset(input_file, shard_size=1)

    assert len(dataset) == 2
    assert dataset.y.shape == (2, 1)
    assert dataset.w.shape == (2, 1)
    assert float(dataset.y[0, 0]) == pytest.approx(1.5)
    assert float(dataset.y[1, 0]) == pytest.approx(-2.0)


def test_materials_loader_forces_variable_atom_counts(tmp_path):
    frames = [
        _make_atoms(numbers=[1, 8],
                    positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                    forces=[[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]]),
        _make_atoms(numbers=[6, 1, 1],
                    positions=[[0.0, 0.0, 0.0], [0.8, 0.0, 0.0],
                               [-0.8, 0.0, 0.0]],
                    forces=[[0.2, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.2]])
    ]
    input_file = _write_atoms_file(tmp_path / "forces.extxyz", frames, "extxyz")

    dataset = _make_loader(["forces"]).create_dataset(input_file)

    assert len(dataset) == 2
    assert dataset.y.shape == (2, 1)
    assert dataset.y[0, 0].shape == (2, 3)
    assert dataset.y[1, 0].shape == (3, 3)
    np.testing.assert_allclose(
        dataset.y[0, 0],
        np.array([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]], dtype=np.float32))


def test_materials_loader_energy_and_forces(tmp_path):
    frames = [
        _make_atoms(numbers=[1, 8],
                    positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                    energy=1.5,
                    forces=[[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]]),
        _make_atoms(numbers=[6, 1, 1],
                    positions=[[0.0, 0.0, 0.0], [0.8, 0.0, 0.0],
                               [-0.8, 0.0, 0.0]],
                    energy=-2.0,
                    forces=[[0.2, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.2]])
    ]
    input_file = _write_atoms_file(tmp_path / "energy_forces.extxyz", frames,
                                   "extxyz")

    dataset = _make_loader(["energy", "forces"]).create_dataset(input_file)

    assert dataset.y.shape == (2, 2)
    assert float(dataset.y[0, 0]) == pytest.approx(1.5)
    assert dataset.y[0, 1].shape == (2, 3)
    assert float(dataset.y[1, 0]) == pytest.approx(-2.0)
    assert dataset.y[1, 1].shape == (3, 3)


def test_materials_loader_energy_forces_stress(tmp_path):
    stress = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3], dtype=np.float32)
    frames = [
        _make_atoms(numbers=[1, 8],
                    positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                    energy=1.5,
                    forces=[[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]],
                    stress=stress)
    ]
    input_file = _write_atoms_file(tmp_path / "stress.extxyz", frames, "extxyz")

    dataset = _make_loader(["energy", "forces",
                            "stress"]).create_dataset(input_file)

    assert dataset.y.shape == (1, 3)
    stored_stress = dataset.y[0, 2]
    assert stored_stress.shape in ((6,), (3, 3))
    if stored_stress.shape == (6,):
        np.testing.assert_allclose(stored_stress, stress)


def test_materials_loader_default_energy_forces_stress_keys(tmp_path):
    stress = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3], dtype=np.float32)
    frames = [
        Atoms(numbers=[1, 8], positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        Atoms(numbers=[6, 1, 1],
              positions=[[0.0, 0.0, 0.0], [0.8, 0.0, 0.0], [-0.8, 0.0, 0.0]])
    ]
    frames[0].info["energy"] = 1.5
    frames[0].arrays["forces"] = np.array([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]],
                                          dtype=np.float32)
    frames[0].info["stress"] = stress
    frames[1].info["energy"] = -2.0
    frames[1].arrays["forces"] = np.array(
        [[0.2, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.2]], dtype=np.float32)
    frames[1].info["stress"] = stress * 2.0
    input_file = _write_atoms_file(tmp_path / "default_keys.extxyz", frames,
                                   "extxyz")

    dataset = _make_default_loader(["energy", "forces",
                                    "stress"]).create_dataset(input_file)

    assert dataset.y.shape == (2, 3)
    assert float(dataset.y[0, 0]) == pytest.approx(1.5)
    assert float(dataset.y[1, 0]) == pytest.approx(-2.0)
    np.testing.assert_allclose(
        dataset.y[0, 1],
        np.array([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]], dtype=np.float32))
    np.testing.assert_allclose(
        dataset.y[1, 1],
        np.array([[0.2, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.2]],
                 dtype=np.float32))
    assert dataset.y[0, 2].shape in ((6,), (3, 3))
    assert dataset.y[1, 2].shape in ((6,), (3, 3))


def test_materials_loader_unlabeled_tasks_empty(tmp_path):
    frames = [
        _make_atoms(numbers=[1, 8],
                    positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        _make_atoms(numbers=[6, 1, 1],
                    positions=[[0.0, 0.0, 0.0], [0.8, 0.0, 0.0],
                               [-0.8, 0.0, 0.0]])
    ]
    input_file = _write_atoms_file(tmp_path / "plain.xyz", frames, "xyz")

    dataset = _make_loader([]).create_dataset(input_file)

    assert len(dataset) == 2
    assert len(dataset.get_task_names()) == 0
    X, y, w, ids = dataset.get_shard(0)
    assert len(X) == 2
    assert y is None
    assert w is None
    assert ids.shape == (2,)


def test_materials_loader_missing_requested_label_raises(tmp_path):
    frames = [
        _make_atoms(numbers=[1, 8],
                    positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                    energy=1.5)
    ]
    input_file = _write_atoms_file(tmp_path / "missing.extxyz", frames,
                                   "extxyz")

    loader = _make_loader(["energy", "forces"])
    with pytest.raises(
            ValueError,
            match="task 'forces'.*key 'my_forces'.*missing.extxyz.*frame 0"):
        loader.create_dataset(input_file)


def test_materials_loader_multiple_input_files_order(tmp_path):
    file_one = _write_atoms_file(tmp_path / "part1.extxyz", [
        _make_atoms(numbers=[1, 8],
                    positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                    energy=1.0),
        _make_atoms(numbers=[6, 1, 1],
                    positions=[[0.0, 0.0, 0.0], [0.8, 0.0, 0.0],
                               [-0.8, 0.0, 0.0]],
                    energy=2.0)
    ], "extxyz")
    file_two = _write_atoms_file(tmp_path / "part2.extxyz", [
        _make_atoms(numbers=[8, 1],
                    positions=[[0.0, 0.0, 0.0], [0.9, 0.0, 0.0]],
                    energy=3.0)
    ], "extxyz")

    dataset = _make_loader(["energy"]).create_dataset([file_one, file_two],
                                                      shard_size=2)

    assert len(dataset) == 3
    assert [float(value) for value in dataset.y[:, 0]] == [1.0, 2.0, 3.0]
    assert list(
        dataset.ids) == [f"{file_one}:0", f"{file_one}:1", f"{file_two}:0"]


def test_materials_loader_graphdata_features(tmp_path):
    frames = [
        _make_atoms(numbers=[1, 8],
                    positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                    energy=1.5)
    ]
    input_file = _write_atoms_file(tmp_path / "graph.extxyz", frames, "extxyz")

    dataset = _make_loader(["energy"]).create_dataset(input_file)

    assert isinstance(dataset.X[0], dc.feat.GraphData)
    assert hasattr(dataset.X[0], "node_features")
    assert hasattr(dataset.X[0], "edge_index")


def test_materials_loader_invalid_task_raises():
    featurizer = dc.feat.AtomisticRadiusGraphFeaturizer(cutoff=2.5)
    with pytest.raises(ValueError, match="Unsupported tasks"):
        dc.data.MaterialsLoader(tasks=["dipole"], featurizer=featurizer)
