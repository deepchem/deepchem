import os
from typing import List

import numpy as np
import pandas as pd
import pytest

import deepchem as dc

pytest.importorskip("ase")


def _asset_path(filename: str) -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, filename)


def _make_loader(tasks: List[str]) -> dc.data.MaterialsLoader:
    featurizer = dc.feat.AtomisticRadiusGraphFeaturizer(cutoff=2.5)
    return dc.data.MaterialsLoader(tasks=tasks,
                                   featurizer=featurizer,
                                   energy_key="my_energy",
                                   forces_key="my_forces")


def test_materials_loader_energy_from_asset():
    input_file = _asset_path("materials_loader_energy_forces.extxyz")

    dataset = _make_loader(["energy"]).create_dataset(input_file, shard_size=1)

    assert len(dataset) == 2
    assert dataset.y.shape == (2, 1)
    assert dataset.w.shape == (2, 1)
    assert float(dataset.y[0, 0]) == pytest.approx(1.5)
    assert float(dataset.y[1, 0]) == pytest.approx(-2.0)


def test_materials_loader_forces_variable_atom_counts_from_asset():
    input_file = _asset_path("materials_loader_energy_forces.extxyz")

    dataset = _make_loader(["forces"]).create_dataset(input_file)

    assert len(dataset) == 2
    assert dataset.y.shape == (2, 1)
    assert dataset.y[0, 0].shape == (2, 3)
    assert dataset.y[1, 0].shape == (3, 3)
    np.testing.assert_allclose(
        dataset.y[0, 0],
        np.array([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]], dtype=np.float32))
    assert dataset.y[0, 0].dtype == np.float32


def test_materials_loader_energy_and_forces_task_layout_from_asset():
    input_file = _asset_path("materials_loader_energy_forces.extxyz")

    dataset = _make_loader(["energy", "forces"]).create_dataset(input_file)

    assert dataset.y.shape == (2, 2)
    assert dataset.y.dtype == object
    assert dataset.w.dtype == np.float32
    assert float(dataset.y[0, 0]) == pytest.approx(1.5)
    assert dataset.y[0, 1].shape == (2, 3)
    assert float(dataset.y[1, 0]) == pytest.approx(-2.0)
    assert dataset.y[1, 1].shape == (3, 3)
    np.testing.assert_array_equal(dataset.w, np.ones((2, 2), dtype=np.float32))


def test_materials_loader_unlabeled_tasks_empty_from_xyz_asset():
    input_file = _asset_path("materials_loader_unlabeled.xyz")

    dataset = _make_loader([]).create_dataset(input_file)

    assert len(dataset) == 2
    assert len(dataset.get_task_names()) == 0
    X, y, w, ids = dataset.get_shard(0)
    assert len(X) == 2
    assert y is None
    assert w is None
    assert list(ids) == [f"{input_file}:0", f"{input_file}:1"]


def test_materials_loader_missing_requested_label_raises():
    input_file = _asset_path("materials_loader_unlabeled.xyz")

    with pytest.raises(ValueError,
                       match=("task 'energy'.*key 'my_energy'.*"
                              "materials_loader_unlabeled.xyz.*frame 0")):
        _make_loader(["energy"]).create_dataset(input_file)


def test_materials_loader_invalid_forces_shape_raises(tmp_path):
    input_file = tmp_path / "bad_forces.extxyz"
    input_file.write_text(
        "2\n"
        "Properties=species:S:1:pos:R:3:my_forces:R:2 my_energy=1.0\n"
        "H 0.0 0.0 0.0 0.1 0.0\n"
        "O 1.0 0.0 0.0 -0.1 0.0\n")

    with pytest.raises(
            ValueError,
            match="task 'forces'.*key 'my_forces'.*bad_forces.extxyz.*frame 0"):
        _make_loader(["forces"]).create_dataset(str(input_file))


def test_materials_loader_multiple_input_files_order_from_asset():
    input_file = _asset_path("materials_loader_energy_forces.extxyz")

    dataset = _make_loader(["energy"]).create_dataset([input_file, input_file],
                                                      shard_size=3)

    assert len(dataset) == 4
    assert len(dataset.get_shard(0)[0]) == 3
    assert len(dataset.get_shard(1)[0]) == 1
    assert [float(value) for value in dataset.y[:, 0]] == [1.5, -2.0, 1.5, -2.0]
    assert list(dataset.ids) == [
        f"{input_file}:0", f"{input_file}:1", f"{input_file}:0",
        f"{input_file}:1"
    ]


def test_materials_loader_graphdata_features_from_asset():
    input_file = _asset_path("materials_loader_energy_forces.extxyz")

    dataset = _make_loader(["energy"]).create_dataset(input_file)

    assert isinstance(dataset.X[0], dc.feat.GraphData)
    assert hasattr(dataset.X[0], "node_features")
    assert hasattr(dataset.X[0], "edge_index")


def test_materials_loader_shard_helpers_from_asset():
    input_file = _asset_path("materials_loader_energy_forces.extxyz")
    loader = _make_loader(["energy", "forces"])

    shards = list(loader._get_shards([input_file], shard_size=None))

    assert len(shards) == 1
    shard = shards[0]
    assert isinstance(shard, pd.DataFrame)
    assert list(shard.columns) == ["atoms", "id", "energy", "forces"]
    assert list(shard["id"]) == [f"{input_file}:0", f"{input_file}:1"]
    assert list(shard["energy"]) == [1.5, -2.0]
    assert shard["forces"].iloc[0].shape == (2, 3)
    assert shard["forces"].iloc[1].shape == (3, 3)

    features, valid_inds = loader._featurize_shard(shard)

    assert len(features) == len(shard)
    assert all(isinstance(feature, dc.feat.GraphData) for feature in features)
    assert valid_inds.dtype == bool
    assert valid_inds.tolist() == [True, True]


def test_materials_loader_invalid_task_raises():
    featurizer = dc.feat.AtomisticRadiusGraphFeaturizer(cutoff=2.5)
    with pytest.raises(ValueError, match="Unsupported tasks"):
        dc.data.MaterialsLoader(tasks=["dipole"], featurizer=featurizer)


@pytest.mark.parametrize("inputs", [5, ["frames.extxyz", 5]])
def test_materials_loader_invalid_inputs_raise(inputs):
    with pytest.raises(ValueError, match="MaterialsLoader"):
        _make_loader([]).create_dataset(inputs)
