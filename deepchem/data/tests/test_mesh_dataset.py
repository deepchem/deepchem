"""
Tests for mesh_dataset.py and mesh_featurizer.py

Run with:
    pytest tests/test_mesh_dataset.py -v
"""

import pytest
import numpy as np
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from mesh_dataset import MeshDataset
from mesh_featurizer import MeshFeaturizer

# helpers

def _minimal_dataset(n=3, with_solutions=False):
    """Return a MeshDataset with n identical 4-node samples."""
    nodes = np.array([[0,0],[1,0],[0,1],[1,1]], dtype=np.float32)
    elements = np.array([[0,1,2],[1,3,2]], dtype=np.int64)
    bm = np.array([True, True, True, False])
    bv = np.zeros(4, dtype=np.float32)
    src = np.ones(4, dtype=np.float32)
    solutions = [np.array([0,0,0,0.25], dtype=np.float32)] * n \
        if with_solutions else None
    return MeshDataset(
        nodes=[nodes] * n,
        elements=[elements] * n,
        boundary_masks=[bm] * n,
        boundary_values=[bv] * n,
        sources=[src] * n,
        solutions=solutions,
    )

#tests

def test_dataset_len():
    ds = _minimal_dataset(n=5)
    assert len(ds) == 5


def test_dataset_getitem_keys():
    ds = _minimal_dataset(n=1)
    sample = ds[0]
    required = {'nodes', 'elements', 'boundary_mask', 'boundary_values', 'source'}
    assert required.issubset(sample.keys()), (
        f"Missing keys: {required - sample.keys()}"
    )


def test_dataset_getitem_shapes():
    ds = _minimal_dataset(n=1)
    sample = ds[0]
    assert sample['nodes'].shape == (4, 2)
    assert sample['elements'].shape == (2, 3)
    assert sample['boundary_mask'].shape == (4,)
    assert sample['boundary_values'].shape == (4,)
    assert sample['source'].shape == (4,)


def test_dataset_getitem_dtypes():
    ds = _minimal_dataset(n=1)
    sample = ds[0]
    assert sample['nodes'].dtype == torch.float32
    assert sample['elements'].dtype == torch.long
    assert sample['boundary_mask'].dtype == torch.bool
    assert sample['boundary_values'].dtype == torch.float32
    assert sample['source'].dtype == torch.float32


def test_dataset_solution_present_when_provided():
    ds = _minimal_dataset(n=2, with_solutions=True)
    sample = ds[0]
    assert 'solution' in sample, "solution key missing when solutions provided"
    assert sample['solution'].shape == (4,)
    assert sample['solution'].dtype == torch.float32


def test_dataset_solution_absent_when_not_provided():
    ds = _minimal_dataset(n=2, with_solutions=False)
    sample = ds[0]
    assert 'solution' not in sample, "solution key present but not provided"


def test_dataset_iter():
    """Iterating over dataset must yield all samples."""
    n = 4
    ds = _minimal_dataset(n=n)
    count = sum(1 for _ in ds)
    assert count == n


def test_dataset_mismatched_lengths_raises():
    nodes = [np.zeros((4, 2), dtype=np.float32)] * 3
    elements = [np.zeros((2, 3), dtype=np.int64)] * 2  # wrong length
    bm = [np.zeros(4, dtype=bool)] * 3
    bv = [np.zeros(4, dtype=np.float32)] * 3
    src = [np.zeros(4, dtype=np.float32)] * 3
    with pytest.raises(ValueError, match="equal length"):
        MeshDataset(nodes, elements, bm, bv, src)


def test_dataset_mismatched_solutions_raises():
    ds_nodes = [np.zeros((4, 2), dtype=np.float32)] * 3
    elements = [np.zeros((2, 3), dtype=np.int64)] * 3
    bm = [np.zeros(4, dtype=bool)] * 3
    bv = [np.zeros(4, dtype=np.float32)] * 3
    src = [np.zeros(4, dtype=np.float32)] * 3
    solutions = [np.zeros(4, dtype=np.float32)] * 2  # wrong length
    with pytest.raises(ValueError):
        MeshDataset(ds_nodes, elements, bm, bv, src, solutions=solutions)


def test_dataset_stats():
    ds = _minimal_dataset(n=3)
    stats = ds.stats()
    assert stats['n_samples'] == 3
    assert stats['mean_nodes'] == 4.0
    assert stats['mean_elements'] == 2.0
    assert stats['has_solutions'] is False


def test_dataset_stats_with_solutions():
    ds = _minimal_dataset(n=3, with_solutions=True)
    assert ds.stats()['has_solutions'] is True


#tets for featurizer

def test_featurizer_node_count():
    feat = MeshFeaturizer(nx=4, ny=4)
    assert feat.n_nodes == 25   # (4+1)*(4+1)


def test_featurizer_element_count():
    feat = MeshFeaturizer(nx=4, ny=4)
    assert feat.n_elements == 32   # 2*4*4


def test_featurizer_output_keys():
    feat = MeshFeaturizer(nx=4, ny=4)
    meshes = feat.featurize(
        source_fns=[lambda x, y: np.ones_like(x)],
        boundary_fns=[lambda x, y: np.zeros_like(x)],
    )
    assert len(meshes) == 1
    required = {'nodes', 'elements', 'boundary_mask', 'boundary_values', 'source'}
    assert required.issubset(meshes[0].keys())


def test_featurizer_output_shapes():
    nx, ny = 4, 4
    feat = MeshFeaturizer(nx=nx, ny=ny)
    meshes = feat.featurize(
        source_fns=[lambda x, y: np.ones_like(x)],
    )
    m = meshes[0]
    N = (nx + 1) * (ny + 1)
    E = 2 * nx * ny
    assert m['nodes'].shape == (N, 2)
    assert m['elements'].shape == (E, 3)
    assert m['boundary_mask'].shape == (N,)
    assert m['boundary_values'].shape == (N,)
    assert m['source'].shape == (N,)


def test_featurizer_output_dtypes():
    feat = MeshFeaturizer(nx=4, ny=4)
    meshes = feat.featurize(source_fns=[lambda x, y: np.ones_like(x)])
    m = meshes[0]
    assert m['nodes'].dtype == np.float32
    assert m['elements'].dtype == np.int64
    assert m['source'].dtype == np.float32
    assert m['boundary_values'].dtype == np.float32


def test_featurizer_default_zero_bc():
    """Default boundary condition must be zero."""
    feat = MeshFeaturizer(nx=4, ny=4)
    meshes = feat.featurize(source_fns=[lambda x, y: np.ones_like(x)])
    bv = meshes[0]['boundary_values']
    assert (bv == 0).all(), f"Default BC is not zero: {bv}"


def test_featurizer_source_evaluated_correctly():
    """Source term must equal the function evaluated at node coordinates."""
    feat = MeshFeaturizer(nx=4, ny=4)
    meshes = feat.featurize(
        source_fns=[lambda x, y: x + y],
    )
    nodes = meshes[0]['nodes']
    expected = (nodes[:, 0] + nodes[:, 1]).astype(np.float32)
    np.testing.assert_allclose(meshes[0]['source'], expected, rtol=1e-6)


def test_featurizer_multiple_problems():
    """Featurizing n problems must return n mesh dicts."""
    feat = MeshFeaturizer(nx=4, ny=4)
    n = 5
    meshes = feat.featurize(
        source_fns=[lambda x, y: np.ones_like(x)] * n,
    )
    assert len(meshes) == n


def test_featurizer_length_mismatch_raises():
    feat = MeshFeaturizer(nx=4, ny=4)
    with pytest.raises(ValueError):
        feat.featurize(
            source_fns=[lambda x, y: np.ones_like(x)] * 3,
            boundary_fns=[lambda x, y: np.zeros_like(x)] * 2,
        )


def test_featurizer_refine():
    """refine(2) must double nx and ny."""
    feat = MeshFeaturizer(nx=4, ny=4)
    fine = feat.refine(2)
    assert fine.nx == 8
    assert fine.ny == 8
    assert fine.n_nodes == 81  # (8+1)*(8+1)


def test_from_featurizer_classmethod():
    """MeshDataset.from_featurizer must produce correct dataset."""
    feat = MeshFeaturizer(nx=4, ny=4)
    ds = MeshDataset.from_featurizer(
        featurizer=feat,
        source_fns=[lambda x, y: np.ones_like(x)] * 3,
        boundary_fns=[lambda x, y: np.zeros_like(x)] * 3,
    )
    assert len(ds) == 3
    assert ds[0]['nodes'].shape == (25, 2)


def test_from_featurizer_with_solutions():
    """Solutions stored via solution_fns must match evaluated function."""
    feat = MeshFeaturizer(nx=4, ny=4)
    ds = MeshDataset.from_featurizer(
        featurizer=feat,
        source_fns=[lambda x, y: np.ones_like(x)],
        solution_fns=[lambda x, y: (x * (1 - x) * y * (1 - y)).astype(np.float32)],
    )
    assert 'solution' in ds[0]
    nodes = ds[0]['nodes'].numpy()
    expected = (nodes[:, 0] * (1 - nodes[:, 0]) *
                nodes[:, 1] * (1 - nodes[:, 1]))
    np.testing.assert_allclose(
        ds[0]['solution'].numpy(), expected, rtol=1e-5)