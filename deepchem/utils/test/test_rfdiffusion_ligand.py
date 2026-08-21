"""Tests for the lightweight SDF/MOL ligand point-cloud parser."""

import numpy as np
import pytest

from deepchem.utils.rfdiffusion_ligand import (
    LigandPointCloud,
    closest_distances,
    pairwise_distances,
)


class TestLigandPointCloud:

    def test_construction_and_shape(self):
        coords = np.zeros((4, 3))
        nums = np.array([6, 7, 8, 1])
        lp = LigandPointCloud(coords=coords, atomic_numbers=nums, name='X')
        assert lp.num_atoms == 4
        assert lp.name == 'X'

    def test_center_zero_centroid(self):
        coords = np.array([[1.0, 1.0, 1.0],
                           [-1.0, -1.0, -1.0],
                           [2.0, 0.0, 0.0],
                           [-2.0, 0.0, 0.0]])
        lp = LigandPointCloud(coords=coords,
                              atomic_numbers=np.array([6, 6, 6, 6]))
        centred = lp.center()
        assert np.allclose(centred.coords.mean(axis=0), 0.0, atol=1e-10)

    def test_shape_validation(self):
        with pytest.raises(ValueError):
            LigandPointCloud(coords=np.zeros((3, 2)),
                             atomic_numbers=np.array([1, 1, 1]))
        with pytest.raises(ValueError):
            LigandPointCloud(coords=np.zeros((3, 3)),
                             atomic_numbers=np.array([1, 1]))


class TestDistances:

    def test_pairwise_known_values(self):
        a = np.array([[0.0, 0.0, 0.0]])
        b = np.array([[3.0, 4.0, 0.0]])
        d = pairwise_distances(a, b)
        assert d.shape == (1, 1)
        assert abs(d[0, 0] - 5.0) < 1e-12

    def test_closest_distances(self):
        ca = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        lig = LigandPointCloud(
            coords=np.array([[1.0, 0.0, 0.0], [11.0, 0.0, 0.0]]),
            atomic_numbers=np.array([6, 6]))
        d, idx = closest_distances(lig, ca)
        assert d.shape == (2,) and idx.shape == (2,)
        assert abs(d[0] - 1.0) < 1e-12
        assert abs(d[1] - 1.0) < 1e-12
        assert idx.tolist() == [0, 1]

    def test_validation(self):
        with pytest.raises(ValueError):
            pairwise_distances(np.zeros((3, 2)), np.zeros((3, 3)))


# Optional RDKit-dependent tests below.
try:
    from rdkit import Chem  # noqa: F401
    has_rdkit = True
except ImportError:
    has_rdkit = False


@pytest.mark.skipif(not has_rdkit, reason='RDKit not installed.')
class TestSDFParsing:

    def test_parse_methane(self, tmp_path):
        # Build a one-atom SDF (carbon) directly via RDKit.
        from rdkit import Chem
        from rdkit.Chem import AllChem
        mol = Chem.MolFromSmiles('C')
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=0)
        path = tmp_path / 'methane.sdf'
        writer = Chem.SDWriter(str(path))
        writer.write(mol)
        writer.close()
        from deepchem.utils.rfdiffusion_ligand import parse_sdf
        clouds = parse_sdf(str(path), heavy_atoms_only=True)
        assert isinstance(clouds, list) and len(clouds) == 1
        lp = clouds[0]
        assert lp.num_atoms == 1
        assert lp.atomic_numbers[0] == 6
