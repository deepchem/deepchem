import numpy as np
import pytest
import torch
from deepchem.feat.RFDiffusion_featurizer import RFDiffusionFeaturizer

pytest.importorskip("Bio.PDB")


def _write_test_pdb(tmp_path, residue_count=4):
    """
    Writes a minimal PDB file with the given number of residues.

    Useful for testing the RFDiffusionFeatures featurizer functions
    """

    pdb_lines = []
    serial = 1

    for res_idx in range(1, residue_count + 1):

        x_offset = float(res_idx - 1) * 3.0
        atoms = [
            ("N", x_offset, 1.0, 0.0, "N"),
            ("CA", x_offset, 0.0, 0.0, "C"),
            ("C", x_offset + 1.0, 0.0, 0.0, "C"),
        ]

        for atom_name, x_coord, y_coord, z_coord, element in atoms:

            pdb_lines.append(
                f"ATOM  {serial:5d} {atom_name:^4} ALA A{res_idx:4d}"
                f"    {x_coord:8.3f}{y_coord:8.3f}{z_coord:8.3f}"
                f"  1.00 20.00           {element:>2}\n")
            serial += 1

    pdb_lines.append("END\n")

    pdb_file = tmp_path / "test_backbone.pdb"
    pdb_file.write_text("".join(pdb_lines))
    return pdb_file


def _pdb_atom_line(serial, atom_name, residue_name, residue_id, coords):
    """
    Helper function to write a PDB atom line.
    """
    x_coord, y_coord, z_coord = coords

    element = atom_name[0]

    return (f"ATOM  {serial:5d} {atom_name:^4} {residue_name} A{residue_id:4d}"
            f"    {x_coord:8.3f}{y_coord:8.3f}{z_coord:8.3f}"
            f"  1.00 20.00           {element:>2}\n")


def test_get_backbone_coords_extracts_and_center_crops(tmp_path):
    """
    Test extraction function for N, CA, C coordinates from a PDB file
    """

    pdb_file = _write_test_pdb(tmp_path, residue_count=4)
    featurizer = RFDiffusionFeaturizer(max_length=2)

    coords = featurizer.get_backbone_coords(str(pdb_file))
    coords = np.asarray(coords)

    assert coords.shape == (2, 3, 3)
    np.testing.assert_allclose(
        coords[0], [[3.0, 1.0, 0.0], [3.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    np.testing.assert_allclose(
        coords[1], [[6.0, 1.0, 0.0], [6.0, 0.0, 0.0], [7.0, 0.0, 0.0]])


def test_get_backbone_coords_does_not_crop_short_backbone(tmp_path):
    """
    Test backbones shorter than max_length are not cropped.
    """

    pdb_file = _write_test_pdb(tmp_path, residue_count=3)
    featurizer = RFDiffusionFeaturizer(max_length=5)

    coords = np.asarray(featurizer.get_backbone_coords(str(pdb_file)))

    assert coords.shape == (3, 3, 3)
    np.testing.assert_allclose(
        coords[0], [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    np.testing.assert_allclose(
        coords[-1], [[6.0, 1.0, 0.0], [6.0, 0.0, 0.0], [7.0, 0.0, 0.0]])


def test_get_backbone_coords_skips_residues_with_missing_backbone_atoms(
        tmp_path):
    """
    Test residues missing N, CA, or C atoms are skipped.
    """

    pdb_lines = [
        _pdb_atom_line(1, "N", "ALA", 1, (0.0, 1.0, 0.0)),
        _pdb_atom_line(2, "CA", "ALA", 1, (0.0, 0.0, 0.0)),
        _pdb_atom_line(3, "N", "GLY", 2, (3.0, 1.0, 0.0)),
        _pdb_atom_line(4, "CA", "GLY", 2, (3.0, 0.0, 0.0)),
        _pdb_atom_line(5, "C", "GLY", 2, (4.0, 0.0, 0.0)),
        "END\n",
    ]
    pdb_file = tmp_path / "missing_backbone_atom.pdb"
    pdb_file.write_text("".join(pdb_lines))
    featurizer = RFDiffusionFeaturizer()

    coords = np.asarray(featurizer.get_backbone_coords(str(pdb_file)))

    assert coords.shape == (1, 3, 3)
    np.testing.assert_allclose(
        coords[0], [[3.0, 1.0, 0.0], [3.0, 0.0, 0.0], [4.0, 0.0, 0.0]])


def test_get_backbone_coords_reads_only_first_model(tmp_path):
    """Test only the first PDB model is used."""

    pdb_lines = [
        "MODEL        1\n",
        _pdb_atom_line(1, "N", "ALA", 1, (0.0, 1.0, 0.0)),
        _pdb_atom_line(2, "CA", "ALA", 1, (0.0, 0.0, 0.0)),
        _pdb_atom_line(3, "C", "ALA", 1, (1.0, 0.0, 0.0)),
        "ENDMDL\n",
        "MODEL        2\n",
        _pdb_atom_line(4, "N", "ALA", 1, (99.0, 1.0, 0.0)),
        _pdb_atom_line(5, "CA", "ALA", 1, (99.0, 0.0, 0.0)),
        _pdb_atom_line(6, "C", "ALA", 1, (100.0, 0.0, 0.0)),
        "ENDMDL\n",
        "END\n",
    ]

    pdb_file = tmp_path / "two_model_backbone.pdb"
    pdb_file.write_text("".join(pdb_lines))
    featurizer = RFDiffusionFeaturizer()

    coords = np.asarray(featurizer.get_backbone_coords(str(pdb_file)))

    assert coords.shape == (1, 3, 3)
    np.testing.assert_allclose(
        coords[0], [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])


@pytest.mark.torch
def test_get_1d_features_shape_and_values():
    """Test 1D features have expected shape and finite values."""

    featurizer = RFDiffusionFeaturizer()
    backbone = torch.zeros((3, 3, 3), dtype=torch.float32)

    features = featurizer.get_1d_features(backbone, t=10, d_model=16)

    assert features.shape == (1, 3, 16)
    assert torch.isfinite(features).all()


@pytest.mark.torch
def test_get_2d_features_shape_and_symmetry():
    """Test 2D distance-bin features are pairwise and symmetric."""

    featurizer = RFDiffusionFeaturizer()
    translations = torch.tensor([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0],
                                 [0.0, 4.0, 0.0]])

    features = featurizer.get_2d_features(translations)

    assert features.shape == (3, 3, 64)
    assert torch.allclose(features, features.transpose(0, 1))
    assert torch.isfinite(features).all()


@pytest.mark.torch
def test_get_frames_returns_orthonormal_rotations_and_ca_translations():
    """Test local frames are built from N, CA, and C backbone atoms."""

    featurizer = RFDiffusionFeaturizer()
    backbone = torch.tensor([[[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0,
                                                                 0.0]],
                             [[2.0, 1.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0,
                                                                 0.0]]])

    rotations, translations = featurizer.get_frames(backbone)

    assert rotations.shape == (2, 3, 3)
    assert translations.shape == (2, 3)
    torch.testing.assert_close(translations, backbone[:, 1, :])

    identity = torch.eye(3).expand(2, 3, 3)
    torch.testing.assert_close(
        rotations.transpose(-1, -2) @ rotations, identity)


@pytest.mark.torch
@pytest.mark.xfail(
    reason="_featurize currently passes PDB-derived coordinates as non-tensor "
    "inputs and passes translations where get_1d_features expects a timestep.")
def test_featurize_integrates_rf_diffusion_feature_tracks(tmp_path):
    """Test featurize returns 1D features, 2D features, and frames."""

    pdb_file = _write_test_pdb(tmp_path, residue_count=2)
    featurizer = RFDiffusionFeaturizer()

    one_d_features, two_d_features, frames = featurizer._featurize(
        str(pdb_file))
    rotations, translations = frames

    assert one_d_features.shape == (1, 2, 64)
    assert two_d_features.shape == (2, 2, 64)
    assert rotations.shape == (2, 3, 3)
    assert translations.shape == (2, 3)
    assert torch.isfinite(one_d_features).all()
    assert torch.isfinite(two_d_features).all()
