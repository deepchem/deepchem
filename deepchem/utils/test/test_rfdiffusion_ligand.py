"""Tests for RFDiffusion All-Atom ligand parsing utilities."""

import numpy as np
import pytest

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    has_rdkit = True
except ImportError:
    has_rdkit = False

from deepchem.utils.rfdiffusion_ligand import (LIGAND_ATOM_TYPES,
                                               LigandPointCloud,
                                               find_ligand_automorphisms,
                                               parse_ligand_file)

requires_rdkit = pytest.mark.skipif(not has_rdkit, reason='RDKit not installed')


def _write_sdf(smiles, path, seed=0):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=seed)
    writer = Chem.SDWriter(path)
    writer.write(mol)
    writer.close()
    return mol


class TestLigandPointCloud:

    def test_valid_construction(self):
        coords = np.zeros((3, 3), dtype=np.float32)
        atom_types = np.array([0, 1, 2])
        atomic_numbers = np.array([6, 7, 8])
        bonds = np.zeros((3, 3), dtype=np.int64)
        ligand = LigandPointCloud(coords, atom_types, atomic_numbers, bonds)
        assert ligand.num_atoms == 3

    def test_name_stored(self):
        coords = np.zeros((1, 3), dtype=np.float32)
        ligand = LigandPointCloud(coords,
                                  np.array([0]),
                                  np.array([6]),
                                  np.zeros((1, 1)),
                                  name='foo.sdf')
        assert ligand.name == 'foo.sdf'

    def test_bad_coords_shape_raises(self):
        with pytest.raises(ValueError):
            LigandPointCloud(np.zeros((3, 2)), np.zeros(3), np.zeros(3),
                             np.zeros((3, 3)))

    def test_atom_types_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            LigandPointCloud(np.zeros((3, 3)), np.zeros(2), np.zeros(3),
                             np.zeros((3, 3)))

    def test_atomic_numbers_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            LigandPointCloud(np.zeros((3, 3)), np.zeros(3), np.zeros(2),
                             np.zeros((3, 3)))

    def test_bond_features_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            LigandPointCloud(np.zeros((3, 3)), np.zeros(3), np.zeros(3),
                             np.zeros((2, 2)))


@pytest.mark.torch
@requires_rdkit
class TestParseLigandFile:

    def test_ethanol_atom_count_and_types(self, tmp_path):
        path = str(tmp_path / 'ethanol.sdf')
        _write_sdf('CCO', path)
        ligand = parse_ligand_file(path)
        assert ligand.num_atoms == 3  # C, C, O (Hs removed by default)
        assert sorted(ligand.atomic_numbers.tolist()) == [6, 6, 8]
        # Oxygen's LIGAND_ATOM_TYPES index.
        o_index = LIGAND_ATOM_TYPES.index('O')
        assert o_index in ligand.atom_types.tolist()

    def test_keep_hydrogens(self, tmp_path):
        path = str(tmp_path / 'ethanol_h.sdf')
        _write_sdf('CCO', path)
        with_h = parse_ligand_file(path, remove_hydrogens=False)
        without_h = parse_ligand_file(path, remove_hydrogens=True)
        assert with_h.num_atoms > without_h.num_atoms

    def test_bond_features_symmetric(self, tmp_path):
        path = str(tmp_path / 'ethanol2.sdf')
        _write_sdf('CCO', path)
        ligand = parse_ligand_file(path)
        assert np.array_equal(ligand.bond_features, ligand.bond_features.T)

    def test_aromatic_bond_coded_as_four(self, tmp_path):
        path = str(tmp_path / 'benzene.sdf')
        _write_sdf('c1ccccc1', path)
        ligand = parse_ligand_file(path)
        # every ring bond should be coded as aromatic (4)
        nonzero = ligand.bond_features[ligand.bond_features != 0]
        assert set(nonzero.tolist()) == {4}

    def test_double_bond_order(self, tmp_path):
        path = str(tmp_path / 'ethene.sdf')
        _write_sdf('C=C', path)
        ligand = parse_ligand_file(path)
        assert ligand.bond_features[0, 1] == 2

    def test_missing_file_raises(self):
        with pytest.raises((ValueError, OSError)):
            parse_ligand_file('/nonexistent/path/to/ligand.sdf')

    def test_unsupported_extension_raises(self, tmp_path):
        path = str(tmp_path / 'ligand.xyz')
        with open(path, 'w') as f:
            f.write('not a real file')
        with pytest.raises(ValueError):
            parse_ligand_file(path)


@pytest.mark.torch
@requires_rdkit
class TestFindLigandAutomorphisms:

    def test_identity_is_first_row(self, tmp_path):
        path = str(tmp_path / 'ethanol3.sdf')
        _write_sdf('CCO', path)
        perms = find_ligand_automorphisms(path)
        assert perms.shape[0] >= 1
        assert perms[0].tolist() == list(range(perms.shape[1]))

    def test_benzene_has_multiple_symmetries(self, tmp_path):
        path = str(tmp_path / 'benzene2.sdf')
        _write_sdf('c1ccccc1', path)
        perms = find_ligand_automorphisms(path)
        assert perms.shape[1] == 6
        # benzene's automorphism group (D6h restricted to atom
        # permutations) has more than just the identity permutation
        assert perms.shape[0] > 1

    def test_asymmetric_molecule_has_only_identity(self, tmp_path):
        # 2-chlorobutane: no two heavy atoms are topologically
        # equivalent, so the only automorphism is the identity.
        path = str(tmp_path / 'asym.sdf')
        _write_sdf('CC(Cl)CC', path)
        perms = find_ligand_automorphisms(path)
        assert perms.shape[0] == 1

    def test_max_matches_respected(self, tmp_path):
        path = str(tmp_path / 'benzene3.sdf')
        _write_sdf('c1ccccc1', path)
        perms = find_ligand_automorphisms(path, max_matches=1)
        assert perms.shape[0] == 1


class TestOptionalDependencyBehavior:

    def test_missing_rdkit_raises_importerror(self, monkeypatch):
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == 'rdkit' or name.startswith('rdkit.'):
                raise ImportError('simulated missing rdkit')
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, '__import__', fake_import)
        with pytest.raises(ImportError):
            parse_ligand_file('does_not_matter.sdf')
