"""
Test Autodock Vina Utility Functions.
"""
import os
import numpy as np
import unittest
from deepchem.utils import docking_utils
from deepchem.utils import rdkit_utils


class TestVinaUtils(unittest.TestCase):

    def setUp(self):
        # TODO test more formats for ligand
        self.current_dir = os.path.dirname(os.path.realpath(__file__))
        self.docked_ligands = os.path.join(self.current_dir, 'assets',
                                           '1jld_ligand_docked.pdbqt')

    def test_load_docked_ligand(self):
        docked_ligands, scores = docking_utils.load_docked_ligands(
            self.docked_ligands)
        assert len(docked_ligands) == 9
        assert len(scores) == 9

        for ligand, score in zip(docked_ligands, scores):
            xyz = rdkit_utils.get_xyz_from_mol(ligand)
            assert score < 0  # This is a binding free energy
            assert np.count_nonzero(xyz) > 0

    def test_write_gnina_conf(self):
        docking_utils.write_gnina_conf(
            'protein.pdb',
            'ligand.sdf',
            'conf.txt',
        )
        assert os.path.exists('conf.txt')
        os.remove('conf.txt')

    def test_read_gnina_log(self):
        log_file = os.path.join(self.current_dir, 'assets', 'gnina_log.txt')
        scores = docking_utils.read_gnina_log(log_file)
        assert np.array_equal(
            scores, np.array([[-4.37, 0.6392, 4.336], [-3.56, 0.6202, 4.162]]))

    def test_prepare_inputs(self):
        pdbid = '3cyx'
        ligand_smiles = 'CC(C)(C)NC(O)C1CC2CCCCC2C[NH+]1CC(O)C(CC1CCCCC1)NC(O)C(CC(N)O)NC(O)C1CCC2CCCCC2N1'

        protein, ligand = docking_utils.prepare_inputs(pdbid,
                                                       ligand_smiles,
                                                       pdb_name=pdbid)

        assert np.isclose(protein.GetNumAtoms(), 1510, atol=3)
        assert np.isclose(ligand.GetNumAtoms(), 124, atol=3)

        protein, ligand = docking_utils.prepare_inputs(
            pdbid + '.pdb', 'ligand_' + pdbid + '.pdb')

        assert np.isclose(protein.GetNumAtoms(), 1510, atol=3)
        assert np.isclose(ligand.GetNumAtoms(), 124, atol=3)

        os.remove(pdbid + '.pdb')
        os.remove('ligand_' + pdbid + '.pdb')
