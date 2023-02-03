import deepchem as dc
import numpy as np
import os
import unittest


class TestAtomicConformation(unittest.TestCase):

    def test_featurize(self):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        sdf_file = os.path.join(current_dir, 'data', 'water.sdf')
        pdb_file = os.path.join(current_dir, 'data', '3zso_ligand_hyd.pdb')
        smiles = 'CCC'
        featurizer = dc.feat.AtomicConformationFeaturizer()
        features = featurizer.featurize([sdf_file, pdb_file, smiles])
        assert len(features) == 3

        # Check the SDF file.

        assert features[0].num_atoms == 60
        assert features[0].atomic_number[0] == 8
        assert features[0].atomic_number[1] == 1
        assert np.all(features[0].formal_charge == 0)
        for i in range(60):
            assert (features[0].partial_charge[i] < 0) == (i % 3 == 0)

        # Check the PDB file.

        assert features[1].num_atoms == 47
        assert features[1].atomic_number[0] == 6
        assert features[1].atomic_number[35] == 7
        assert features[1].atomic_number[46] == 1
        for i in range(47):
            if i == 36:
                assert features[1].formal_charge[i] == 1
            else:
                assert features[1].formal_charge[i] == 0
            if features[1].atomic_number[i] in (
                    7, 8):  # N and O should be negative
                assert features[1].partial_charge[i] < 0
            elif features[1].atomic_number[i] == 1:  # H should be positive
                assert features[1].partial_charge[i] > 0

        # Check the SMILES string.

        assert features[2].num_atoms == 11
        assert sum(features[2].atomic_number == 6) == 3
        assert sum(features[2].atomic_number == 1) == 8
        assert np.all(features[2].formal_charge == 0)
        for i in range(11):
            assert (features[2].partial_charge[i] <
                    0) == (features[2].atomic_number[i] == 6)
