"""
Test featurizer class.
"""
import numpy as np
import unittest

from rdkit import Chem

from deepchem.featurizers.basic import MolecularWeight
from vs_utils.utils.parallel_utils import LocalCluster
from vs_utils.utils.rdkit_utils import conformers


class TestFeaturizer(unittest.TestCase):
    """
    Tests for Featurizer.
    """
    def setUp(self):
        """
        Set up tests.
        """
        smiles = 'CC(=O)OC1=CC=CC=C1C(=O)O'
        mol = Chem.MolFromSmiles(smiles)
        engine = conformers.ConformerGenerator(max_conformers=1)
        self.mol = engine.generate_conformers(mol)
        assert self.mol.GetNumConformers() > 0

    def test_featurizer(self):
        """
        Test basic functionality of Featurizer.
        """
        f = MolecularWeight()
        rval = f([self.mol])
        assert rval.shape == (1, 1)

    def test_flatten_conformers(self):
        """
        Calculate molecule-level features for a multiconformer molecule.
        """
        f = MolecularWeight()
        rval = f([self.mol])
        assert rval.shape == (1, 1)

    def test_parallel(self):
        """
        Test parallel featurization.
        """
        cluster = LocalCluster(1)
        f = MolecularWeight()
        rval = f([self.mol])
        parallel_rval = f([self.mol], parallel=True,
                          client_kwargs={'cluster_id': cluster.cluster_id})
        assert np.array_equal(rval, parallel_rval)
