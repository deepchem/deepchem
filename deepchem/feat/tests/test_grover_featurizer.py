"""
Test Grover Featurizer
"""
import deepchem as dc
import numpy as np
from deepchem.feat import GroverFeaturizer
from deepchem.feat.molecule_featurizers.grover_featurizer import \
    GROVER_RDKIT_PROPS
from rdkit import Chem


def test_grover_featurizer():
    featurizer = GroverFeaturizer()
    smiles = 'CC(=O)OC1=CC=CC=C1C(=O)O'
    molgraph = featurizer.featurize(smiles)[0]
    assert molgraph.num_nodes == 13
    assert molgraph.num_edges == 26
    # 151 = 133 + 18 (133 -> one hot encoding from _ATOM_FEATURES, 18 other features)
    assert molgraph.node_features.shape == (13, 151)
    assert molgraph.edge_index.shape == (2, 26)
    assert molgraph.edge_features.shape == (26, 165)
    assert molgraph.fg_labels.shape == (len(GROVER_RDKIT_PROPS),)

    smiles = 'CCC'
    mol = Chem.MolFromSmiles(smiles)
    num_atoms, num_bonds = mol.GetNumAtoms(), mol.GetNumBonds()
    cfp_size = 1024
    featurizer = GroverFeaturizer(
        features_generator=dc.feat.CircularFingerprint(size=cfp_size))
    molgraph = featurizer.featurize(smiles)[0]
    assert molgraph.num_nodes == num_atoms
    assert molgraph.num_edges == num_bonds * 2
    assert molgraph.node_features.shape == (num_atoms, 151)
    np.testing.assert_array_equal(molgraph.edge_index,
                                  np.asarray([[0, 2, 1, 2], [2, 0, 2, 1]]))
    assert molgraph.additional_features.shape == (cfp_size,)
