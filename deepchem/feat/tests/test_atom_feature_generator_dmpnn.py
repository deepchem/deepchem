"""
Test for atom feature vector generator and its helper functions
"""

from deepchem.feat.molecule_featurizers.dmpnn_featurizer import get_atomic_num_one_hot, get_atom_chiral_tag_one_hot, get_atom_mass, atom_features, GraphConvConstants
from rdkit import Chem
import pytest
import numpy as np


@pytest.fixture
def example_smiles_n_features():
    """
    Sample data for testing

    Returns
    -------
    dictionary
    format {'smiles':required feature vector : List}
    """
    feature_vector_C = [[
        0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0.12011
    ]]
    feature_vector_NN = [[
        0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1,
        0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0.14007
    ],
                         [
                             0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0,
                             1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0.14007
                         ]]
    return {'C': feature_vector_C, 'N#N': feature_vector_NN}


def test_helper_functions():
    """
    Test for get_atomic_num_one_hot(), get_atom_chiral_tag_one_hot() and get_atom_mass() helper functions
    """
    smiles = 'C'
    m = Chem.MolFromSmiles(smiles)
    atom = m.GetAtoms()[0]
    f_atomic = get_atomic_num_one_hot(
        atom, GraphConvConstants.ATOM_FEATURES['atomic_num'])
    req_f = list(np.zeros((101,), dtype=float))
    req_f[5] = 1.0
    assert len(f_atomic) == len(req_f)
    assert f_atomic == req_f

    f_chiral_tag = get_atom_chiral_tag_one_hot(
        atom, GraphConvConstants.ATOM_FEATURES['chiral_tag'])
    ref_f = [1.0, 0.0, 0.0, 0.0, 0.0]
    assert len(f_chiral_tag) == len(ref_f)
    assert f_chiral_tag == ref_f

    f_mass = get_atom_mass(atom)
    ref_f = [0.12011]
    assert len(f_mass) == len(ref_f)
    assert f_mass == ref_f


def test_atom_features_none():
    """
    Test for atom_features() with 'None' input for Atom value
    """
    f_atom = atom_features(None)
    req_f = list(np.zeros((133,), dtype=int))
    assert len(f_atom) == len(req_f)
    assert f_atom == req_f


def test_atom_features_only_atom_num():
    """
    Test for atom_features() when only_atom_num is True
    """
    smiles = 'C'
    m = Chem.MolFromSmiles(smiles)
    atom = m.GetAtoms()[0]
    features = atom_features(atom, only_atom_num=True)
    req_f = list(np.zeros((133,), dtype=int))
    req_f[5] = 1
    assert len(features) == len(req_f)
    assert features == req_f


def test_atom_features(example_smiles_n_features):
    """
    Test for atom_features() function
    """
    for smiles in example_smiles_n_features.keys():
        m = Chem.MolFromSmiles(smiles)
        f = []
        for atom in m.GetAtoms():
            features = atom_features(atom)
            f.append(features)
        k = np.array(f)
        req_f = np.array(example_smiles_n_features[smiles])
        assert k.shape == req_f.shape
        assert f == example_smiles_n_features[smiles]
