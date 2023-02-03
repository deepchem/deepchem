"""
Test for bond feature vector generator and reactions mapping
"""

from deepchem.feat.molecule_featurizers.dmpnn_featurizer import bond_features, map_reac_to_prod
from rdkit import Chem
import pytest
import numpy as np


@pytest.fixture
def example_smiles_n_b_features():
    """
    Sample data for testing

    Returns
    -------
    dictionary
    format {'smiles':required feature vector}
    """
    feature_vector_C1OC1 = [[
        0, True, False, False, False, False, True, 1, 0, 0, 0, 0, 0, 0
    ], [0, True, False, False, False, False, True, 1, 0, 0, 0, 0, 0,
        0], [0, True, False, False, False, False, True, 1, 0, 0, 0, 0, 0, 0]]
    feature_vector_NN = [[
        0, False, False, True, False, False, False, 1, 0, 0, 0, 0, 0, 0
    ]]
    return {'C1OC1': feature_vector_C1OC1, 'N#N': feature_vector_NN}


def test_bond_features_none():
    """
    Test for bond_features() with 'None' input for bond
    """
    f_bond = bond_features(None)
    req_f = list(np.zeros((14,), dtype=int))
    req_f[0] = 1
    assert len(f_bond) == len(req_f)
    assert f_bond == req_f


def test_bond_features(example_smiles_n_b_features):
    """
    Test for bond_features() function
    """
    for smiles in example_smiles_n_b_features.keys():
        b_f = []
        m = Chem.MolFromSmiles(smiles)
        for b in m.GetBonds():
            b_f.append(bond_features(b))
        print(b_f)
        k = np.array(b_f)
        req_f = np.array(example_smiles_n_b_features[smiles])
        assert k.shape == req_f.shape
        assert b_f == example_smiles_n_b_features[smiles]


def test_reaction_mapping():
    """
    Test for map_reac_to_prod() function
    """

    def mappings(r_smile, p_smile):
        """
        Function to return mappings based on given reactant and product smiles
        """
        r_mol = Chem.MolFromSmiles(r_smile)
        p_mol = Chem.MolFromSmiles(p_smile)
        return map_reac_to_prod(r_mol, p_mol)

    # both reactant and product are null
    r_smile = ''
    p_smile = ''
    assert mappings(r_smile, p_smile) == ({}, [], [])

    # reactant is null
    r_smile = ''
    p_smile = '[H:6][CH2:1][CH:2]=[CH:3][CH:4]=[CH2:5]'
    assert mappings(r_smile, p_smile) == ({}, [0, 1, 2, 3, 4], [])

    # product is null
    r_smile = '[CH2:1]=[CH:2][CH:3]=[CH:4][CH2:5][H:6]'
    p_smile = ''
    assert mappings(r_smile, p_smile) == ({}, [], [0, 1, 2, 3, 4])

    # valid reaction: [CH2:1]=[CH:2][CH:3]=[CH:4][CH2:5][H:6]>> [H:6][CH2:1][CH:2]=[CH:3][CH:4]=[CH2:5]
    r_smile = '[CH2:1]=[CH:2][CH:3]=[CH:4][CH2:5][H:6]'
    p_smile = '[H:6][CH2:1][CH:2]=[CH:3][CH:4]=[CH2:5]'
    assert mappings(r_smile, p_smile) == ({
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 4
    }, [], [])
