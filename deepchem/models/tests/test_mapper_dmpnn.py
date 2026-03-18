"""
Test for `_MapperDMPNN` helper class for DMPNN model
"""

from deepchem.feat import DMPNNFeaturizer
import numpy as np
import pytest

try:
    from deepchem.models.torch_models.dmpnn import _MapperDMPNN
    has_torch = True
except:
    has_torch = False

# Set up tests.
smiles_list = ["C", "CC", "CCC", "C1=CC=CC=C1", "[I-].[K+]"]
featurizer = DMPNNFeaturizer(use_original_atom_ranks=True,
                             features_generators=['morgan'])
graphs = featurizer.featurize(smiles_list)
benezene_atom_to_incoming_bonds: np.ndarray = np.asarray([[1, 10], [0,
                                                                    3], [2, 5],
                                                          [4, 7], [6, 9],
                                                          [8, 11]])
benezene_mapping: np.ndarray = np.asarray([[-1, 10], [-1, 3], [0, -1], [-1, 5],
                                           [2, -1], [-1, 7], [4, -1], [-1, 9],
                                           [6, -1], [-1, 11], [8, -1], [1, -1],
                                           [-1, -1]])


@pytest.mark.torch
def test_mapper_general_attributes():
    """
  General tests for the mapper class
  """
    for graph in graphs:
        mapper = _MapperDMPNN(graph)
        assert (mapper.atom_features == graph.node_features).all()
        assert (mapper.bond_features == graph.edge_features).all()
        assert (mapper.bond_index == graph.edge_index).all()
        assert (mapper.global_features == graph.global_features).all()
        concat_feature_dim = graph.num_node_features + graph.num_edge_features
        assert mapper.f_ini_atoms_bonds.shape == (graph.num_edges + 1,
                                                  concat_feature_dim)
        assert (
            mapper.f_ini_atoms_bonds[-1] == np.zeros(concat_feature_dim)).all()
        assert len(mapper.f_ini_atoms_bonds) == len(mapper.mapping)
        assert (mapper.values[0] == mapper.atom_features).all()
        assert (mapper.values[1] == mapper.f_ini_atoms_bonds).all()
        assert (mapper.values[2] == mapper.atom_to_incoming_bonds).all()
        assert (mapper.values[3] == mapper.mapping).all()
        assert (mapper.values[4] == mapper.global_features).all()


@pytest.mark.torch
def test_mapper_no_bond():
    """
  Test 'C' in _MapperDMPNN (no bond present)
  """
    mapper = _MapperDMPNN(graphs[0])
    assert (mapper.bond_to_ini_atom == np.empty(0)).all()
    assert (mapper.atom_to_incoming_bonds == np.asarray([[-1]])).all()
    assert (mapper.mapping == np.asarray([[-1]])).all()


@pytest.mark.torch
def test_mapper_two_directed_bonds_btw_two_atoms():
    """
  Test 'CC' in _MapperDMPNN (1 bond present (2 directed))
  """
    mapper = _MapperDMPNN(graphs[1])
    assert (mapper.bond_to_ini_atom == np.asarray([0, 1])).all()
    assert (mapper.atom_to_incoming_bonds == np.asarray([[1], [0]])).all()
    assert (mapper.mapping == np.asarray([[-1], [-1], [-1]])).all()


@pytest.mark.torch
def test_mapper_two_adjacent_bonds():
    """
  Test 'CCC' in _MapperDMPNN (2 adjacent bonds present (4 directed))
  """
    mapper = _MapperDMPNN(graphs[2])
    assert (mapper.bond_to_ini_atom == np.asarray([0, 1, 1, 2])).all()
    print(mapper.atom_to_incoming_bonds)
    assert (mapper.atom_to_incoming_bonds == np.asarray([[1, -1], [0, 3],
                                                         [2, -1]])).all()
    assert (mapper.mapping == np.asarray([[-1, -1], [-1, 3], [0, -1], [-1, -1],
                                          [-1, -1]])).all()


@pytest.mark.torch
def test_mapper_ring():
    """
  Test 'C1=CC=CC=C1' in _MapperDMPNN (benezene ring)
  """
    mapper = _MapperDMPNN(graphs[3])
    assert (mapper.bond_to_ini_atom == np.asarray(
        [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 0])).all()
    assert (
        mapper.atom_to_incoming_bonds == benezene_atom_to_incoming_bonds).all()
    assert (mapper.mapping == benezene_mapping).all()


@pytest.mark.torch
def test_mapper_disconnected_compounds():
    """
  Test '[I-].[K+]' in _MapperDMPNN (disconnected compounds)
  """
    mapper = _MapperDMPNN(graphs[4])
    assert (mapper.bond_to_ini_atom == np.empty(0)).all()
    assert (mapper.atom_to_incoming_bonds == np.asarray([[-1], [-1]])).all()
    assert (mapper.mapping == np.asarray([[-1]])).all()
