import pytest


@pytest.mark.torch
def testGroverPretrain(grover_graph_attributes):
    from deepchem.models.torch_models.grover import GroverPretrain
    from deepchem.models.torch_models.grover_layers import GroverEmbedding, GroverAtomVocabPredictor, GroverBondVocabPredictor, GroverFunctionalGroupPredictor
    f_atoms, f_bonds, a2b, b2a, b2revb, a2a, a_scope, b_scope, _, _ = grover_graph_attributes
    components = {}
    components['embedding'] = GroverEmbedding(node_fdim=f_atoms.shape[1],
                                              edge_fdim=f_bonds.shape[1])
    components['atom_vocab_task_atom'] = GroverAtomVocabPredictor(
        vocab_size=10, in_features=128)
    components['atom_vocab_task_bond'] = GroverAtomVocabPredictor(
        vocab_size=10, in_features=128)
    components['bond_vocab_task_atom'] = GroverBondVocabPredictor(
        vocab_size=10, in_features=128)
    components['bond_vocab_task_bond'] = GroverBondVocabPredictor(
        vocab_size=10, in_features=128)
    components['functional_group_predictor'] = GroverFunctionalGroupPredictor(
        functional_group_size=10)
    model = GroverPretrain(**components)

    inputs = f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a
    output = model(inputs)
    assert len(output) == 8
    # 9: number of atoms
    assert output[0].shape == (9, 10)
    assert output[1].shape == (9, 10)
    # 6: number of bonds
    assert output[2].shape == (6, 10)
    assert output[3].shape == (6, 10)
    # 3: number of molecules
    assert output[4].shape == (3, 10)
    assert output[5].shape == (3, 10)
    assert output[6].shape == (3, 10)
    assert output[7].shape == (3, 10)


@pytest.mark.torch
def testGroverFinetune(grover_graph_attributes):
    import torch.nn as nn
    from deepchem.models.torch_models.grover_layers import GroverEmbedding
    from deepchem.models.torch_models.readout import GroverReadout
    from deepchem.models.torch_models.grover import GroverFinetune

    f_atoms, f_bonds, a2b, b2a, b2revb, a2a, a_scope, b_scope, fg_labels, additional_features = grover_graph_attributes
    inputs = f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a
    components = {}
    components['embedding'] = GroverEmbedding(node_fdim=f_atoms.shape[1],
                                              edge_fdim=f_bonds.shape[1])
    components['readout'] = GroverReadout(rtype="mean", in_features=128)
    components['mol_atom_from_atom_ffn'] = nn.Linear(
        in_features=additional_features.shape[1] + 128, out_features=1)
    components['mol_atom_from_bond_ffn'] = nn.Linear(
        in_features=additional_features.shape[1] + 128, out_features=1)
    model = GroverFinetune(**components, mode='regression')
    model.training = False
    output = model(inputs, additional_features)
    assert output.shape == (3, 1)
