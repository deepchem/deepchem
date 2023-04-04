import os
import pytest
import numpy as np
import deepchem as dc


def test_atom_random_mask():
    from deepchem.models.torch_models.grover import GroverModel
    from deepchem.feat.vocabulary_builders import GroverAtomVocabularyBuilder
    smiles = np.array(['CC', 'CCC'])
    dataset = dc.data.NumpyDataset(X=smiles)

    atom_vocab = GroverAtomVocabularyBuilder()
    atom_vocab.build(dataset)

    vocab_labels = GroverModel.atom_random_mask(atom_vocab, smiles)
    assert len(vocab_labels) == 5  # 5 atoms


def test_bond_random_mask():
    from deepchem.models.torch_models.grover import GroverModel
    from deepchem.feat.vocabulary_builders import GroverBondVocabularyBuilder

    smiles = np.array(['CC', 'CCC'])
    dataset = dc.data.NumpyDataset(X=smiles)

    bond_vocab = GroverBondVocabularyBuilder()
    bond_vocab.build(dataset)

    vocab_labels = GroverModel.bond_random_mask(bond_vocab, smiles)
    assert len(vocab_labels) == 3  # 3 bonds


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


def test_grover_overfit():
    import deepchem as dc
    from deepchem.feat.vocabulary_builders import (GroverAtomVocabularyBuilder,
                                                   GroverBondVocabularyBuilder)
    from deepchem.models.torch_models.grover import Grover
    dataset_path = os.path.join(os.path.dirname(__file__),
                                '../../tests/assets/example.csv')
    loader = dc.data.CSVLoader(tasks=['log-solubility'],
                               featurizer=dc.feat.DummyFeaturizer(),
                               feature_field=['smiles'])
    dataset = loader.create_dataset(dataset_path)

    av = GroverAtomVocabularyBuilder()
    av.build(dataset)

    bv = GroverBondVocabularyBuilder()
    bv.build(dataset)

    fg = dc.feat.CircularFingerprint()
    loader2 = dc.data.CSVLoader(
        tasks=['log-solubility'],
        featurizer=dc.feat.GroverFeaturizer(features_generator=fg),
        feature_field='smiles')
    graph_data = loader2.create_dataset(dataset_path)

    model = Grover(node_fdim=151,
                   edge_fdim=165,
                   atom_vocab=av,
                   bond_vocab=bv,
                   atom_vocab_size=300,
                   bond_vocab_size=300,
                   hidden_size=128,
                   functional_group_size=85,
                   mode='regression',
                   task='pretraining')

    model.fit(graph_data, nb_epoch=1)
    metric = dc.metrics.Metric(dc.metrics.mean_squared_error)
    scores = model.evaluate(graph_data, [metric])
    assert scores['mean_squared_error'] < 0.1
>>>>>>> 3af36497 (grover layer tests [skip ci])
