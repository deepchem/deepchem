import os
import pytest
import torch
import numpy as np
import deepchem as dc


def test_atom_vocab_random_mask():
    from deepchem.models.torch_models.grover import GroverModel
    from deepchem.feat.vocabulary_builders import GroverAtomVocabularyBuilder
    smiles = np.array(['CC', 'CCC'])
    dataset = dc.data.NumpyDataset(X=smiles)

    atom_vocab = GroverAtomVocabularyBuilder()
    atom_vocab.build(dataset)

    vocab_labels = GroverModel.atom_vocab_random_mask(atom_vocab, smiles)
    assert len(vocab_labels) == 5  # 5 atoms


def test_bond_vocab_random_mask():
    from deepchem.models.torch_models.grover import GroverModel
    from deepchem.feat.vocabulary_builders import GroverBondVocabularyBuilder

    smiles = np.array(['CC', 'CCC'])
    dataset = dc.data.NumpyDataset(X=smiles)

    bond_vocab = GroverBondVocabularyBuilder()
    bond_vocab.build(dataset)

    vocab_labels = GroverModel.bond_vocab_random_mask(bond_vocab, smiles)
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


def test_grover_pretraining_task_overfit():
    import deepchem as dc
    from deepchem.feat.vocabulary_builders import (GroverAtomVocabularyBuilder,
                                                   GroverBondVocabularyBuilder)
    from deepchem.models.torch_models.grover import GroverModel
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

    model = GroverModel(node_fdim=151,
                        edge_fdim=165,
                        atom_vocab=av,
                        bond_vocab=bv,
                        features_dim=2048,
                        hidden_size=128,
                        functional_group_size=85,
                        task='pretraining',
                        model_dir='gm')

    # since pretraining is a self-supervision task where labels are generated during
    # preparing batch, we mock _prepare_batch_for_pretraining to set all labels to 0.
    # The test here is checking whether the model predict 0's after overfitting.
    def _prepare_batch_for_pretraining(batch):
        from deepchem.feat.graph_data import BatchGraphData
        from deepchem.utils.grover import extract_grover_attributes
        X, y, w = batch
        batchgraph = BatchGraphData(X[0])
        fgroup_label = getattr(batchgraph, 'fg_labels')

        f_atoms, f_bonds, a2b, b2a, b2revb, a2a, a_scope, b_scope, _, _ = extract_grover_attributes(
            batchgraph)

        atom_vocab_label = torch.zeros(f_atoms.shape[0]).long()
        bond_vocab_label = torch.zeros(f_bonds.shape[0] // 2).long()
        fg_task = torch.zeros(fgroup_label.shape)
        labels = {
            "av_task": atom_vocab_label,
            "bv_task": bond_vocab_label,
            "fg_task": fg_task
        }
        inputs = (f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a)
        return inputs, labels, w

    model._prepare_batch_for_pretraining = _prepare_batch_for_pretraining
    model.fit(graph_data, nb_epoch=1)

    assert np.allclose(preds[0], np.zeros_like(preds[0]))
    assert np.allclose(preds[1], np.zeros_like(preds[1]))
    assert np.allclose(preds[2], np.zeros_like(preds[2]))
    assert np.allclose(preds[3], np.zeros_like(preds[3]))
    assert np.allclose(preds[4], np.zeros_like(preds[4]))
    assert np.allclose(preds[5], np.zeros_like(preds[5]))
    assert np.allclose(preds[6], np.zeros_like(preds[6]))
    assert np.allclose(preds[7], np.zeros_like(preds[7]))
