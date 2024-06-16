import os
import pytest
import numpy as np
import deepchem as dc
from flaky import flaky

try:
    import torch
except ModuleNotFoundError:
    pass


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
def testGroverPretrain(grover_batched_graph):
    from deepchem.models.torch_models.grover import GroverPretrain
    from deepchem.models.torch_models.grover_layers import GroverEmbedding, GroverAtomVocabPredictor, GroverBondVocabPredictor, GroverFunctionalGroupPredictor
    f_atoms, f_bonds, a2b, b2a, b2revb, a2a, a_scope, b_scope, _ = grover_batched_graph.get_components(
    )
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
def test_grover_finetune_regression(grover_batched_graph):
    import torch.nn as nn
    from deepchem.models.torch_models.grover_layers import GroverEmbedding
    from deepchem.models.torch_models.readout import GroverReadout
    from deepchem.models.torch_models.grover import GroverFinetune

    f_atoms, f_bonds, a2b, b2a, b2revb, a2a, a_scope, b_scope, fg_labels = grover_batched_graph.get_components(
    )
    additional_features = grover_batched_graph.additional_features
    inputs = f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a
    components = {}
    components['embedding'] = GroverEmbedding(node_fdim=f_atoms.shape[1],
                                              edge_fdim=f_bonds.shape[1])
    components['readout'] = GroverReadout(rtype="mean", in_features=128)
    components['mol_atom_from_atom_ffn'] = nn.Linear(
        in_features=additional_features.shape[1] + 128, out_features=128)
    components['mol_atom_from_bond_ffn'] = nn.Linear(
        in_features=additional_features.shape[1] + 128, out_features=128)
    model = GroverFinetune(**components, mode='regression', hidden_size=128)
    model.training = False
    output = model((inputs, additional_features))
    assert output.shape == (3, 1)


@pytest.mark.torch
def test_grover_finetune_classification(grover_batched_graph):
    import torch.nn as nn
    from deepchem.models.torch_models.grover_layers import GroverEmbedding
    from deepchem.models.torch_models.readout import GroverReadout
    from deepchem.models.torch_models.grover import GroverFinetune

    f_atoms, f_bonds, a2b, b2a, b2revb, a2a, a_scope, b_scope, fg_labels = grover_batched_graph.get_components(
    )
    additional_features = grover_batched_graph.additional_features
    inputs = f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a
    components = {}
    components['embedding'] = GroverEmbedding(node_fdim=f_atoms.shape[1],
                                              edge_fdim=f_bonds.shape[1])
    components['readout'] = GroverReadout(rtype="mean", in_features=128)
    components['mol_atom_from_atom_ffn'] = nn.Linear(
        in_features=additional_features.shape[1] + 128, out_features=128)
    components['mol_atom_from_bond_ffn'] = nn.Linear(
        in_features=additional_features.shape[1] + 128, out_features=128)
    n_classes = 2
    model = GroverFinetune(**components,
                           mode='classification',
                           n_classes=n_classes,
                           hidden_size=128)
    model.training = False
    output = model((inputs, additional_features))
    assert len(output) == n_classes
    # logits for class 1
    assert output[0].shape == (3, 2)
    # logits for class 2
    assert output[1].shape == (3, 2)


@pytest.mark.torch
def test_grover_pretraining_task_overfit(tmpdir):
    import deepchem as dc
    from deepchem.feat.vocabulary_builders import (GroverAtomVocabularyBuilder,
                                                   GroverBondVocabularyBuilder)
    from deepchem.models.torch_models.grover import GroverModel

    import pandas as pd

    df = pd.DataFrame({'smiles': ['CC'], 'preds': [0]})

    filepath = os.path.join(tmpdir, 'example.csv')
    df.to_csv(filepath, index=False)

    dataset_path = os.path.join(filepath)
    loader = dc.data.CSVLoader(tasks=['preds'],
                               featurizer=dc.feat.DummyFeaturizer(),
                               feature_field=['smiles'])
    dataset = loader.create_dataset(dataset_path)

    av = GroverAtomVocabularyBuilder()
    av.build(dataset)

    bv = GroverBondVocabularyBuilder()
    bv.build(dataset)

    fg = dc.feat.CircularFingerprint()
    loader2 = dc.data.CSVLoader(
        tasks=['preds'],
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
                        device=torch.device('cpu'))

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

        # preparing for test by setting 0 labels
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
    loss = model.fit(graph_data, nb_epoch=200)
    assert loss < 0.1


@flaky(max_runs=4, min_passes=1)
@pytest.mark.torch
def test_grover_model_overfit_finetune(tmpdir):
    from deepchem.models.torch_models.grover import GroverModel
    from deepchem.feat.vocabulary_builders import (GroverAtomVocabularyBuilder,
                                                   GroverBondVocabularyBuilder)
    # arranging test - preparing dataset
    import pandas as pd

    df = pd.DataFrame({'smiles': ['CC', 'CCC'], 'preds': [0, 0]})

    filepath = os.path.join(tmpdir, 'example.csv')
    df.to_csv(filepath, index=False)

    dataset_path = os.path.join(filepath)
    loader = dc.data.CSVLoader(tasks=['preds'],
                               featurizer=dc.feat.DummyFeaturizer(),
                               feature_field=['smiles'])
    dataset = loader.create_dataset(dataset_path)

    av = GroverAtomVocabularyBuilder()
    av.build(dataset)

    bv = GroverBondVocabularyBuilder()
    bv.build(dataset)

    fg = dc.feat.CircularFingerprint()
    loader2 = dc.data.CSVLoader(
        tasks=['preds'],
        featurizer=dc.feat.GroverFeaturizer(features_generator=fg),
        feature_field='smiles')
    graph_data = loader2.create_dataset(dataset_path)

    # acting - tests
    model = GroverModel(node_fdim=151,
                        edge_fdim=165,
                        atom_vocab=av,
                        bond_vocab=bv,
                        features_dim=2048,
                        hidden_size=128,
                        functional_group_size=85,
                        mode='regression',
                        task='finetuning',
                        model_dir='gm_ft',
                        device=torch.device('cpu'))

    loss = model.fit(graph_data, nb_epoch=200)
    scores = model.evaluate(
        graph_data,
        metrics=[dc.metrics.Metric(dc.metrics.mean_squared_error, np.mean)])

    # asserting
    assert loss < 0.01
    assert scores['mean-mean_squared_error'] < 0.01


@pytest.mark.torch
@pytest.mark.parametrize('task', ['pretraining', 'finetuning'])
def test_grover_model_save_restore(tmpdir, task):
    # arranging for tests
    from deepchem.models.torch_models.grover import GroverModel
    from deepchem.feat.vocabulary_builders import (GroverAtomVocabularyBuilder,
                                                   GroverBondVocabularyBuilder)
    atom_vocabulary = GroverAtomVocabularyBuilder(max_size=100)
    bond_vocabulary = GroverBondVocabularyBuilder(max_size=100)

    model_config = {
        'node_fdim': 151,
        'edge_fdim': 165,
        'atom_vocab': atom_vocabulary,
        'bond_vocab': bond_vocabulary,
        'features_dim': 2048,
        'hidden_size': 128,
        'functional_group_size': 85,
        'mode': 'regression',
        'model_dir': tmpdir,
        'task': task
    }

    old_model = GroverModel(**model_config, device=torch.device('cpu'))
    old_model._ensure_built()
    old_model.save_checkpoint()

    new_model = GroverModel(**model_config, device=torch.device('cpu'))
    new_model._ensure_built()
    # checking weights don't match before restore
    old_state = old_model.model.state_dict()
    new_state = new_model.model.state_dict()

    for key in new_state.keys():
        # norm layers and cached zero vectors have constant weights
        if 'norm' not in key and 'zero' not in key:
            assert not torch.allclose(old_state[key], new_state[key])

    # restoring model
    new_model.restore()

    # checking matching of weights after restore
    old_state = old_model.model.state_dict()
    new_state = new_model.model.state_dict()

    for key in new_state.keys():
        assert torch.allclose(old_state[key], new_state[key])


@pytest.mark.torch
def test_load_from_pretrained_embeddings(tmpdir):
    from deepchem.models.torch_models.grover import GroverModel
    from deepchem.feat.vocabulary_builders import (GroverAtomVocabularyBuilder,
                                                   GroverBondVocabularyBuilder)
    atom_vocabulary = GroverAtomVocabularyBuilder(max_size=100)
    bond_vocabulary = GroverBondVocabularyBuilder(max_size=100)

    pretrain_dir = os.path.join(tmpdir, 'pretrain_model')
    model_config = {
        'node_fdim': 151,
        'edge_fdim': 165,
        'atom_vocab': atom_vocabulary,
        'bond_vocab': bond_vocabulary,
        'features_dim': 2048,
        'hidden_size': 128,
        'functional_group_size': 85,
        'mode': 'regression',
        'model_dir': pretrain_dir,
    }
    model_config['task'] = 'pretraining'

    pretrain_model = GroverModel(**model_config, device=torch.device('cpu'))
    pretrain_model._ensure_built()
    pretrain_model.save_checkpoint()

    model_config['task'] = 'finetuning'
    model_config['model_dir'] = os.path.join(tmpdir, 'finetune_model')

    finetune_model = GroverModel(**model_config, device=torch.device('cpu'))
    finetune_model._ensure_built()

    pm_e_sdict = pretrain_model.model.embedding.state_dict()
    fm_e_sdict = finetune_model.model.embedding.state_dict()

    # asserting that weights are not same before reloading
    for key in pm_e_sdict.keys():
        # notm and bias layers have constant weights, hence they are not checked
        if 'norm' not in key and 'bias' not in key:
            assert not torch.allclose(pm_e_sdict[key], fm_e_sdict[key])

    # acting - loading pretrained weights
    finetune_model.load_from_pretrained(source_model=pretrain_model,
                                        components=['embedding'])

    fm_pretrained_e_sdict = finetune_model.model.embedding.state_dict()

    # asserting that weight matches after loading
    for key in pm_e_sdict.keys():
        assert torch.allclose(pm_e_sdict[key], fm_pretrained_e_sdict[key])
