import pytest
import numpy as np
import os
import deepchem as dc
import shutil
import pickle
try:
    import torch
    from deepchem.models.torch_models.text_cnn import default_dict, TextCNN
    from deepchem.models.torch_models import TextCNNModel
    import torch.nn as nn
    has_torch = True
except ModuleNotFoundError:
    has_torch = False
    pass


@pytest.mark.torch
def test_textcnn_base():
    model = TextCNN(1, default_dict, 1)
    assert model.seq_length == max(model.kernel_sizes)
    large_length = 500
    model = TextCNN(1, default_dict, large_length)
    assert model.seq_length == large_length


@pytest.mark.torch
def test_textcnn_base_forward():
    batch_size = 1
    input_tensor = torch.randint(34, (batch_size, 64))
    cls_model = TextCNN(1, default_dict, 1, mode="classification")
    reg_model = TextCNN(1, default_dict, 1, mode="regression")
    cls_output = cls_model.forward(input_tensor)
    reg_output = reg_model.forward(input_tensor)
    assert len(cls_output) == 2
    assert len(reg_output) == 1
    assert np.allclose(torch.sum(cls_output[0]).item(), 1, rtol=1e-5, atol=1e-6)


@pytest.mark.torch
def test_textcnn_overfit_classification():
    np.random.seed(123)
    n_tasks = 1
    current_dir = os.path.dirname(os.path.abspath(__file__))

    featurizer = dc.feat.RawFeaturizer()
    tasks = ["outcome"]
    input_file = os.path.join(current_dir,
                              "../../tests/assets/example_classification.csv")
    loader = dc.data.CSVLoader(tasks=tasks,
                               feature_field="smiles",
                               featurizer=featurizer)
    dataset = loader.create_dataset(input_file)

    classification_metric = dc.metrics.Metric(dc.metrics.accuracy_score)

    char_dict, length = TextCNNModel.build_char_dict(dataset)
    batch_size = 10

    model = TextCNNModel(n_tasks,
                         char_dict=char_dict,
                         seq_length=length,
                         batch_size=batch_size,
                         learning_rate=0.001,
                         use_queue=False,
                         mode="classification",
                         log_frequency=10)

    # Fit trained model
    model.fit(dataset, nb_epoch=200)

    # Eval model on train
    scores = model.evaluate(dataset, [classification_metric])

    assert scores[classification_metric.name] > .8


@pytest.mark.torch
def test_textcnn_overfit_regression():
    """Test textCNN model overfits tiny data."""
    np.random.seed(123)
    n_tasks = 1
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Load mini log-solubility dataset.
    featurizer = dc.feat.RawFeaturizer()
    tasks = ["outcome"]
    input_file = os.path.join(current_dir,
                              "../../tests/assets/example_regression.csv")
    loader = dc.data.CSVLoader(tasks=tasks,
                               feature_field="smiles",
                               featurizer=featurizer)
    dataset = loader.create_dataset(input_file)

    regression_metric = dc.metrics.Metric(dc.metrics.pearson_r2_score,
                                          task_averager=np.mean)

    char_dict, length = TextCNNModel.build_char_dict(dataset)
    batch_size = 10

    model = TextCNNModel(n_tasks,
                         char_dict,
                         seq_length=length,
                         batch_size=batch_size,
                         learning_rate=0.001,
                         use_queue=False,
                         mode="regression")

    # Fit trained model
    model.fit(dataset, nb_epoch=200)

    # Eval model on train
    scores = model.evaluate(dataset, [regression_metric])

    assert scores[regression_metric.name] > .9


@pytest.mark.torch
def test_textcnn_reload():

    np.random.seed(123)
    n_tasks = 1
    current_dir = os.path.dirname(os.path.abspath(__file__))

    featurizer = dc.feat.RawFeaturizer()
    tasks = ["outcome"]
    input_file = os.path.join(current_dir,
                              "../../tests/assets/example_classification.csv")
    loader = dc.data.CSVLoader(tasks=tasks,
                               feature_field="smiles",
                               featurizer=featurizer)
    dataset = loader.create_dataset(input_file)

    classification_metric = dc.metrics.Metric(dc.metrics.accuracy_score)

    char_dict, length = TextCNNModel.build_char_dict(dataset)
    batch_size = 10
    model_dir = os.path.join(current_dir, "textcnn_tmp")
    os.mkdir(model_dir)

    model = TextCNNModel(n_tasks,
                         char_dict=char_dict,
                         seq_length=length,
                         batch_size=batch_size,
                         learning_rate=0.001,
                         use_queue=False,
                         mode="classification",
                         model_dir=model_dir)

    # Fit trained model
    model.fit(dataset, nb_epoch=200)

    # Eval model on train
    scores = model.evaluate(dataset, [classification_metric])

    reloaded_model = TextCNNModel(n_tasks,
                                  char_dict=char_dict,
                                  seq_length=length,
                                  batch_size=batch_size,
                                  learning_rate=0.001,
                                  use_queue=False,
                                  mode="classification",
                                  model_dir=model_dir)
    reloaded_model.restore()

    reloaded_model.fit(dataset, nb_epoch=200)
    reloaded_scores = reloaded_model.evaluate(dataset, [classification_metric])
    shutil.rmtree(model_dir)

    assert scores == reloaded_scores


@pytest.mark.torch
def test_textcnn_compare_with_tf_impl():
    # Load dataset
    tasks = ["outcome"]
    n_tasks = 1
    current_dir = os.path.dirname(os.path.abspath(__file__))
    featurizer = dc.feat.RawFeaturizer()

    input_file = os.path.join(current_dir,
                              "../../tests/assets/example_regression.csv")
    tensorflow_weights_dir = os.path.join(current_dir, "assets/text_cnn")
    loader = dc.data.CSVLoader(tasks=tasks,
                               feature_field="smiles",
                               featurizer=featurizer)
    dataset = loader.create_dataset(input_file)
    batch_size = 1

    # Intiliaze torch TextCNN
    char_dict, length = TextCNNModel.build_char_dict(dataset)
    torch_model = TextCNNModel(n_tasks,
                               char_dict=char_dict,
                               seq_length=length,
                               batch_size=batch_size,
                               learning_rate=0.001,
                               use_queue=False,
                               mode="regression")

    with torch.no_grad():

        for i, torch_layer in enumerate(torch_model.model.conv_layers):
            assert isinstance(torch_layer, nn.Conv1d)
            with open(
                    os.path.join(tensorflow_weights_dir,
                                 'conv_{}.pickle'.format(i)), 'rb') as f:
                conv_w = pickle.load(f)
            weights_1 = np.transpose(conv_w[0], (2, 1, 0))
            weights_2 = conv_w[1]
            torch_layer.weight.copy_(torch.from_numpy(weights_1))
            torch_layer.bias.copy_(torch.from_numpy(weights_2))

        # Copy other layer weights
        torch_layer_names = ["embedding_layer", "linear1", "linear2", "highway"]
        dense_idx = 0
        for torch_layer_name in torch_layer_names:
            torch_layer = getattr(torch_model.model, torch_layer_name)

            if ("linear" in torch_layer_name):
                with open(
                        os.path.join(tensorflow_weights_dir,
                                     'dense_{}.pickle'.format(dense_idx)),
                        'rb') as f:
                    dense_w = pickle.load(f)
                weights_1 = np.transpose(dense_w[0])
                weights_2 = dense_w[1]

                torch_layer.weight.copy_(torch.from_numpy(weights_1).float())
                torch_layer.bias.copy_(torch.from_numpy(weights_2).float())
                dense_idx += 1

            elif ("embedding" in torch_layer_name):
                with open(
                        os.path.join(tensorflow_weights_dir, 'dtnn_emb.pickle'),
                        'rb') as f:
                    dtnn_w = pickle.load(f)
                torch_layer.embedding_list.data.copy_(
                    torch.from_numpy(dtnn_w).float())

            elif ("highway" in torch_layer_name):
                with open(
                        os.path.join(tensorflow_weights_dir, 'highway.pickle'),
                        'rb') as f:
                    highway_w = pickle.load(f)
                weights_1 = highway_w[0]
                weights_2 = highway_w[1]
                torch_layer.H.weight.copy_(
                    torch.from_numpy(weights_1).float().T)
                torch_layer.H.bias.copy_(torch.from_numpy(weights_2).float())

                weights_3 = highway_w[2]
                weights_4 = highway_w[3]
                torch_layer.T.weight.copy_(
                    torch.from_numpy(weights_3).float().T)
                torch_layer.T.bias.copy_(torch.from_numpy(weights_4).float())

        # Run prediction
        torch_outputs = torch_model.predict(dataset)

        with open(os.path.join(tensorflow_weights_dir, 'tf_outputs.pickle'),
                  'rb') as f:
            tf_outputs = pickle.load(f)
        assert np.allclose(torch_outputs, tf_outputs, rtol=1e-5, atol=1e-6)


@pytest.mark.torch
def test_textcnn_empty_and_short_smiles():
    """Test TextCNN behavior with empty and very short SMILES strings.
    
    This test verifies that the model handles edge cases gracefully:
    - Empty strings
    - Single atom SMILES ("C")
    - Minimal molecules ("CC")
    """
    n_tasks = 1
    seq_length = 50

    model = TextCNNModel(n_tasks,
                         char_dict=default_dict,
                         seq_length=seq_length,
                         mode="classification")

    # Test single atom SMILES - should work
    single_atom_smiles = "C"
    seq = model.smiles_to_seq(single_atom_smiles)
    assert seq is not None
    assert len(seq) == seq_length
    # First element is 0 (start token), second should be 'C' mapping
    assert seq[0] == 0
    assert seq[1] == default_dict['C']

    # Test minimal molecule - should work
    minimal_smiles = "CC"
    seq = model.smiles_to_seq(minimal_smiles)
    assert seq is not None
    assert len(seq) == seq_length
    assert seq[1] == default_dict['C']
    assert seq[2] == default_dict['C']

    # Test empty string - should produce only padding
    empty_smiles = ""
    seq = model.smiles_to_seq(empty_smiles)
    assert seq is not None
    assert len(seq) == seq_length
    # First element is start token (0), rest should be padding ('_')
    assert seq[0] == 0
    for i in range(1, seq_length):
        assert seq[i] == default_dict['_']

    # Test batch processing with short SMILES
    short_smiles_batch = ["C", "CC", "CCC"]
    seqs = model.smiles_to_seq_batch(short_smiles_batch)
    assert seqs.shape == (3, seq_length)


@pytest.mark.torch
def test_textcnn_long_sequences():
    """Test TextCNN with extremely long SMILES sequences.
    
    This test verifies behavior when SMILES strings exceed the seq_length:
    - Sequences at exactly seq_length boundary
    - Sequences exceeding seq_length (should be truncated)
    """
    n_tasks = 1
    seq_length = 20  # Short seq_length to test truncation

    model = TextCNNModel(n_tasks,
                         char_dict=default_dict,
                         seq_length=seq_length,
                         mode="classification")

    # Create a SMILES string that's exactly at the boundary
    # Note: seq_length includes the start token (0), so actual SMILES can be seq_length - 1
    boundary_smiles = "C" * (seq_length - 1)
    seq = model.smiles_to_seq(boundary_smiles)
    assert len(seq) == seq_length

    # Create a very long SMILES string (longer than seq_length)
    # This tests the truncation/padding behavior
    long_smiles = "C" * 100  # Much longer than seq_length
    seq = model.smiles_to_seq(long_smiles)
    # The sequence should still be seq_length (truncated or handled)
    assert len(seq) == seq_length

    # Test with build_char_dict on dataset with long SMILES
    # This verifies the max_length * 1.2 calculation
    long_smiles_list = ["C" * 50, "C" * 100, "C" * 200]
    dataset = dc.data.NumpyDataset(X=np.zeros((3, 1)),
                                   y=np.zeros((3, 1)),
                                   ids=np.array(long_smiles_list))
    char_dict, calculated_length = TextCNNModel.build_char_dict(dataset)
    # Length should be max(200) * 1.2 = 240
    assert calculated_length == int(200 * 1.2)


@pytest.mark.torch
def test_textcnn_invalid_characters():
    """Test TextCNN error handling for invalid SMILES characters.
    
    This test verifies that ValueError is raised with appropriate message
    when SMILES contains characters not in the char_dict.
    """
    n_tasks = 1
    seq_length = 50

    model = TextCNNModel(n_tasks,
                         char_dict=default_dict,
                         seq_length=seq_length,
                         mode="classification")

    # Test with invalid Unicode character
    invalid_unicode_smiles = "C€O"
    with pytest.raises(ValueError, match="character not found in dict"):
        model.smiles_to_seq(invalid_unicode_smiles)

    # Test with uncommon symbols not in default_dict
    invalid_symbol_smiles = "C@#$%O"
    with pytest.raises(ValueError, match="character not found in dict"):
        model.smiles_to_seq(invalid_symbol_smiles)

    # Test with lowercase letters not in dict (e.g., 'x', 'y', 'z')
    invalid_lowercase_smiles = "CxyzO"
    with pytest.raises(ValueError, match="character not found in dict"):
        model.smiles_to_seq(invalid_lowercase_smiles)

    # Test batch processing with one invalid SMILES
    mixed_batch = ["CC", "C€O", "CCC"]
    with pytest.raises(ValueError, match="character not found in dict"):
        model.smiles_to_seq_batch(mixed_batch)

    # Verify that valid SMILES with all default_dict characters work
    valid_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
    seq = model.smiles_to_seq(valid_smiles)
    assert seq is not None
    assert len(seq) == seq_length
