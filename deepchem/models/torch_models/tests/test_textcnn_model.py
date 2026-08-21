import pytest
import numpy as np
import os
import deepchem as dc
import shutil
import pickle
try:
    import torch
    from deepchem.models.torch_models.text_cnn import default_dict, TextCNN, TextCNNModel
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

    This test verifies that the model can handle edge cases including:
    - Empty strings
    - Single character SMILES
    - Very short molecules

    The model should handle these gracefully due to seq_length protection
    (seq_length = max(calculated_length, max_kernel_size)) and padding.
    """
    np.random.seed(123)
    n_tasks = 1

    # Create dataset with empty and very short SMILES
    # Empty string, single atom, two atoms, three atoms
    smiles = ["", "C", "CC", "CCO"]
    # Add some labels for each SMILES (arbitrary values for testing)
    labels = np.array([[0.0], [1.0], [0.0], [1.0]])

    dataset = dc.data.NumpyDataset(X=np.array(smiles), y=labels, ids=smiles)

    # Build char_dict - should handle empty and short SMILES
    char_dict, length = TextCNNModel.build_char_dict(dataset)

    # Note: build_char_dict returns calculated length (max_len * 1.2)
    # The TextCNN constructor will ensure seq_length >= max(kernel_sizes)

    batch_size = 4
    model = TextCNNModel(n_tasks,
                         char_dict=char_dict,
                         seq_length=length,
                         batch_size=batch_size,
                         learning_rate=0.001,
                         use_queue=False,
                         mode="regression")

    # Verify the model's seq_length is at least max kernel size (20)
    assert model.model.seq_length >= 20, f"Model seq_length should be at least 20, got {model.model.seq_length}"

    # Make predictions - should work without errors
    predictions = model.predict(dataset)

    # Verify output shape (n_samples, n_tasks, 1) for regression
    assert predictions.shape == (
        4, 1, 1), f"Expected shape (4, 1, 1), got {predictions.shape}"

    # Verify predictions are valid numbers (not NaN or Inf)
    assert np.all(
        np.isfinite(predictions)), "Predictions contain NaN or Inf values"


@pytest.mark.torch
def test_textcnn_long_sequences():
    """Test TextCNN with extremely long SMILES sequences.

    This test verifies the model's behavior with SMILES that are much longer
    than typical molecules. The model should handle these either by proper
    padding/truncation or by raising an informative error.
    """
    np.random.seed(123)
    n_tasks = 1

    # Create a very long SMILES string (simulating large polymers/dendrimers)
    # Using valid SMILES characters to create a long sequence
    normal_smiles = "CCO"
    # Create a SMILES with 1000+ characters
    long_smiles = "C" * 500 + "N" * 500

    smiles = [normal_smiles, long_smiles]
    labels = np.array([[0.0], [1.0]])

    dataset = dc.data.NumpyDataset(X=np.array(smiles), y=labels, ids=smiles)

    # Build char_dict with long sequence
    char_dict, length = TextCNNModel.build_char_dict(dataset)

    # Verify seq_length was calculated with the long SMILES
    # Should be approximately 1000 * 1.2 = 1200
    assert length > 1000, f"seq_length should account for long SMILES, got {length}"

    batch_size = 2
    model = TextCNNModel(n_tasks,
                         char_dict=char_dict,
                         seq_length=length,
                         batch_size=batch_size,
                         learning_rate=0.001,
                         use_queue=False,
                         mode="regression")

    # Make predictions - should work without errors
    predictions = model.predict(dataset)

    # Verify output shape (n_samples, n_tasks, 1) for regression
    assert predictions.shape == (
        2, 1, 1), f"Expected shape (2, 1, 1), got {predictions.shape}"

    # Verify predictions are valid numbers
    assert np.all(
        np.isfinite(predictions)), "Predictions contain NaN or Inf values"


@pytest.mark.torch
def test_textcnn_invalid_characters():
    """Test TextCNN error handling for invalid SMILES characters.

    This test verifies two scenarios:
    1. Invalid characters in training data get added to char_dict automatically
    2. Characters not in char_dict during inference raise ValueError

    The second scenario tests the error path at line 414 in text_cnn.py which
    was previously untested.
    """
    np.random.seed(123)
    n_tasks = 1

    # Scenario 1: Invalid characters in training data
    # These should be automatically added to char_dict
    train_smiles = ["CCO", "C=O", "CC(C)O"]
    train_labels = np.array([[0.0], [1.0], [0.0]])
    train_dataset = dc.data.NumpyDataset(X=np.array(train_smiles),
                                         y=train_labels,
                                         ids=train_smiles)

    # Build char_dict from training data
    char_dict, length = TextCNNModel.build_char_dict(train_dataset)

    batch_size = 3
    model = TextCNNModel(n_tasks,
                         char_dict=char_dict,
                         seq_length=length,
                         batch_size=batch_size,
                         learning_rate=0.001,
                         use_queue=False,
                         mode="regression")

    # This should work fine
    predictions = model.predict(train_dataset)
    assert predictions.shape == (3, 1, 1)

    # Scenario 2: New invalid character during inference
    # Create a SMILES with a character not in the training char_dict
    # Using a Unicode character that's unlikely to be in standard SMILES
    test_smiles_invalid = ["C☺O"]  # ☺ not in char_dict
    test_labels = np.array([[0.0]])
    test_dataset = dc.data.NumpyDataset(X=np.array(test_smiles_invalid),
                                        y=test_labels,
                                        ids=test_smiles_invalid)

    # This should raise ValueError because ☺ is not in char_dict
    with pytest.raises(ValueError, match="character not found in dict"):
        model.predict(test_dataset)
