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
