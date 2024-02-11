import pytest
import numpy as np
import deepchem as dc
import os

try:
    import torch
    from deepchem.models.torch_models import TextCNNModel
    from deepchem.models.text_cnn import default_dict
    has_torch = True
except ModuleNotFoundError:
    has_torch = False
    pass
import shutil
import torch.nn as nn
import tensorflow as tf


@pytest.mark.torch
def test_textcnn_module():
    model = TextCNNModel(1, default_dict, 1)
    assert model.seq_length == max(model.kernel_sizes)
    large_length = 500
    model = TextCNNModel(1, default_dict, large_length)
    assert model.seq_length == large_length


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
    ## Load dataset
    tasks = ["outcome"]
    n_tasks = 1
    current_dir = os.path.dirname(os.path.abspath(__file__))
    featurizer = dc.feat.RawFeaturizer()

    input_file = os.path.join(current_dir,
                              "../../tests/assets/example_regression.csv")
    loader = dc.data.CSVLoader(tasks=tasks,
                               feature_field="smiles",
                               featurizer=featurizer)
    dataset = loader.create_dataset(input_file)
    batch_size = 1

    ## Load tensorflow TextCNN checkpoint
    char_dict, length = dc.models.TextCNNModel.build_char_dict(dataset)
    TF_MODEL_CKPT_PATH = os.path.join(current_dir,
                                      "../../tests/assets/TF_text_CNN_reg")
    tf_model = dc.models.TextCNNModel(n_tasks,
                                      char_dict=char_dict,
                                      seq_length=length,
                                      batch_size=batch_size,
                                      learning_rate=0.001,
                                      use_queue=False,
                                      mode="regression")
    tf_model.restore(TF_MODEL_CKPT_PATH)

    ## Intiliaze torch TextCNN
    char_dict, length = TextCNNModel.build_char_dict(dataset)
    torch_model = TextCNNModel(n_tasks,
                               char_dict=char_dict,
                               seq_length=length,
                               batch_size=batch_size,
                               learning_rate=0.001,
                               use_queue=False,
                               mode="regression")

    with torch.no_grad():
        ## Copy conv layer weights
        tf_conv_layers = []
        for layer in tf_model.model.layers:
            if ("conv" in layer.name):
                tf_conv_layers.append(layer)
        for i, (torch_layer, tf_layer) in enumerate(
                zip(torch_model.model.conv_layers, tf_conv_layers)):
            assert isinstance(torch_layer, nn.Conv1d)

            weights_1 = np.transpose(tf_layer.get_weights()[0], (2, 1, 0))
            weights_2 = tf_layer.get_weights()[1]
            torch_layer.weight.copy_(torch.from_numpy(weights_1))
            torch_layer.bias.copy_(torch.from_numpy(weights_2))

        ## Copy other layer weights
        non_conv_layers_tf_torch_name_map = {
            "dtnn_embedding": "embedding_layer",
            "dense": "linear1",
            "dense_1": "linear2",
            "highway": "highway"
        }

        for tf_layer_name, torch_layer_name in non_conv_layers_tf_torch_name_map.items(
        ):
            tf_layer = tf_model.model.get_layer(name=tf_layer_name)
            torch_layer = getattr(torch_model.model, torch_layer_name)

            if ("dense" in tf_layer_name):
                weights_1 = np.transpose(tf_layer.get_weights()[0])
                weights_2 = tf_layer.get_weights()[1]

                torch_layer.weight.copy_(torch.from_numpy(weights_1).float())
                torch_layer.bias.copy_(torch.from_numpy(weights_2).float())

            elif ("dtnn_embedding" in tf_layer_name):
                weights = tf_layer.embedding_list.numpy()
                torch_layer.embedding_list.data.copy_(
                    torch.from_numpy(weights).float())

            elif ("highway" in tf_layer_name):
                weights_1 = tf_layer.get_weights()[0]
                weights_2 = tf_layer.get_weights()[1]
                torch_layer.H.weight.copy_(
                    torch.from_numpy(weights_1).float().T)
                torch_layer.H.bias.copy_(torch.from_numpy(weights_2).float())

                weights_3 = tf_layer.get_weights()[2]
                weights_4 = tf_layer.get_weights()[3]
                torch_layer.T.weight.copy_(
                    torch.from_numpy(weights_3).float().T)
                torch_layer.T.bias.copy_(torch.from_numpy(weights_4).float())

        ## Run prediction
        torch_outputs = torch_model.predict(dataset)
        tf_outputs = tf_model.predict(dataset)

        assert np.allclose(torch_outputs, tf_outputs, rtol=1e-5, atol=1e-6)
