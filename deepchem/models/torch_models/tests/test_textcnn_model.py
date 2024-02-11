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

torch.set_default_dtype(torch.float32)


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
    model_dir = os.path.join(current_dir, "textcnn_tmp_reload")
    os.mkdir(model_dir)

    model = TextCNNModel(n_tasks,
                         char_dict=char_dict,
                         seq_length=length,
                         batch_size=batch_size,
                         learning_rate=0.001,
                         use_queue=False,
                         mode="classification",
                         model_dir=model_dir)

    print("Model dir: ", model.model_dir)
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
    print("checkpoints: ", reloaded_model.get_checkpoints())
    reloaded_model.restore()

    reloaded_model.fit(dataset, nb_epoch=200)
    reloaded_scores = reloaded_model.evaluate(dataset, [classification_metric])
    shutil.rmtree(model_dir)

    assert scores == reloaded_scores


def train_save_tf_weights():

    np.random.seed(123)
    n_tasks = 1
    current_dir = os.path.dirname(os.path.abspath(__file__))

    featurizer = dc.feat.RawFeaturizer()
    tasks = ["outcome"]
    input_file = os.path.join(current_dir,
                              "../../tests/assets/example_regression.csv")
    loader = dc.data.CSVLoader(tasks=tasks,
                               feature_field="smiles",
                               featurizer=featurizer)
    dataset = loader.create_dataset(input_file)

    classification_metric = dc.metrics.Metric(dc.metrics.accuracy_score)

    char_dict, length = dc.models.TextCNNModel.build_char_dict(dataset)
    batch_size = 10
    model_dir = os.path.join(current_dir, "textcnn_tmp")
    os.mkdir(model_dir)

    model = dc.models.TextCNNModel(n_tasks,
                                   char_dict=char_dict,
                                   seq_length=length,
                                   batch_size=batch_size,
                                   learning_rate=0.001,
                                   use_queue=False,
                                   mode="regression",
                                   model_dir=model_dir)
    model.fit(dataset, nb_epoch=200)
    return model


@pytest.mark.torch
def test_textcnn_compare_with_tf_impl():
    torch.set_default_dtype(torch.float32)
    torch.set_printoptions(precision=9)
    tf.keras.backend.set_floatx('float32')

    SAVE_TF_WEIGHTS = False
    tasks = ["outcome"]

    current_dir = os.path.dirname(os.path.abspath(__file__))
    featurizer = dc.feat.RawFeaturizer()

    input_file = os.path.join(current_dir,
                              "../../tests/assets/example_regression.csv")
    loader = dc.data.CSVLoader(tasks=tasks,
                               feature_field="smiles",
                               featurizer=featurizer)
    dataset = loader.create_dataset(input_file)
    batch_size = 1

    np.random.seed(123)
    if (SAVE_TF_WEIGHTS):
        tf_model = train_save_tf_weights()
        print("SAVED TF weights")
    else:
        n_tasks = 1

        # classification_metric = dc.metrics.Metric(dc.metrics.accuracy_score)

        char_dict, length = dc.models.TextCNNModel.build_char_dict(dataset)
        tf_char_dict = char_dict.copy()
        print("TF CHAR DICT")
        print(length)
        print(char_dict)
        model_dir = os.path.join(current_dir, "textcnn_tmp")

        tf_model = dc.models.TextCNNModel(n_tasks,
                                          char_dict=char_dict,
                                          seq_length=length,
                                          batch_size=batch_size,
                                          learning_rate=0.001,
                                          use_queue=False,
                                          mode="regression",
                                          model_dir=model_dir)
        tf_model.restore(
            "/home/shiva/projects/deepchem/deepchem/models/torch_models/tests/textcnn_tmp/ckpt-1"
        )
        # tf_scores = tf_model.evaluate(dataset, [classification_metric])
        # assert tf_scores[classification_metric.name] > .8

    n_tasks = 1

    # classification_metric = dc.metrics.Metric(dc.metrics.accuracy_score)
    char_dict, length = TextCNNModel.build_char_dict(dataset)
    print("Torch CHAR DICT")
    print(tf_char_dict == char_dict)
    print(length)
    print(char_dict)
    torch_model = TextCNNModel(n_tasks,
                               char_dict=char_dict,
                               seq_length=length,
                               batch_size=batch_size,
                               learning_rate=0.001,
                               use_queue=False,
                               mode="regression")

    tf_conv_layers = []

    for layer in tf_model.model.layers:
        if ("conv" in layer.name):
            tf_conv_layers.append(layer)

    with torch.no_grad():
        for i, (torch_layer, tf_layer) in enumerate(
                zip(torch_model.model.conv_layers, tf_conv_layers)):
            assert isinstance(torch_layer, nn.Conv1d)

            # Transpose weights to match the PyTorch convention
            weights_1 = np.transpose(tf_layer.get_weights()[0], (2, 1, 0))
            # return
            weights_2 = tf_layer.get_weights()[1]

            # Load weights into PyTorch layer
            # print(weights_1)
            # print(torch.from_numpy(weights_1.astype(np.float64)))
            torch_layer.weight.copy_(torch.from_numpy(weights_1))
            torch_layer.bias.copy_(torch.from_numpy(weights_2))

        print("Copied conv weights")

        non_conv_layers_tf_torch_name_map = {
            "dtnn_embedding": "embedding_layer",
            "dense": "linear1",
            "dense_1": "linear2",
            "highway": "highway"
        }

        print("!!!!!!!!!\n\n\n")
        for tf_layer_name, torch_layer_name in non_conv_layers_tf_torch_name_map.items(
        ):
            tf_layer = tf_model.model.get_layer(name=tf_layer_name)
            torch_layer = getattr(torch_model.model, torch_layer_name)

            if ("dense" in tf_layer_name):
                weights_1 = np.transpose(tf_layer.get_weights()[0])
                weights_2 = tf_layer.get_weights()[1]

                # Load weights into PyTorch layer
                torch_layer.weight.copy_(torch.from_numpy(weights_1).float())
                torch_layer.bias.copy_(torch.from_numpy(weights_2).float())
                print("Loaded Dense")

            elif ("dtnn_embedding" in tf_layer_name):
                weights = tf_layer.embedding_list.numpy()
                torch_layer.embedding_list.data.copy_(
                    torch.from_numpy(weights).float())
                print("Loaded DTNN weights")
            elif ("highway" in tf_layer_name):
                weights_1 = tf_layer.get_weights()[0]
                weights_2 = tf_layer.get_weights()[1]
                torch_layer.H.weight.copy_(torch.from_numpy(weights_1).float().T)
                torch_layer.H.bias.copy_(torch.from_numpy(weights_2).float())
                # Transfer weights for T
                weights_3 = tf_layer.get_weights()[2]
                weights_4 = tf_layer.get_weights()[3]
                torch_layer.T.weight.copy_(torch.from_numpy(weights_3).float().T)
                torch_layer.T.bias.copy_(torch.from_numpy(weights_4).float())
                print("Loaded Highway weights")

        layer_name_map = {
            "dtnn_embedding": "embedding_layer",
            "dense": "linear1",
            "dense_1": "linear2",
            "highway": "highway"
        }
        tf_conv_weigts = []
        torch_conv_weights = []
        # Iterate through layers and compare weights
        for tf_layer_name, torch_layer_name in layer_name_map.items():
            # Get weights from TensorFlow model
            # if ("highway" not in tf_layer_name):
            #     continue
            tf_weights = tf_model.model.get_layer(
                name=tf_layer_name).get_weights()

            # Get weights from PyTorch model
            torch_layer = getattr(torch_model.model, torch_layer_name)
            if "embedding" in torch_layer_name:
                import pickle
                print("CHEKING EMBEDDING")
                torch_weights = [torch_layer.embedding_list.data.numpy()]
                with open(
                        "/home/shiva/projects/deepchem/deepchem/models/torch_models/tests/tf_output/torch_dtnn_weight.pickle",
                        "wb") as fp:
                    pickle.dump(torch_weights, fp)
                with open(
                        "/home/shiva/projects/deepchem/deepchem/models/torch_models/tests/tf_output/tf_dtnn_weight.pickle",
                        "wb") as fp:
                    pickle.dump(tf_weights, fp)
            elif "highway" in torch_layer_name:
                # Access parameters of the HighwayLayer
                torch_weights = [
                    torch_layer.H.weight.data.numpy().T,
                    torch_layer.H.bias.data.numpy(),
                    torch_layer.T.weight.data.numpy().T,
                    torch_layer.T.bias.data.numpy(),
                ]
                with open(
                        "/home/shiva/projects/deepchem/deepchem/models/torch_models/tests/tf_output/torch_highway_weights.pickle",
                        "wb") as fp:
                    pickle.dump(torch_weights, fp)
                with open(
                        "/home/shiva/projects/deepchem/deepchem/models/torch_models/tests/tf_output/tf_highway_weights.pickle",
                        "wb") as fp:
                    pickle.dump(tf_weights, fp)

            # elif("conv" in torch_layer_name):
            #     torch_weights = [
            #         torch_layer.weight.data.numpy().T,
            #         torch_layer.bias.data.numpy()
            #     ]
            #     torch_conv_weights.append(torch_weights)
            #     tf_conv_weigts.append(tf_weights)

            else:
                torch_weights = [
                    torch_layer.weight.data.numpy().T,
                    torch_layer.bias.data.numpy()
                ]

            # Compare weights
            for tf_w, torch_w in zip(tf_weights, torch_weights):
                assert np.allclose(
                    tf_w, torch_w, rtol=1e-5,
                    atol=1e-8), f"Weights mismatch in layer {tf_layer_name}"
        import pickle

        print("WEIGHTS MATCH")

        for conv_layer in torch_model.model.conv_layers:
            torch_weights = [
                conv_layer.weight.data.numpy().T,
                conv_layer.bias.data.numpy()
            ]
            torch_conv_weights.append(torch_weights)

        with open(
                "/home/shiva/projects/deepchem/deepchem/models/torch_models/tests/tf_output/torch_conv_weights.pickle",
                "wb") as fp:
            pickle.dump(torch_conv_weights, fp)

        tf_conv_layer_names = ["conv1d_{}".format(i) for i in range(1, 12)]
        tf_conv_layer_names = ["conv1d"] + tf_conv_layer_names

        for tf_conv_laber_name in tf_conv_layer_names:
            tf_weights = tf_model.model.get_layer(
                name=tf_conv_laber_name).get_weights()
            tf_conv_weigts.append(tf_weights)

        with open(
                "/home/shiva/projects/deepchem/deepchem/models/torch_models/tests/tf_output/tf_conv_weights.pickle",
                "wb") as fp:
            pickle.dump(tf_conv_weigts, fp)

        # print(torch_model.model.conv_layers[0].weight.data.numpy().T)
        # print(tf_model.model.layers[2].get_weights()[0])
        with torch.no_grad():
            torch_outputs = torch_model.predict(dataset)
        # print(torch_outputs.shape)

        tf_outputs = tf_model.predict(
            dataset)  # ,, output_types=['prediction', "dtnn_embedding"]
        # print(tf_outputs.shape)

        print("TORCH OUPUTS: ")
        print(torch_outputs)
        print("TF OUTPUTS:")
        print(tf_outputs)

        assert np.allclose(torch_outputs, tf_outputs, rtol=1e-5, atol=1e-6)
    """
    print in beteeen layer ouptus
    could be highway, that transpose could mess up
    """


# @pytest.mark.torch
# def test_textcnn_compare_with_tf_impl():
#     SAVE_TF_WEIGHTS = False
#     np.random.seed(123)
#     if (SAVE_TF_WEIGHTS):
#         tf_model = train_save_tf_weights()
#         print("SAVED TF weights")
#     else:
#         n_tasks = 1
#         current_dir = os.path.dirname(os.path.abspath(__file__))

#         featurizer = dc.feat.RawFeaturizer()
#         tasks = ["outcome"]
#         input_file = os.path.join(
#             current_dir, "../../tests/assets/example_classification.csv")
#         loader = dc.data.CSVLoader(tasks=tasks,
#                                    feature_field="smiles",
#                                    featurizer=featurizer)
#         dataset = loader.create_dataset(input_file)
#         # classification_metric = dc.metrics.Metric(dc.metrics.accuracy_score)

#         char_dict, length = dc.models.TextCNNModel.build_char_dict(dataset)
#         batch_size = 10
#         model_dir = os.path.join(current_dir, "textcnn_tmp")

#         tf_model = dc.models.TextCNNModel(n_tasks,
#                                           char_dict=char_dict,
#                                           seq_length=length,
#                                           batch_size=batch_size,
#                                           learning_rate=0.001,
#                                           use_queue=False,
#                                           mode="classification",
#                                           model_dir=model_dir)
#         tf_model.restore(
#             "/home/shiva/projects/deepchem/deepchem/models/torch_models/tests/textcnn_tmp/ckpt-1"
#         )
#         # tf_scores = tf_model.evaluate(dataset, [classification_metric])
#         # assert tf_scores[classification_metric.name] > .8

#     n_tasks = 1

#     featurizer = dc.feat.RawFeaturizer()
#     tasks = ["outcome"]
#     input_file = os.path.join(current_dir,
#                               "../../tests/assets/example_classification.csv")
#     loader = dc.data.CSVLoader(tasks=tasks,
#                                feature_field="smiles",
#                                featurizer=featurizer)
#     dataset = loader.create_dataset(input_file)
#     classification_metric = dc.metrics.Metric(dc.metrics.accuracy_score)
#     char_dict, length = TextCNNModel.build_char_dict(dataset)
#     batch_size = 10

#     torch_model = TextCNNModel(n_tasks,
#                                char_dict=char_dict,
#                                seq_length=length,
#                                batch_size=batch_size,
#                                learning_rate=0.001,
#                                use_queue=False,
#                                mode="classification",
#                                log_frequency=10)

#     tf_conv_layers = []

#     for layer in tf_model.model.layers:
#         if ("conv" in layer.name):
#             tf_conv_layers.append(layer)

#     with torch.no_grad():
#         for i, (torch_layer, tf_layer) in enumerate(
#                 zip(torch_model.model.conv_layers, tf_conv_layers)):
#             assert isinstance(torch_layer, nn.Conv1d)

#             # Transpose weights to match the PyTorch convention
#             weights_1 = np.transpose(tf_layer.get_weights()[0], (2, 1, 0))
#             weights_2 = tf_layer.get_weights()[1]

#             # Load weights into PyTorch layer
#             torch_layer.weight.copy_(torch.from_numpy(weights_1).float())
#             torch_layer.bias.copy_(torch.from_numpy(weights_2).float())

#         print("Copied conv weights")

#         non_conv_layers_tf_torch_name_map = {
#             "dtnn_embedding": "embedding_layer",
#             "dense": "linear1",
#             "dense_1": "linear2",
#             "highway": "highway"
#         }
#         print("!!!!!!!!!\n\n\n")
#         for tf_layer_name, torch_layer_name in non_conv_layers_tf_torch_name_map.items(
#         ):
#             tf_layer = tf_model.model.get_layer(name=tf_layer_name)
#             torch_layer = getattr(torch_model.model, torch_layer_name)

#             if ("dense" in tf_layer_name):
#                 weights_1 = np.transpose(tf_layer.get_weights()[0])
#                 weights_2 = tf_layer.get_weights()[1]

#                 # Load weights into PyTorch layer
#                 torch_layer.weight.copy_(torch.from_numpy(weights_1).float())
#                 torch_layer.bias.copy_(torch.from_numpy(weights_2).float())
#                 print("Loaded Dense")

#             elif ("dtnn_embedding" in tf_layer_name):
#                 weights = tf_layer.embedding_list.numpy()
#                 torch_layer.embedding_list.data.copy_(
#                     torch.from_numpy(weights).float())
#                 print("Loaded DTNN weights")
#             elif ("highway" in tf_layer_name):
#                 weights_1 = tf_layer.get_weights()[0]
#                 weights_2 = tf_layer.get_weights()[1]
#                 torch_layer.H.weight.copy_(torch.from_numpy(weights_1).float())
#                 torch_layer.H.bias.copy_(torch.from_numpy(weights_2).float())
#                 # Transfer weights for T
#                 weights_3 = tf_layer.get_weights()[2]
#                 weights_4 = tf_layer.get_weights()[3]
#                 torch_layer.T.weight.copy_(torch.from_numpy(weights_3).float())
#                 torch_layer.T.bias.copy_(torch.from_numpy(weights_4).float())
#                 print("Loaded Highway weights")

#     layer_name_map = {
#         "dtnn_embedding": "embedding_layer",
#         "dense": "linear1",
#         "dense_1": "linear2",
#         "highway": "highway"
#     }

#     # Iterate through layers and compare weights
#     for tf_layer_name, torch_layer_name in layer_name_map.items():
#         # Get weights from TensorFlow model
#         if ("highway" not in tf_layer_name):
#             continue
#         tf_weights = tf_model.model.get_layer(name=tf_layer_name).get_weights()

#         # Get weights from PyTorch model
#         torch_layer = getattr(torch_model.model, torch_layer_name)
#         if "embedding" in torch_layer_name:
#             torch_weights = [torch_layer.embedding_list.data.numpy()]
#         elif "highway" in torch_layer_name:
#             # Access parameters of the HighwayLayer
#             torch_weights = [
#                 torch_layer.H.weight.data.numpy(),
#                 torch_layer.H.bias.data.numpy(),
#                 torch_layer.T.weight.data.numpy(),
#                 torch_layer.T.bias.data.numpy(),
#             ]
#         else:
#             torch_weights = [
#                 torch_layer.weight.data.numpy().T,
#                 torch_layer.bias.data.numpy()
#             ]

#         # Compare weights
#         for tf_w, torch_w in zip(tf_weights, torch_weights):
#             assert np.allclose(
#                 tf_w, torch_w, rtol=1e-5,
#                 atol=1e-8), f"Weights mismatch in layer {tf_layer_name}"

#     print("WEIGHTS MATCH")

#     torch_outputs = torch_model.predict(dataset)
#     print(torch_outputs.shape)

#     tf_outputs = tf_model.predict(dataset)
#     print(tf_outputs.shape)

#     print("TORCH OUPUTS: ")
#     print(torch_outputs)
#     print("TF OUTPUTS:")
#     print(tf_outputs)
#     assert np.allclose(torch_outputs, tf_outputs, rtol=1e-5, atol=1e-8)

# scores = torch_model.evaluate(dataset, [classification_metric])

# assert scores[classification_metric.name] > 0.8
"""
[[[9.3935674e-01 6.0643267e-02]]

 [[9.9986064e-01 1.3929828e-04]]

 [[9.8899579e-01 1.1004158e-02]]

 [[9.6090144e-01 3.9098546e-02]]

 [[6.4965068e-05 9.9993503e-01]]

 [[1.3250426e-03 9.9867493e-01]]

 [[9.2744862e-04 9.9907255e-01]]

 [[7.6805037e-03 9.9231952e-01]]

 [[9.9064746e-05 9.9990094e-01]]

 [[2.6711467e-01 7.3288530e-01]]]

TF OUTPUTS:
[[[9.9999982e-01 6.1895250e-08]]

 [[9.9999994e-01 1.2225500e-08]]

 [[9.9999946e-01 4.2065278e-07]]

 [[9.9999982e-01 1.0298383e-07]]

 [[1.6100559e-07 9.9999982e-01]]

 [[1.9315176e-07 9.9999970e-01]]

 [[9.0442484e-08 9.9999982e-01]]

 [[3.6010203e-08 9.9999994e-01]]

 [[2.3516764e-09 1.0000000e+00]]

 [[8.0098994e-09 1.0000000e+00]]]
"""
"""
Layer: conv_layers.0.weight, Shape: torch.Size([100, 75, 1])
Layer: conv_layers.0.bias, Shape: torch.Size([100])
Layer: conv_layers.1.weight, Shape: torch.Size([200, 75, 2])
Layer: conv_layers.1.bias, Shape: torch.Size([200])
Layer: conv_layers.2.weight, Shape: torch.Size([200, 75, 3])
Layer: conv_layers.2.bias, Shape: torch.Size([200])
Layer: conv_layers.3.weight, Shape: torch.Size([200, 75, 4])
Layer: conv_layers.3.bias, Shape: torch.Size([200])
Layer: conv_layers.4.weight, Shape: torch.Size([200, 75, 5])
Layer: conv_layers.4.bias, Shape: torch.Size([200])
Layer: conv_layers.5.weight, Shape: torch.Size([100, 75, 6])
Layer: conv_layers.5.bias, Shape: torch.Size([100])
Layer: conv_layers.6.weight, Shape: torch.Size([100, 75, 7])
Layer: conv_layers.6.bias, Shape: torch.Size([100])
Layer: conv_layers.7.weight, Shape: torch.Size([100, 75, 8])
Layer: conv_layers.7.bias, Shape: torch.Size([100])
Layer: conv_layers.8.weight, Shape: torch.Size([100, 75, 9])
Layer: conv_layers.8.bias, Shape: torch.Size([100])
Layer: conv_layers.9.weight, Shape: torch.Size([100, 75, 10])
Layer: conv_layers.9.bias, Shape: torch.Size([100])
Layer: conv_layers.10.weight, Shape: torch.Size([160, 75, 15])
Layer: conv_layers.10.bias, Shape: torch.Size([160])
Layer: conv_layers.11.weight, Shape: torch.Size([160, 75, 20])
Layer: conv_layers.11.bias, Shape: torch.Size([160])
Layer: embedding_layer.embedding_list, Shape: torch.Size([34, 75])
Layer: linear1.weight, Shape: torch.Size([200, 1720])
Layer: linear1.bias, Shape: torch.Size([200])
Layer: linear2.weight, Shape: torch.Size([2, 200])
Layer: linear2.bias, Shape: torch.Size([2])
Layer: highway.H.weight, Shape: torch.Size([200, 200])
Layer: highway.H.bias, Shape: torch.Size([200])
Layer: highway.T.weight, Shape: torch.Size([200, 200])
Layer: highway.T.bias, Shape: torch.Size([200])
"""
"""

TF layer weights
Layer: input_1

Layer: dtnn_embedding
  Weight 1 shape: (34, 75)

Layer: conv1d
  Weight 1 shape: (1, 75, 100)
  Weight 2 shape: (100,)

Layer: conv1d_1
  Weight 1 shape: (2, 75, 200)
  Weight 2 shape: (200,)

Layer: conv1d_2
  Weight 1 shape: (3, 75, 200)
  Weight 2 shape: (200,)

Layer: conv1d_3
  Weight 1 shape: (4, 75, 200)
  Weight 2 shape: (200,)

Layer: conv1d_4
  Weight 1 shape: (5, 75, 200)
  Weight 2 shape: (200,)

Layer: conv1d_5
  Weight 1 shape: (6, 75, 100)
  Weight 2 shape: (100,)

Layer: conv1d_6
  Weight 1 shape: (7, 75, 100)
  Weight 2 shape: (100,)

Layer: conv1d_7
  Weight 1 shape: (8, 75, 100)
  Weight 2 shape: (100,)

Layer: conv1d_8
  Weight 1 shape: (9, 75, 100)
  Weight 2 shape: (100,)

Layer: conv1d_9
  Weight 1 shape: (10, 75, 100)
  Weight 2 shape: (100,)

Layer: conv1d_10
  Weight 1 shape: (15, 75, 160)
  Weight 2 shape: (160,)

Layer: conv1d_11
  Weight 1 shape: (20, 75, 160)
  Weight 2 shape: (160,)

Layer: lambda

Layer: lambda_1

Layer: lambda_2

Layer: lambda_3

Layer: lambda_4

Layer: lambda_5

Layer: lambda_6

Layer: lambda_7

Layer: lambda_8

Layer: lambda_9

Layer: lambda_10

Layer: lambda_11

Layer: concatenate

Layer: dropout

Layer: dense
  Weight 1 shape: (1720, 200)
  Weight 2 shape: (200,)

Layer: highway
  Weight 1 shape: (200, 200)
  Weight 2 shape: (200,)
  Weight 3 shape: (200, 200)
  Weight 4 shape: (200,)

Layer: dense_1
  Weight 1 shape: (200, 2)
  Weight 2 shape: (2,)

Layer: reshape

Layer: softmax
"""
