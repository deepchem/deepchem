import os
import numpy as np
import tempfile
import pytest
import pickle

import deepchem as dc
from deepchem.feat import create_char_to_idx, SmilesToSeq
from deepchem.molnet.load_function.chembl25_datasets import CHEMBL25_TASKS

try:
    import torch
    from deepchem.models.torch_models import Smiles2VecModel
    import torch.nn.functional as F
except ModuleNotFoundError:
    pass


def get_dataset(mode="regression",
                featurizer="smiles2seq",
                max_seq_len=20,
                data_points=10,
                n_tasks=5,
                n_classes=2):

    dataset_file = os.path.join(os.path.dirname(__file__), "assets",
                                "chembl_25_small.csv")
    if featurizer == "smiles2seq":
        max_len = 250
        pad_len = 10
        char_to_idx = create_char_to_idx(dataset_file,
                                         max_len=max_len,
                                         smiles_field="smiles")

        feat = SmilesToSeq(char_to_idx=char_to_idx,
                           max_len=max_len,
                           pad_len=pad_len)

    loader = dc.data.CSVLoader(tasks=CHEMBL25_TASKS,
                               feature_field='smiles',
                               featurizer=feat)

    dataset = loader.create_dataset(inputs=[dataset_file],
                                    shard_size=10000,
                                    data_dir=tempfile.mkdtemp())

    w = np.ones(shape=(data_points, n_tasks))

    if mode == 'classification':
        y = np.random.randint(0, 2, size=(data_points, n_tasks))
        y = torch.from_numpy(y.flatten()).long()
        y = F.one_hot(y, n_classes).view(-1, n_tasks, n_classes)
        y = y.float()
        metric = dc.metrics.Metric(dc.metrics.roc_auc_score,
                                   np.mean,
                                   mode="classification")
    else:
        y = np.random.normal(size=(data_points, n_tasks))
        metric = dc.metrics.Metric(dc.metrics.mean_absolute_error,
                                   mode="regression")

    if featurizer == "smiles2seq":
        dataset = dc.data.NumpyDataset(dataset.X[:data_points, :max_seq_len], y,
                                       w, dataset.ids[:data_points])
    else:
        dataset = dc.data.NumpyDataset(dataset.X[:data_points], y, w,
                                       dataset.ids[:data_points])

    if featurizer == "smiles2seq":
        return dataset, metric, char_to_idx
    else:
        return dataset, metric


@pytest.mark.torch
def test_Smiles2Vec_forward():
    from deepchem.models.torch_models import Smiles2Vec

    n_tasks = 5
    max_seq_len = 20

    _, _, char_to_idx = get_dataset(
        mode="regression",
        featurizer="smiles2seq",
        n_tasks=n_tasks,
        max_seq_len=max_seq_len,
    )
    model = Smiles2Vec(char_to_idx=char_to_idx,
                       max_seq_len=max_seq_len,
                       n_tasks=n_tasks)

    input = torch.randint(low=0, high=len(char_to_idx), size=(1, max_seq_len))
    # Ex: input = torch.tensor([[32,32,32,32,32,32,25,29,15,17,29,29,32,32,32,32,32,32,32,32]])

    logits = model.forward(input)
    assert np.shape(logits) == (1, n_tasks, 1)


@pytest.mark.torch
def test_Smiles2VecModel_regression():
    from deepchem.models.torch_models import Smiles2VecModel

    n_tasks = 5
    max_seq_len = 20

    dataset, metric, char_to_idx = get_dataset(
        mode="regression",
        featurizer="smiles2seq",
        n_tasks=n_tasks,
        max_seq_len=max_seq_len,
    )
    model = Smiles2VecModel(char_to_idx=char_to_idx,
                            max_seq_len=max_seq_len,
                            use_conv=True,
                            n_tasks=n_tasks,
                            model_dir=None,
                            mode="regression")

    model.fit(dataset, nb_epoch=500)
    scores = model.evaluate(dataset, [metric], [])
    assert scores['mean_absolute_error'] < 0.1


@pytest.mark.torch
def test_Smiles2VecModel_classification():
    from deepchem.models.torch_models import Smiles2VecModel

    n_tasks = 5
    max_seq_len = 20

    dataset, metric, char_to_idx, = get_dataset(mode="classification",
                                                featurizer="smiles2seq",
                                                n_tasks=n_tasks,
                                                max_seq_len=max_seq_len)

    model = Smiles2VecModel(char_to_idx=char_to_idx,
                            max_seq_len=max_seq_len,
                            use_conv=True,
                            n_tasks=n_tasks,
                            mode="classification")

    model.fit(dataset, nb_epoch=500)
    scores = model.evaluate(dataset, [metric], [])
    assert scores['mean-roc_auc_score'] >= 0.9


@pytest.mark.torch
def test_Smiles2VecModel_reload():
    from deepchem.models.torch_models import Smiles2VecModel

    n_tasks = 5
    max_seq_len = 20

    # Create a temporary directory for the model
    model_dir = tempfile.mkdtemp()

    # Load dataset
    dataset, metric, char_to_idx = get_dataset(mode="regression",
                                               featurizer="smiles2seq",
                                               n_tasks=n_tasks,
                                               max_seq_len=max_seq_len)
    # Initialize and train the model
    model = Smiles2VecModel(char_to_idx=char_to_idx,
                            max_seq_len=max_seq_len,
                            use_conv=True,
                            n_tasks=n_tasks,
                            model_dir=model_dir,
                            mode="regression")
    model.fit(dataset, nb_epoch=10)
    scores = model.evaluate(dataset, [metric], [])

    # Reload the trained model
    reloaded_model = Smiles2VecModel(char_to_idx=char_to_idx,
                                     max_seq_len=max_seq_len,
                                     use_conv=True,
                                     n_tasks=n_tasks,
                                     model_dir=model_dir,
                                     mode="regression")
    reloaded_model.restore()
    reloaded_scores = reloaded_model.evaluate(dataset, [metric], [])

    assert scores == reloaded_scores


@pytest.mark.torch
def test_smiles2vec_compare_with_tf_impl():

    dataset_file = os.path.join(os.path.dirname(__file__), "assets",
                                "chembl_25_small.csv")
    tensorflow_weights_dir = os.path.join(os.path.dirname(__file__),
                                          "assets/smiles2vec")
    max_seq_len = 20
    max_len = 250
    n_tasks = 1
    data_points = 1

    char_to_idx = create_char_to_idx(dataset_file,
                                     max_len=max_len,
                                     smiles_field="smiles")

    smiles = torch.tensor([[
        32, 32, 32, 32, 32, 32, 25, 29, 15, 17, 29, 29, 32, 32, 32, 32, 32, 32,
        32, 32
    ]])

    X = smiles
    w = np.ones(shape=(data_points, n_tasks))
    y = np.random.normal(size=(data_points, n_tasks))
    dataset = dc.data.NumpyDataset(X=X, y=y)

    dataset = dc.data.NumpyDataset(dataset.X[:data_points, :max_seq_len], y, w,
                                   dataset.ids[:data_points])

    torch_model = Smiles2VecModel(char_to_idx=char_to_idx,
                                  max_seq_len=max_seq_len,
                                  use_conv=True,
                                  n_tasks=n_tasks,
                                  model_dir=None,
                                  mode="regression",
                                  device=torch.device('cpu'))

    # Copy layer weights
    torch_layer_names = [
        "embedding", "conv1d", "rnn_layers", "last_rnn_layer", "fc"
    ]

    def regroup_params_gru(weight_or_bias_gru, axis=0):
        assert len(weight_or_bias_gru.shape
                  ) == 2 and weight_or_bias_gru.shape[axis] % 3 == 0
        # change params for the 3 gates from tf order(z,r,h) into torch order(r,z,h)
        [z, r, h] = np.split(weight_or_bias_gru, 3, axis=axis)
        return np.concatenate((r, z, h), axis=axis)

    # Copy layer weights
    torch_layer_names = [
        "embedding", "conv1d", "rnn_layers", "last_rnn_layer", "fc"
    ]

    with torch.no_grad():

        for torch_layer_name in torch_layer_names:
            torch_layer = getattr(torch_model.model, torch_layer_name)

            if ("fc" in torch_layer_name):
                with open(os.path.join(tensorflow_weights_dir, 'dense.pickle'),
                          'rb') as f:
                    dense_w = pickle.load(f)
                weights_1 = np.transpose(dense_w[0])
                weights_2 = dense_w[1]
                torch_layer.weight.copy_(torch.from_numpy(weights_1).float())
                torch_layer.bias.copy_(torch.from_numpy(weights_2).float())

            elif ("embedding" in torch_layer_name):
                with open(
                        os.path.join(tensorflow_weights_dir,
                                     'embedding.pickle'), 'rb') as f:
                    emb_w = pickle.load(f)
                torch_layer.weight.data.copy_(
                    torch.from_numpy(emb_w[0]).float())

            elif ("conv1d" in torch_layer_name):
                with open(
                        os.path.join(tensorflow_weights_dir, 'conv_1d.pickle'),
                        'rb') as f:
                    conv_w = pickle.load(f)
                weights_1 = np.transpose(conv_w[0], (2, 1, 0))
                weights_2 = conv_w[1]
                # Copy weights and biases to the PyTorch layer
                torch_layer.weight.data.copy_(
                    torch.from_numpy(weights_1).float())
                torch_layer.bias.data.copy_(torch.from_numpy(weights_2).float())

            elif ("last_rnn_layer" in torch_layer_name):
                with open(
                        os.path.join(tensorflow_weights_dir,
                                     'last_rnn_layer.pickle'), 'rb') as f:
                    rnn_w = pickle.load(f)

                weights_ih = regroup_params_gru(rnn_w[0].T, axis=0)
                weights_hh = regroup_params_gru(rnn_w[1].T, axis=0)
                bias = regroup_params_gru(rnn_w[2], axis=1)
                bias_ih = bias[0]
                bias_hh = bias[1]

                weights_ih_reverse = regroup_params_gru(rnn_w[3].T, axis=0)
                weights_hh_reverse = regroup_params_gru(rnn_w[4].T, axis=0)
                bias_reverse = regroup_params_gru(rnn_w[5], axis=1)
                bias_ih_reverse = bias_reverse[0]
                bias_hh_reverse = bias_reverse[1]

                if torch_layer.bidirectional:
                    # Copy weights and biases to the PyTorch layer
                    torch_layer.weight_ih_l0.data.copy_(
                        torch.from_numpy(weights_ih).float())
                    torch_layer.weight_hh_l0.data.copy_(
                        torch.from_numpy(weights_hh).float())
                    torch_layer.bias_ih_l0.data.copy_(
                        torch.from_numpy(bias_ih).float())
                    torch_layer.bias_hh_l0.data.copy_(
                        torch.from_numpy(bias_hh).float())

                    torch_layer.weight_ih_l0_reverse.data.copy_(
                        torch.from_numpy(weights_ih_reverse).float())
                    torch_layer.weight_hh_l0_reverse.data.copy_(
                        torch.from_numpy(weights_hh_reverse).float())
                    torch_layer.bias_ih_l0_reverse.data.copy_(
                        torch.from_numpy(bias_ih_reverse).float())
                    torch_layer.bias_hh_l0_reverse.data.copy_(
                        torch.from_numpy(bias_hh_reverse).float())

            elif ("rnn_layers" in torch_layer_name):

                for _, torch_layer in enumerate(torch_model.model.rnn_layers):

                    with open(
                            os.path.join(tensorflow_weights_dir,
                                         'rnn_layer_0.pickle'), 'rb') as f:
                        rnn_layer_w = pickle.load(f)

                    weights_ih = regroup_params_gru(rnn_layer_w[0].T, axis=0)
                    weights_hh = regroup_params_gru(rnn_layer_w[1].T, axis=0)
                    bias = regroup_params_gru(rnn_layer_w[2], axis=1)
                    bias_ih = bias[0]
                    bias_hh = bias[1]

                    torch_layer.weight_ih_l0.data.copy_(
                        torch.from_numpy(weights_ih).float())
                    torch_layer.weight_hh_l0.data.copy_(
                        torch.from_numpy(weights_hh).float())
                    torch_layer.bias_ih_l0.data.copy_(
                        torch.from_numpy(bias_ih).float())
                    torch_layer.bias_hh_l0.data.copy_(
                        torch.from_numpy(bias_hh).float())

                    if torch_layer.bidirectional:
                        weights_ih_reverse = regroup_params_gru(
                            rnn_layer_w[3].T, axis=0)
                        weights_hh_reverse = regroup_params_gru(
                            rnn_layer_w[4].T, axis=0)
                        bias_reverse = regroup_params_gru(rnn_layer_w[5],
                                                          axis=1)
                        bias_ih_reverse = bias_reverse[0]
                        bias_hh_reverse = bias_reverse[1]

                        torch_layer.weight_ih_l0_reverse.data.copy_(
                            torch.from_numpy(weights_ih_reverse).float())
                        torch_layer.weight_hh_l0_reverse.data.copy_(
                            torch.from_numpy(weights_hh_reverse).float())
                        torch_layer.bias_ih_l0_reverse.data.copy_(
                            torch.from_numpy(bias_ih_reverse).float())
                        torch_layer.bias_hh_l0_reverse.data.copy_(
                            torch.from_numpy(bias_hh_reverse).float())

    # predicting using the torch model with copied weights and biases
    torch_outputs = torch_model.predict(dataset)
    with open(os.path.join(tensorflow_weights_dir, 'tf_outputs.pickle'),
              'rb') as f:
        tf_outputs = pickle.load(f)

    # comparing tf_outputs and torch_outputs
    assert np.allclose(torch_outputs, tf_outputs, rtol=1e-5, atol=1e-6)
