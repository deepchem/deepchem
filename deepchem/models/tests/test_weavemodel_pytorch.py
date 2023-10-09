import deepchem as dc
import numpy as np
import os
import pytest
import tempfile
from flaky import flaky

from deepchem.data import NumpyDataset
from deepchem.metrics import Metric, roc_auc_score, mean_absolute_error
from deepchem.molnet import load_bace_classification, load_delaney
from deepchem.feat import WeaveFeaturizer

try:
    import torch
    from deepchem.models.torch_models import Weave, WeaveModel
    has_torch = True
except:
    has_torch = False


def get_dataset(mode='classification',
                featurizer='GraphConv',
                num_tasks=2,
                data_points=20):
    if mode == 'classification':
        tasks, all_dataset, transformers = load_bace_classification(
            featurizer, reload=False)
    else:
        tasks, all_dataset, transformers = load_delaney(featurizer,
                                                        reload=False)

    train, _, _ = all_dataset
    for _ in range(1, num_tasks):
        tasks.append("random_task")
    w = np.ones(shape=(data_points, len(tasks)))

    if mode == 'classification':
        y = np.random.randint(0, 2, size=(data_points, len(tasks)))
        metric = Metric(roc_auc_score, np.mean, mode="classification")
    else:
        y = np.random.normal(size=(data_points, len(tasks)))
        metric = Metric(mean_absolute_error, mode="regression")

    ds = NumpyDataset(train.X[:data_points], y, w, train.ids[:data_points])

    return tasks, ds, transformers, metric


@pytest.mark.torch
def test_compute_features_on_infinity_distance():
    """Test that WeaveModel correctly transforms WeaveMol objects into tensors with infinite max_pair_distance."""
    featurizer = WeaveFeaturizer(max_pair_distance=None)
    X = featurizer(["C", "CCC"])
    batch_size = 20
    model = WeaveModel(1,
                       batch_size=batch_size,
                       mode='classification',
                       fully_connected_layer_sizes=[2000, 1000],
                       batch_normalize=True,
                       learning_rate=0.0005)
    atom_feat, pair_feat, pair_split, atom_split, atom_to_pair = model.compute_features_on_batch(
        X)

    # There are 4 atoms each of which have 75 atom features
    assert atom_feat.shape == (4, 75)
    # There are 10 pairs with infinity distance and 14 pair features
    assert pair_feat.shape == (10, 14)
    # 4 atoms in total
    assert atom_split.shape == (4,)
    assert np.all(atom_split == np.array([0, 1, 1, 1]))
    # 10 pairs in total
    assert pair_split.shape == (10,)
    assert np.all(pair_split == np.array([0, 1, 1, 1, 2, 2, 2, 3, 3, 3]))
    # 10 pairs in total each with start/finish
    assert atom_to_pair.shape == (10, 2)
    assert np.all(
        atom_to_pair == np.array([[0, 0], [1, 1], [1, 2], [1, 3], [2, 1],
                                  [2, 2], [2, 3], [3, 1], [3, 2], [3, 3]]))


@pytest.mark.torch
def test_compute_features_on_distance_1():
    """Test that WeaveModel correctly transforms WeaveMol objects into tensors with finite max_pair_distance."""
    featurizer = WeaveFeaturizer(max_pair_distance=1)
    X = featurizer(["C", "CCC"])
    batch_size = 20
    model = WeaveModel(1,
                       batch_size=batch_size,
                       mode='classification',
                       fully_connected_layer_sizes=[2000, 1000],
                       batch_normalize=True,
                       learning_rate=0.0005)
    atom_feat, pair_feat, pair_split, atom_split, atom_to_pair = model.compute_features_on_batch(
        X)

    # There are 4 atoms each of which have 75 atom features
    assert atom_feat.shape == (4, 75)
    # There are 8 pairs with distance 1 and 14 pair features. (To see why 8,
    # there's the self pair for "C". For "CCC" there are 7 pairs including self
    # connections and accounting for symmetry.)
    assert pair_feat.shape == (8, 14)
    # 4 atoms in total
    assert atom_split.shape == (4,)
    assert np.all(atom_split == np.array([0, 1, 1, 1]))
    # 10 pairs in total
    assert pair_split.shape == (8,)
    # The center atom is self connected and to both neighbors so it appears
    # thrice. The canonical ranking used in MolecularFeaturizer means this
    # central atom is ranked last in ordering.
    assert np.all(pair_split == np.array([0, 1, 1, 2, 2, 3, 3, 3]))
    # 10 pairs in total each with start/finish
    assert atom_to_pair.shape == (8, 2)
    assert np.all(atom_to_pair == np.array([[0, 0], [1, 1], [1, 3], [2, 2],
                                            [2, 3], [3, 1], [3, 2], [3, 3]]))


@pytest.mark.torch
def test_weave_classification():
    "Test involking torch equivalent of Weave Module."
    featurizer = dc.feat.WeaveFeaturizer()
    X = featurizer(["C", "CC"])
    batch_size = 2
    model = WeaveModel(n_tasks=1,
                       n_weave=2,
                       fully_connected_layer_sizes=[2000, 1000],
                       mode="classification",
                       batch_size=batch_size)
    atom_feat, pair_feat, pair_split, atom_split, atom_to_pair = model.compute_features_on_batch(
        X)
    torch.set_printoptions(precision=8)
    model = Weave(n_tasks=1,
                  n_weave=2,
                  fully_connected_layer_sizes=[2000, 1000],
                  mode="classification")
    input_data = [atom_feat, pair_feat, pair_split, atom_split, atom_to_pair]

    for i in range(2):
        model.layers[i].W_AA = torch.from_numpy(
            np.load(f'deepchem/models/tests/assets/weavelayer_W_AA_{i}.npy'))
        model.layers[i].W_PA = torch.from_numpy(
            np.load(f'deepchem/models/tests/assets/weavelayer_W_PA_{i}.npy'))
        model.layers[i].W_A = torch.from_numpy(
            np.load(f'deepchem/models/tests/assets/weavelayer_W_A_{i}.npy'))
        if model.layers[i].update_pair:
            model.layers[i].W_AP = torch.from_numpy(
                np.load(
                    f'deepchem/models/tests/assets/weavelayer_W_AP_{i}.npy'))
            model.layers[i].W_PP = torch.from_numpy(
                np.load(
                    f'deepchem/models/tests/assets/weavelayer_W_PP_{i}.npy'))
            model.layers[i].W_P = torch.from_numpy(
                np.load(f'deepchem/models/tests/assets/weavelayer_W_P_{i}.npy'))
    dense1_weights = np.load(
        'deepchem/models/tests/assets/dense1_weights.npy').astype(np.float32)
    dense1_bias = np.load(
        'deepchem/models/tests/assets/dense1_bias.npy').astype(np.float32)
    model.dense1.weight.data = torch.from_numpy(np.transpose(dense1_weights))
    model.dense1.bias.data = torch.from_numpy(dense1_bias)

    layers2_0_weights = np.load(
        'deepchem/models/tests/assets/layers2_0_weights.npy').astype(np.float32)
    layers2_0_bias = np.load(
        'deepchem/models/tests/assets/layers2_0_bias.npy').astype(np.float32)

    model.layers2[0].weight.data = torch.from_numpy(
        np.transpose(layers2_0_weights))
    model.layers2[0].bias.data = torch.from_numpy(layers2_0_bias)

    if model.weave_gather.compress_post_gaussian_expansion:
        model.weave_gather.W = torch.from_numpy(
            np.load('deepchem/models/tests/assets/weavegather.npy'))

    layers2_1_weights = np.load(
        'deepchem/models/tests/assets/layers2_1_weights.npy').astype(np.float32)
    layers2_1_bias = np.load(
        'deepchem/models/tests/assets/layers2_1_bias.npy').astype(np.float32)

    model.layers2[1].weight.data = torch.from_numpy(
        np.transpose(layers2_1_weights))
    model.layers2[1].bias.data = torch.from_numpy(layers2_1_bias)

    layer_2_weights = np.load(
        'deepchem/models/tests/assets/layer_2_weights.npy').astype(np.float32)
    layer_2_bias = np.load(
        'deepchem/models/tests/assets/layer_2_bias.npy').astype(np.float32)

    model.layer_2.weight.data = torch.from_numpy(np.transpose(layer_2_weights))
    model.layer_2.bias.data = torch.from_numpy(layer_2_bias)

    outputs = model(input_data)
    assert len(outputs) == 2
    assert np.allclose(
        outputs[1].detach().numpy(),
        np.load('deepchem/models/tests/assets/classification_logits.npy'),
        atol=1e-4)
    assert np.allclose(
        outputs[0].detach().numpy(),
        np.load('deepchem/models/tests/assets/classification_output.npy'),
        atol=1e-4)


@flaky
@pytest.mark.slow
@pytest.mark.torch
def test_weave_model():
    np.random.seed(22)
    torch.manual_seed(22)
    tasks, dataset, transformers, metric = get_dataset('classification',
                                                       'Weave',
                                                       data_points=10)
    # Note: Following are some changes
    # compared to the TensorFlow unit test:
    # 1. Changed nb_epoch to 300.
    # 2. Increased the learning_rate to 0.0003.

    batch_size = 10
    weave_model = WeaveModel(len(tasks),
                             batch_size=batch_size,
                             mode='classification',
                             dropouts=0,
                             learning_rate=0.0003)
    weave_model.fit(dataset, nb_epoch=300)
    scores = weave_model.evaluate(dataset, [metric], transformers)
    # Note: This needs to be inspected in future to understand low score as compared to a score of 0.9 in tensorflow unit test.
    assert scores['mean-roc_auc_score'] >= 0.8


@pytest.mark.torch
def test_weave_fit_simple_distance_1():
    featurizer = WeaveFeaturizer(max_pair_distance=1)
    X = featurizer(["C", "CCC"])
    y = np.array([0, 1.])
    dataset = NumpyDataset(X, y)

    batch_size = 20
    model = WeaveModel(1,
                       batch_size=batch_size,
                       mode='classification',
                       fully_connected_layer_sizes=[2000, 1000],
                       batch_normalize=True,
                       learning_rate=0.0005)
    model.fit(dataset, nb_epoch=200)
    transformers = []
    metric = Metric(roc_auc_score, np.mean, mode="classification")
    scores = model.evaluate(dataset, [metric], transformers)
    assert scores['mean-roc_auc_score'] >= 0.9


@pytest.mark.slow
@pytest.mark.torch
def test_weave_regression_model():
    np.random.seed(22)
    torch.manual_seed(22)
    tasks, dataset, transformers, metric = get_dataset('regression',
                                                       'Weave',
                                                       data_points=2)
    # Note: Following are some changes
    # compared to the TensorFlow unit test:
    # 1. Changed nb_epoch to 400.
    # 2. Reduced the number of data points to 2.
    # 3. Increased batch_size to 20.
    # 4. Increased the learning_rate to 0.0003.
    batch_size = 20
    weave_model = WeaveModel(len(tasks),
                             batch_size=batch_size,
                             mode='regression',
                             dropouts=0,
                             learning_rate=0.0003)
    weave_model.fit(dataset, nb_epoch=400)
    scores = weave_model.evaluate(dataset, [metric], transformers)
    assert scores['mean_absolute_error'] < 0.1


@pytest.mark.torch
def test_weave_model_reload():
    "Test WeaveModel class for reloading the model."
    torch.manual_seed(0)
    featurizer = WeaveFeaturizer(max_pair_distance=1)
    X = featurizer(["C", "CCC"])
    y = np.array([0, 1.])
    dataset = NumpyDataset(X, y)

    batch_size = 20
    model_dir = tempfile.mkdtemp()
    original_model = WeaveModel(1,
                                batch_size=batch_size,
                                mode='classification',
                                fully_connected_layer_sizes=[2000, 1000],
                                batch_normalize=True,
                                learning_rate=0.0005,
                                model_dir=model_dir)
    original_model.fit(dataset, nb_epoch=200)

    reloaded_model = WeaveModel(1,
                                batch_size=batch_size,
                                mode='classification',
                                fully_connected_layer_sizes=[2000, 1000],
                                batch_normalize=True,
                                learning_rate=0.0005,
                                model_dir=model_dir)
    reloaded_model.restore()
    original_predict = original_model.predict(dataset)
    reloaded_predict = reloaded_model.predict(dataset)
    assert np.all(original_predict == reloaded_predict)


@pytest.mark.torch
def test_weave_singletask_classification_overfit():
    """Test weave model overfits tiny data."""
    np.random.seed(22)
    torch.manual_seed(22)
    n_tasks = 1
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Load mini log-solubility dataset.
    featurizer = dc.feat.WeaveFeaturizer()
    tasks = ["outcome"]
    input_file = os.path.join(current_dir, "assets/example_classification.csv")
    loader = dc.data.CSVLoader(tasks=tasks,
                               feature_field="smiles",
                               featurizer=featurizer)
    dataset = loader.create_dataset(input_file)

    classification_metric = dc.metrics.Metric(dc.metrics.accuracy_score)

    batch_size = 10
    model = WeaveModel(n_tasks,
                       batch_size=batch_size,
                       learning_rate=0.0003,
                       dropout=0.0,
                       mode="classification")

    # Fit trained model
    model.fit(dataset, nb_epoch=300)

    # Eval model on train
    scores = model.evaluate(dataset, [classification_metric])
    # Note: This needs to be inspected in future to understand low score.
    assert scores[classification_metric.name] > .65


@pytest.mark.slow
@pytest.mark.torch
def test_weave_singletask_regression_overfit():
    """Test weave model overfits tiny data."""
    np.random.seed(22)
    torch.manual_seed(22)
    n_tasks = 1
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Load mini log-solubility dataset.
    featurizer = dc.feat.WeaveFeaturizer()
    tasks = ["outcome"]
    input_file = os.path.join(current_dir, "assets/example_regression.csv")
    loader = dc.data.CSVLoader(tasks=tasks,
                               feature_field="smiles",
                               featurizer=featurizer)
    dataset = loader.create_dataset(input_file)

    regression_metric = dc.metrics.Metric(dc.metrics.pearson_r2_score,
                                          task_averager=np.mean)

    batch_size = 10

    model = WeaveModel(n_tasks,
                       batch_size=batch_size,
                       learning_rate=0.0003,
                       dropout=0.0,
                       mode="regression")

    # Fit trained model
    # Note: Compared to the tensorflow unit test increased the nb_epoch to 600.
    model.fit(dataset, nb_epoch=600)

    # Eval model on train
    scores = model.evaluate(dataset, [regression_metric])
    # Note: This needs to be inspected in future to understand low score as compared to a score of 0.8 in tensorflow unit test.
    assert scores[regression_metric.name] > .7
