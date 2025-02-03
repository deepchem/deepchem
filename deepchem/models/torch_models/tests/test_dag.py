import pytest
try:
    import torch
    from deepchem.models.torch_models.dag import DAGModel
except (ModuleNotFoundError, ImportError):
    pass
import numpy as np
import deepchem as dc
import tempfile
from deepchem.trans.transformers import DAGTransformer
from deepchem.metrics import Metric, roc_auc_score, mean_absolute_error
from deepchem.molnet import load_bace_classification, load_delaney
from deepchem.data import NumpyDataset
import os


def get_dataset(mode='classification', featurizer='GraphConv', num_tasks=2):
    data_points = 20
    if mode == 'classification':
        tasks, all_dataset, transformers = load_bace_classification(
            featurizer, reload=False)
    else:
        tasks, all_dataset, transformers = load_delaney(featurizer,
                                                        reload=False)

    train, valid, test = all_dataset
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


@pytest.mark.slow
@pytest.mark.torch
def test_dag_model():
    tasks, dataset, transformers, metric = get_dataset('classification',
                                                       'GraphConv')

    max_atoms = max([mol.get_num_atoms() for mol in dataset.X])
    transformer = DAGTransformer(max_atoms=max_atoms)
    dataset = transformer.transform(dataset)

    model = DAGModel(len(tasks),
                     max_atoms=max_atoms,
                     mode='classification',
                     learning_rate=0.001)

    model.fit(dataset, nb_epoch=30)
    scores = model.evaluate(dataset, [metric], transformers)
    assert scores['mean-roc_auc_score'] >= 0.9


@pytest.mark.slow
@pytest.mark.torch
def test_dag_regression_model():
    np.random.seed(1234)
    torch.manual_seed(1234)
    tasks, dataset, transformers, metric = get_dataset('regression',
                                                       'GraphConv')

    max_atoms = max([mol.get_num_atoms() for mol in dataset.X])
    transformer = DAGTransformer(max_atoms=max_atoms)
    dataset = transformer.transform(dataset)

    model = DAGModel(len(tasks),
                     max_atoms=max_atoms,
                     mode='regression',
                     learning_rate=0.003)

    model.fit(dataset, nb_epoch=100)
    scores = model.evaluate(dataset, [metric], transformers)
    assert scores['mean_absolute_error'] < 0.15


@pytest.mark.slow
@pytest.mark.torch
def test_dag_regression_uncertainty():
    np.random.seed(1234)
    torch.manual_seed(1234)
    tasks, dataset, _, _ = get_dataset('regression', 'GraphConv')

    batch_size = 10
    max_atoms = max([mol.get_num_atoms() for mol in dataset.X])
    transformer = DAGTransformer(max_atoms=max_atoms)
    dataset = transformer.transform(dataset)

    model = DAGModel(len(tasks),
                     max_atoms=max_atoms,
                     mode='regression',
                     learning_rate=0.003,
                     batch_size=batch_size,
                     use_queue=False,
                     dropout=0.05,
                     uncertainty=True)

    model.fit(dataset, nb_epoch=750)

    # Predict the output and uncertainty.
    pred, std = model.predict_uncertainty(dataset)
    mean_error = np.mean(np.abs(dataset.y - pred))
    mean_value = np.mean(np.abs(dataset.y))
    mean_std = np.mean(std)
    # The DAG models have high error with dropout
    # Despite a lot of effort tweaking it , there appears to be
    # a limit to how low the error can go with dropout.
    assert mean_error < .7 * mean_value
    assert mean_std > 0.5 * mean_error
    assert mean_std < mean_value


@pytest.mark.torch
def test_DAG_regression_reload():
    """Test DAG regressor reloads."""
    np.random.seed(123)
    n_tasks = 1

    # Load mini log-solubility dataset.
    featurizer = dc.feat.ConvMolFeaturizer()
    mols = [
        "CC", "CCO", "CC", "CCC", "CCCCO", "CO", "CC", "CCCCC", "CCC", "CCCO"
    ]
    n_samples = len(mols)
    X = featurizer(mols)
    y = np.random.rand(n_samples, n_tasks)
    dataset = dc.data.NumpyDataset(X, y)

    regression_metric = dc.metrics.Metric(dc.metrics.pearson_r2_score,
                                          task_averager=np.mean)

    n_feat = 75
    batch_size = 10
    transformer = dc.trans.DAGTransformer(max_atoms=50)
    dataset = transformer.transform(dataset)

    model_dir = tempfile.mkdtemp()
    model = DAGModel(n_tasks,
                     max_atoms=50,
                     n_atom_feat=n_feat,
                     batch_size=batch_size,
                     learning_rate=0.001,
                     use_queue=False,
                     mode="regression",
                     model_dir=model_dir)

    # Fit trained model
    model.fit(dataset, nb_epoch=100)

    # Eval model on train
    scores = model.evaluate(dataset, [regression_metric])
    assert scores[regression_metric.name] > .1

    reloaded_model = DAGModel(n_tasks,
                              max_atoms=50,
                              n_atom_feat=n_feat,
                              batch_size=batch_size,
                              learning_rate=0.001,
                              use_queue=False,
                              mode="regression",
                              model_dir=model_dir)

    reloaded_model.restore()

    # Check predictions match on random sample
    predmols = ["CCCC", "CCCCCO", "CCCCC"]
    Xpred = featurizer(predmols)
    predset = dc.data.NumpyDataset(Xpred)
    predset = transformer.transform(predset)
    origpred = model.predict(predset)
    reloadpred = reloaded_model.predict(predset)

    assert torch.allclose(torch.tensor(origpred),
                          torch.tensor(reloadpred),
                          atol=1e-6)


@pytest.mark.slow
@pytest.mark.torch
def test_DAG_singletask_regression_overfit():
    """Test DAG regressor overfits tiny data."""
    np.random.seed(123)
    torch.manual_seed(123)
    n_tasks = 1
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Load mini log-solubility dataset.
    featurizer = dc.feat.ConvMolFeaturizer()
    tasks = ["outcome"]

    input_file = os.path.join(
        current_dir.replace(r'/torch_models/',
                            '/').replace('\\torch_models\\', '\\'), "assets",
        "example_regression.csv")
    loader = dc.data.CSVLoader(tasks=tasks,
                               feature_field="smiles",
                               featurizer=featurizer)
    dataset = loader.create_dataset(input_file)

    regression_metric = dc.metrics.Metric(dc.metrics.pearson_r2_score,
                                          task_averager=np.mean)

    n_feat = 75
    batch_size = 10
    transformer = dc.trans.DAGTransformer(max_atoms=50)
    dataset = transformer.transform(dataset)

    model = DAGModel(n_tasks,
                     max_atoms=50,
                     n_atom_feat=n_feat,
                     batch_size=batch_size,
                     learning_rate=0.001,
                     use_queue=False,
                     mode="regression")

    # Fit trained model
    model.fit(dataset, nb_epoch=1200)
    # Eval model on train
    scores = model.evaluate(dataset, [regression_metric])

    assert scores[regression_metric.name] > .8


@pytest.mark.torch
def test_DAG_correctness():
    """
    DAGModel correctness test using a fixed np seed wrt keras version of DAG.
    """
    # Set random seed
    np.random.seed(123)
    tasks, dataset, transformers, metric = get_dataset('classification',
                                                       'GraphConv')

    max_atoms = max([mol.get_num_atoms() for mol in dataset.X])
    transformer = DAGTransformer(max_atoms=max_atoms)
    dataset = transformer.transform(dataset)
    # Initialize the model
    model = DAGModel(n_tasks=1,
                     max_atoms=max_atoms,
                     n_atom_feat=75,
                     n_graph_feat=30,
                     n_outputs=30,
                     layer_sizes=[100],
                     layer_sizes_gather=[100],
                     mode='classification',
                     n_classes=2,
                     batch_size=100,
                     device='cpu')

    def create_weight(shape):
        """
        Creates a weight matrix of a given shape.
        """
        return np.random.normal(0, 0.1, size=shape)

    # Expected output for the first 2 samples obtained from the keras version of DAG
    expected_output = torch.tensor([[[0.74370146, 0.25629854]],
                                    [[0.6930733, 0.30692676]]])

    with torch.no_grad():
        # class daglayers has 2 sets of weights and biases
        # W_layers[0] initializes the weights for the first layer
        model.model.dag_layer.W_layers[0].copy_(
            torch.from_numpy(
                create_weight(
                    (75 + (max_atoms - 1) * 30, 100)).astype(np.float32)))

        # b_layers[0] initializes the biases for the first layer
        model.model.dag_layer.b_layers[0].copy_(
            torch.from_numpy(create_weight((100,)).astype(np.float32)))

        # W_layers[1] initializes the weights for the second layer
        model.model.dag_layer.W_layers[1].copy_(
            torch.from_numpy(create_weight((100, 30)).astype(np.float32)))

        # b_layers[1] initializes the biases for the second layer
        model.model.dag_layer.b_layers[1].copy_(
            torch.from_numpy(create_weight((30,)).astype(np.float32)))

        # class daggather has 2 sets of weights and biases
        # W_layers[0] initializes the weights for the first layer
        model.model.dag_gather.W_layers[0].copy_(
            torch.from_numpy(create_weight((30, 100)).astype(np.float32)))

        # b_layers[0] initializes the biases for the first layer
        model.model.dag_gather.b_layers[0].copy_(
            torch.from_numpy(create_weight((100,)).astype(np.float32)))

        # W_layers[1] initializes the weights for the second layer
        model.model.dag_gather.W_layers[1].copy_(
            torch.from_numpy(create_weight((100, 30)).astype(np.float32)))

        # b_layers[1] initializes the biases for the second layer
        model.model.dag_gather.b_layers[1].copy_(
            torch.from_numpy(create_weight((30,)).astype(np.float32)))

        # linear layers in the class _DAG
        model.model.dense.weight.copy_(
            torch.from_numpy(create_weight((30, 2)).astype(np.float32).T))
        model.model.dense.bias.copy_(
            torch.from_numpy(create_weight(2).astype(np.float32)))

    assert torch.allclose(torch.tensor(model.predict(dataset)[:2]),
                          expected_output,
                          atol=1e-6)
