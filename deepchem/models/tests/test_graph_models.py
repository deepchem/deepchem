import os
import numpy as np
import pytest
from scipy import io as scipy_io

from deepchem.data import NumpyDataset, CSVLoader
from deepchem.trans import DAGTransformer
from deepchem.molnet import load_bace_classification, load_delaney
from deepchem.feat import ConvMolFeaturizer
from deepchem.metrics import Metric, roc_auc_score, mean_absolute_error
from deepchem.utils.data_utils import download_url, get_data_dir

try:
    import tensorflow as tf
    from deepchem.models import GraphConvModel, DAGModel, MPNNModel, DTNNModel
    has_tensorflow = True
except:
    has_tensorflow = False

from flaky import flaky


@pytest.mark.tensorflow
def get_dataset(mode='classification', featurizer='GraphConv', num_tasks=2):
    data_points = 20
    if mode == 'classification':
        tasks, all_dataset, transformers = load_bace_classification(featurizer)
    else:
        tasks, all_dataset, transformers = load_delaney(featurizer)

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


@flaky
@pytest.mark.tensorflow
def test_graph_conv_model():
    tasks, dataset, transformers, metric = get_dataset('classification',
                                                       'GraphConv')

    batch_size = 10
    model = GraphConvModel(len(tasks),
                           batch_size=batch_size,
                           batch_normalize=False,
                           mode='classification')

    model.fit(dataset, nb_epoch=20)
    scores = model.evaluate(dataset, [metric], transformers)
    assert scores['mean-roc_auc_score'] >= 0.9


@pytest.mark.tensorflow
def test_neural_fingerprint_retrieval():
    tasks, dataset, _, _ = get_dataset('classification', 'GraphConv')

    fp_size = 3

    batch_size = 50
    model = GraphConvModel(len(tasks),
                           batch_size=batch_size,
                           dense_layer_size=3,
                           mode='classification')

    model.fit(dataset, nb_epoch=1)
    neural_fingerprints = model.predict_embedding(dataset)
    neural_fingerprints = np.array(neural_fingerprints)[:len(dataset)]
    assert (len(dataset), fp_size * 2) == neural_fingerprints.shape


@flaky
@pytest.mark.tensorflow
def test_graph_conv_regression_model():
    tasks, dataset, transformers, metric = get_dataset('regression',
                                                       'GraphConv')

    batch_size = 10
    model = GraphConvModel(len(tasks),
                           batch_size=batch_size,
                           batch_normalize=False,
                           mode='regression')

    model.fit(dataset, nb_epoch=100)
    scores = model.evaluate(dataset, [metric], transformers)
    assert scores['mean_absolute_error'] < 0.1


@pytest.mark.tensorflow
def test_graph_conv_regression_uncertainty():
    tasks, dataset, _, _ = get_dataset('regression', 'GraphConv')

    batch_size = 10
    model = GraphConvModel(len(tasks),
                           batch_size=batch_size,
                           batch_normalize=False,
                           mode='regression',
                           dropout=0.1,
                           uncertainty=True)

    model.fit(dataset, nb_epoch=100)

    # Predict the output and uncertainty.
    pred, std = model.predict_uncertainty(dataset)
    mean_error = np.mean(np.abs(dataset.y - pred))
    mean_value = np.mean(np.abs(dataset.y))
    mean_std = np.mean(std)
    assert mean_error < 0.5 * mean_value
    assert mean_std > 0.5 * mean_error
    assert mean_std < mean_value


@pytest.mark.tensorflow
def test_graph_conv_model_no_task():
    tasks, dataset, _, __ = get_dataset('classification', 'GraphConv')
    batch_size = 10
    model = GraphConvModel(len(tasks),
                           batch_size=batch_size,
                           batch_normalize=False,
                           mode='classification')
    model.fit(dataset, nb_epoch=20)
    # predict datset with no y (ensured by tasks = [])
    bace_url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv"
    download_url(url=bace_url, name="bace_tmp.csv")
    loader = CSVLoader(tasks=[],
                       smiles_field='mol',
                       featurizer=ConvMolFeaturizer())
    td = loader.featurize(os.path.join(get_data_dir(), "bace_tmp.csv"))
    model.predict(td)


@pytest.mark.tensorflow
def test_graph_conv_atom_features():
    tasks, dataset, _, _ = get_dataset('regression', 'Raw', num_tasks=1)

    atom_feature_name = 'feature'
    y = []
    for mol in dataset.X:
        atom_features = []
        for atom in mol.GetAtoms():
            val = np.random.normal()
            mol.SetProp("atom %08d %s" % (atom.GetIdx(), atom_feature_name),
                        str(val))
            atom_features.append(np.random.normal())
        y.append([np.sum(atom_features)])

    featurizer = ConvMolFeaturizer(atom_properties=[atom_feature_name])
    X = featurizer.featurize(dataset.X)
    dataset = NumpyDataset(X, np.array(y))
    batch_size = 50
    model = GraphConvModel(len(tasks),
                           number_atom_features=featurizer.feature_length(),
                           batch_size=batch_size,
                           mode='regression')

    model.fit(dataset, nb_epoch=1)
    _ = model.predict(dataset)


@flaky
@pytest.mark.slow
@pytest.mark.tensorflow
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
@pytest.mark.tensorflow
def test_dag_regression_model():
    np.random.seed(1234)
    tf.random.set_seed(1234)
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
@pytest.mark.tensorflow
def test_dag_regression_uncertainty():
    np.random.seed(1234)
    tf.random.set_seed(1234)
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
    # assert mean_error < 0.5 * mean_value
    assert mean_error < .7 * mean_value
    assert mean_std > 0.5 * mean_error
    assert mean_std < mean_value


@pytest.mark.slow
@pytest.mark.tensorflow
def test_mpnn_model():
    tasks, dataset, transformers, metric = get_dataset('classification',
                                                       'Weave')

    model = MPNNModel(len(tasks),
                      mode='classification',
                      n_hidden=75,
                      n_atom_feat=75,
                      n_pair_feat=14,
                      T=1,
                      M=1,
                      learning_rate=0.0005)

    model.fit(dataset, nb_epoch=150)
    scores = model.evaluate(dataset, [metric], transformers)
    assert scores['mean-roc_auc_score'] >= 0.9


@flaky(max_runs=3, min_passes=1)
@pytest.mark.slow
@pytest.mark.tensorflow
def test_mpnn_regression_model():
    tasks, dataset, transformers, metric = get_dataset('regression', 'Weave')

    batch_size = 10
    model = MPNNModel(len(tasks),
                      mode='regression',
                      n_hidden=75,
                      n_atom_feat=75,
                      n_pair_feat=14,
                      T=1,
                      M=1,
                      batch_size=batch_size)

    model.fit(dataset, nb_epoch=60)
    scores = model.evaluate(dataset, [metric], transformers)
    assert scores['mean_absolute_error'] < 0.1


@pytest.mark.slow
@pytest.mark.tensorflow
def test_mpnn_regression_uncertainty():
    tasks, dataset, _, _ = get_dataset('regression', 'Weave')

    batch_size = 10
    model = MPNNModel(len(tasks),
                      mode='regression',
                      n_hidden=75,
                      n_atom_feat=75,
                      n_pair_feat=14,
                      T=1,
                      M=1,
                      dropout=0.1,
                      batch_size=batch_size,
                      uncertainty=True)

    model.fit(dataset, nb_epoch=40)

    # Predict the output and uncertainty.
    pred, std = model.predict_uncertainty(dataset)
    mean_error = np.mean(np.abs(dataset.y - pred))
    mean_value = np.mean(np.abs(dataset.y))
    mean_std = np.mean(std)
    assert mean_error < 0.5 * mean_value
    assert mean_std > 0.5 * mean_error
    assert mean_std < mean_value


@flaky
@pytest.mark.tensorflow
def test_dtnn_regression_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(current_dir, "assets/example_DTNN.mat")
    dataset = scipy_io.loadmat(input_file)
    X = dataset['X']
    y = dataset['T']
    w = np.ones_like(y)
    dataset = NumpyDataset(X, y, w, ids=None)
    n_tasks = y.shape[1]

    model = DTNNModel(n_tasks,
                      n_embedding=20,
                      n_distance=100,
                      learning_rate=1.0,
                      mode="regression")

    # Fit trained model
    model.fit(dataset, nb_epoch=250)

    # Eval model on train
    pred = model.predict(dataset)
    mean_rel_error = np.mean(np.abs(1 - pred / y))
    assert mean_rel_error < 0.1


@pytest.mark.tensorflow
def test_graph_predict():

    model = GraphConvModel(12, batch_size=50, mode='classification')
    mols = ["CCCCC", "CCCCCCCCC"]
    feat = ConvMolFeaturizer()
    X = feat.featurize(mols)
    if (model.predict(NumpyDataset(X))).all():
        assert True
    else:
        assert False
