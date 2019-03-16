import unittest
import os
import numpy as np
import scipy

import deepchem as dc
from deepchem.data import NumpyDataset
from deepchem.models import GraphConvModel, DAGModel, WeaveModel, MPNNModel
from deepchem.models import TensorGraph
from deepchem.molnet import load_bace_classification, load_delaney
from deepchem.models.tensorgraph.layers import ReduceSum, L2Loss
from deepchem.feat import ConvMolFeaturizer
from nose.plugins.attrib import attr
from flaky import flaky


class TestGraphModels(unittest.TestCase):

  def get_dataset(self,
                  mode='classification',
                  featurizer='GraphConv',
                  num_tasks=2):
    data_points = 10
    if mode == 'classification':
      tasks, all_dataset, transformers = load_bace_classification(featurizer)
    else:
      tasks, all_dataset, transformers = load_delaney(featurizer)

    train, valid, test = all_dataset
    for i in range(1, num_tasks):
      tasks.append("random_task")
    w = np.ones(shape=(data_points, len(tasks)))

    if mode == 'classification':
      y = np.random.randint(0, 2, size=(data_points, len(tasks)))
      metric = dc.metrics.Metric(
          dc.metrics.roc_auc_score, np.mean, mode="classification")
    else:
      y = np.random.normal(size=(data_points, len(tasks)))
      metric = dc.metrics.Metric(
          dc.metrics.mean_absolute_error, mode="regression")

    ds = NumpyDataset(train.X[:data_points], y, w, train.ids[:data_points])

    return tasks, ds, transformers, metric

  def test_graph_conv_model(self):
    tasks, dataset, transformers, metric = self.get_dataset(
        'classification', 'GraphConv')

    batch_size = 50
    model = GraphConvModel(
        len(tasks), batch_size=batch_size, mode='classification')

    model.fit(dataset, nb_epoch=10)
    scores = model.evaluate(dataset, [metric], transformers)
    assert scores['mean-roc_auc_score'] >= 0.9

    model.save()
    model = TensorGraph.load_from_dir(model.model_dir)
    scores2 = model.evaluate(dataset, [metric], transformers)
    assert np.allclose(scores['mean-roc_auc_score'],
                       scores2['mean-roc_auc_score'])

  def test_neural_fingerprint_retrieval(self):
    tasks, dataset, transformers, metric = self.get_dataset(
        'classification', 'GraphConv')

    fp_size = 3

    batch_size = 50
    model = GraphConvModel(
        len(tasks),
        batch_size=batch_size,
        dense_layer_size=3,
        mode='classification')

    model.fit(dataset, nb_epoch=1)
    neural_fingerprints = model.predict(
        dataset, outputs=model.neural_fingerprint)
    neural_fingerprints = np.array(neural_fingerprints)[:len(dataset)]
    self.assertEqual((len(dataset), fp_size * 2), neural_fingerprints.shape)

  def test_graph_conv_regression_model(self):
    tasks, dataset, transformers, metric = self.get_dataset(
        'regression', 'GraphConv')

    batch_size = 50
    model = GraphConvModel(len(tasks), batch_size=batch_size, mode='regression')

    model.fit(dataset, nb_epoch=100)
    scores = model.evaluate(dataset, [metric], transformers)
    assert all(s < 0.1 for s in scores['mean_absolute_error'])

    model.save()
    model = TensorGraph.load_from_dir(model.model_dir)
    scores2 = model.evaluate(dataset, [metric], transformers)
    assert np.allclose(
        scores['mean_absolute_error'],
        scores2['mean_absolute_error'],
        rtol=1e-4)

  def test_graph_conv_regression_uncertainty(self):
    tasks, dataset, transformers, metric = self.get_dataset(
        'regression', 'GraphConv')

    batch_size = 50
    model = GraphConvModel(
        len(tasks),
        batch_size=batch_size,
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

  def test_graph_conv_atom_features(self):
    tasks, dataset, transformers, metric = self.get_dataset(
        'regression', 'Raw', num_tasks=1)

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
    dataset = dc.data.NumpyDataset(X, np.array(y))
    batch_size = 50
    model = GraphConvModel(
        len(tasks),
        number_atom_features=featurizer.feature_length(),
        batch_size=batch_size,
        mode='regression')

    model.fit(dataset, nb_epoch=1)
    y_pred1 = model.predict(dataset)
    model.save()

    model2 = TensorGraph.load_from_dir(model.model_dir)
    y_pred2 = model2.predict(dataset)
    self.assertTrue(np.allclose(y_pred1, y_pred2))

  def test_change_loss_function(self):
    tasks, dataset, transformers, metric = self.get_dataset(
        'regression', 'GraphConv', num_tasks=1)

    batch_size = 50
    model = GraphConvModel(len(tasks), batch_size=batch_size, mode='regression')

    model.fit(dataset, nb_epoch=1)
    model.save()

    model2 = TensorGraph.load_from_dir(model.model_dir, restore=False)
    dummy_label = model2.labels[-1]
    dummy_ouput = model2.outputs[-1]
    loss = ReduceSum(L2Loss(in_layers=[dummy_label, dummy_ouput]))
    module = model2.create_submodel(loss=loss)
    model2.restore()
    model2.fit(dataset, nb_epoch=1, submodel=module)

  def test_change_loss_function_weave(self):
    tasks, dataset, transformers, metric = self.get_dataset(
        'regression', 'Weave', num_tasks=1)

    batch_size = 50
    model = WeaveModel(
        len(tasks), batch_size=batch_size, mode='regression', use_queue=False)

    model.fit(dataset, nb_epoch=1)
    model.save()

    model2 = TensorGraph.load_from_dir(model.model_dir, restore=False)
    dummy_label = model2.labels[-1]
    dummy_ouput = model2.outputs[-1]
    loss = ReduceSum(L2Loss(in_layers=[dummy_label, dummy_ouput]))
    module = model2.create_submodel(loss=loss)
    model2.restore()
    model2.fit(dataset, nb_epoch=1, submodel=module)

  @attr("slow")
  def test_weave_model(self):
    tasks, dataset, transformers, metric = self.get_dataset(
        'classification', 'Weave')

    model = WeaveModel(len(tasks), mode='classification')

    model.fit(dataset, nb_epoch=50)
    scores = model.evaluate(dataset, [metric], transformers)
    assert scores['mean-roc_auc_score'] >= 0.9

    model.save()
    model = TensorGraph.load_from_dir(model.model_dir)
    scores2 = model.evaluate(dataset, [metric], transformers)
    assert np.allclose(scores['mean-roc_auc_score'],
                       scores2['mean-roc_auc_score'])

  @flaky
  def test_weave_regression_model(self):
    tasks, dataset, transformers, metric = self.get_dataset(
        'regression', 'Weave')

    model = WeaveModel(len(tasks), mode='regression')

    model.fit(dataset, nb_epoch=80)
    scores = model.evaluate(dataset, [metric], transformers)
    assert all(s < 0.1 for s in scores['mean_absolute_error'])

    model.save()
    model = TensorGraph.load_from_dir(model.model_dir)
    scores2 = model.evaluate(dataset, [metric], transformers)
    assert np.allclose(scores['mean_absolute_error'],
                       scores2['mean_absolute_error'])

  @attr("slow")
  def test_dag_model(self):
    tasks, dataset, transformers, metric = self.get_dataset(
        'classification', 'GraphConv')

    max_atoms = max([mol.get_num_atoms() for mol in dataset.X])
    transformer = dc.trans.DAGTransformer(max_atoms=max_atoms)
    dataset = transformer.transform(dataset)

    model = DAGModel(
        len(tasks), max_atoms=max_atoms, mode='classification', use_queue=False)

    model.fit(dataset, nb_epoch=10)
    scores = model.evaluate(dataset, [metric], transformers)
    assert scores['mean-roc_auc_score'] >= 0.9

    model.save()
    model = TensorGraph.load_from_dir(model.model_dir)
    scores2 = model.evaluate(dataset, [metric], transformers)
    assert np.allclose(scores['mean-roc_auc_score'],
                       scores2['mean-roc_auc_score'])

  @attr("slow")
  def test_dag_regression_model(self):
    tasks, dataset, transformers, metric = self.get_dataset(
        'regression', 'GraphConv')

    max_atoms = max([mol.get_num_atoms() for mol in dataset.X])
    transformer = dc.trans.DAGTransformer(max_atoms=max_atoms)
    dataset = transformer.transform(dataset)

    model = DAGModel(
        len(tasks),
        max_atoms=max_atoms,
        mode='regression',
        learning_rate=0.003,
        use_queue=False)

    model.fit(dataset, nb_epoch=100)
    scores = model.evaluate(dataset, [metric], transformers)
    assert all(s < 0.15 for s in scores['mean_absolute_error'])

    model.save()
    model = TensorGraph.load_from_dir(model.model_dir)
    scores2 = model.evaluate(dataset, [metric], transformers)
    assert np.allclose(scores['mean_absolute_error'],
                       scores2['mean_absolute_error'])

  @attr("slow")
  def test_dag_regression_uncertainty(self):
    tasks, dataset, transformers, metric = self.get_dataset(
        'regression', 'GraphConv')

    max_atoms = max([mol.get_num_atoms() for mol in dataset.X])
    transformer = dc.trans.DAGTransformer(max_atoms=max_atoms)
    dataset = transformer.transform(dataset)

    model = DAGModel(
        len(tasks),
        max_atoms=max_atoms,
        mode='regression',
        learning_rate=0.002,
        use_queue=False,
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

  @attr("slow")
  def test_mpnn_model(self):
    tasks, dataset, transformers, metric = self.get_dataset(
        'classification', 'Weave')

    model = MPNNModel(
        len(tasks),
        mode='classification',
        n_hidden=75,
        n_atom_feat=75,
        n_pair_feat=14,
        T=1,
        M=1)

    model.fit(dataset, nb_epoch=20)
    scores = model.evaluate(dataset, [metric], transformers)
    assert scores['mean-roc_auc_score'] >= 0.9

    model.save()
    model = TensorGraph.load_from_dir(model.model_dir)
    scores2 = model.evaluate(dataset, [metric], transformers)
    assert np.allclose(scores['mean-roc_auc_score'],
                       scores2['mean-roc_auc_score'])

  @attr("slow")
  def test_mpnn_regression_model(self):
    tasks, dataset, transformers, metric = self.get_dataset(
        'regression', 'Weave')

    model = MPNNModel(
        len(tasks),
        mode='regression',
        n_hidden=75,
        n_atom_feat=75,
        n_pair_feat=14,
        T=1,
        M=1)

    model.fit(dataset, nb_epoch=50)
    scores = model.evaluate(dataset, [metric], transformers)
    assert all(s < 0.1 for s in scores['mean_absolute_error'])

    model.save()
    model = TensorGraph.load_from_dir(model.model_dir)
    scores2 = model.evaluate(dataset, [metric], transformers)
    assert np.allclose(scores['mean_absolute_error'],
                       scores2['mean_absolute_error'])

  @attr("slow")
  def test_mpnn_regression_uncertainty(self):
    tasks, dataset, transformers, metric = self.get_dataset(
        'regression', 'Weave')

    model = MPNNModel(
        len(tasks),
        mode='regression',
        n_hidden=75,
        n_atom_feat=75,
        n_pair_feat=14,
        T=1,
        M=1,
        dropout=0.1,
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
  def test_dtnn_regression_model(self):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(current_dir, "example_DTNN.mat")
    dataset = scipy.io.loadmat(input_file)
    X = dataset['X']
    y = dataset['T']
    w = np.ones_like(y)
    dataset = dc.data.NumpyDataset(X, y, w, ids=None)
    n_tasks = y.shape[1]

    model = dc.models.DTNNModel(
        n_tasks,
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
