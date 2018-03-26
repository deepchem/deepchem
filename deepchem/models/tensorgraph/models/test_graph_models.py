import unittest

import numpy as np

import deepchem
from deepchem.data import NumpyDataset
from deepchem.models import GraphConvModel
from deepchem.models import TensorGraph
from deepchem.molnet.load_function.delaney_datasets import load_delaney
from deepchem.models.tensorgraph.layers import ReduceSum, L2Loss
from deepchem.models import WeaveModel
from deepchem.feat import ConvMolFeaturizer


class TestGraphModels(unittest.TestCase):

  def get_dataset(self,
                  mode='classification',
                  featurizer='GraphConv',
                  num_tasks=2):
    data_points = 10
    tasks, all_dataset, transformers = load_delaney(featurizer)
    train, valid, test = all_dataset
    for i in range(1, num_tasks):
      tasks.append("random_task")
    w = np.ones(shape=(data_points, len(tasks)))

    if mode == 'classification':
      y = np.random.randint(0, 2, size=(data_points, len(tasks)))
      metric = deepchem.metrics.Metric(
          deepchem.metrics.roc_auc_score, np.mean, mode="classification")
    else:
      y = np.random.normal(size=(data_points, len(tasks)))
      metric = deepchem.metrics.Metric(
          deepchem.metrics.mean_absolute_error, mode="regression")

    ds = NumpyDataset(train.X[:10], y, w, train.ids[:10])

    return tasks, ds, transformers, metric

  def test_graph_conv_model(self):
    tasks, dataset, transformers, metric = self.get_dataset(
        'classification', 'GraphConv')

    batch_size = 50
    model = GraphConvModel(
        len(tasks), batch_size=batch_size, mode='classification')

    model.fit(dataset, nb_epoch=1)
    scores = model.evaluate(dataset, [metric], transformers)

    model.save()
    model = TensorGraph.load_from_dir(model.model_dir)
    scores = model.evaluate(dataset, [metric], transformers)

  def test_graph_conv_regression_model(self):
    tasks, dataset, transformers, metric = self.get_dataset(
        'regression', 'GraphConv')

    batch_size = 50
    model = GraphConvModel(len(tasks), batch_size=batch_size, mode='regression')

    model.fit(dataset, nb_epoch=1)
    scores = model.evaluate(dataset, [metric], transformers)

    model.save()
    model = TensorGraph.load_from_dir(model.model_dir)
    scores = model.evaluate(dataset, [metric], transformers)

  def test_graph_conv_error_bars(self):
    tasks, dataset, transformers, metric = self.get_dataset(
        'regression', 'GraphConv', num_tasks=1)

    batch_size = 50
    model = GraphConvModel(len(tasks), batch_size=batch_size, mode='regression')

    model.fit(dataset, nb_epoch=1)

    mu, sigma = model.bayesian_predict(
        dataset, transformers, untransform=True, n_passes=24)
    assert mu.shape == (len(dataset), len(tasks))
    assert sigma.shape == (len(dataset), len(tasks))

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
      y.append(np.sum(atom_features))

    featurizer = ConvMolFeaturizer(atom_properties=[atom_feature_name])
    X = featurizer.featurize(dataset.X)
    dataset = deepchem.data.NumpyDataset(X, np.array(y))
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
    self.assertTrue(np.all(y_pred1 == y_pred2))

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
