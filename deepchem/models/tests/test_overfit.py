"""
Tests to make sure deepchem models can overfit on tiny datasets.
"""

import os

import numpy as np
import pytest
from flaky import flaky
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

try:
  import tensorflow as tf
  from tensorflow.python.framework import test_util
  has_tensorflow = True
except:
  has_tensorflow = False

import deepchem as dc
from deepchem.models.optimizers import Adam


def test_sklearn_regression_overfit():
  """Test that sklearn models can overfit simple regression datasets."""
  n_samples = 10
  n_features = 3
  n_tasks = 1

  # Generate dummy dataset
  np.random.seed(123)
  ids = np.arange(n_samples)
  X = np.random.rand(n_samples, n_features)
  y = np.random.rand(n_samples, n_tasks)
  w = np.ones((n_samples, n_tasks))
  dataset = dc.data.NumpyDataset(X, y, w, ids)

  regression_metric = dc.metrics.Metric(dc.metrics.r2_score)
  sklearn_model = RandomForestRegressor()
  model = dc.models.SklearnModel(sklearn_model)

  # Fit trained model
  model.fit(dataset)
  model.save()

  # Eval model on train
  scores = model.evaluate(dataset, [regression_metric])
  assert scores[regression_metric.name] > .7


def test_sklearn_classification_overfit():
  """Test that sklearn models can overfit simple classification datasets."""
  n_samples = 10
  n_features = 3
  n_tasks = 1

  # Generate dummy dataset
  np.random.seed(123)
  ids = np.arange(n_samples)
  X = np.random.rand(n_samples, n_features)
  y = np.random.randint(2, size=(n_samples, n_tasks))
  w = np.ones((n_samples, n_tasks))
  dataset = dc.data.NumpyDataset(X, y, w, ids)

  classification_metric = dc.metrics.Metric(dc.metrics.roc_auc_score)
  sklearn_model = RandomForestClassifier()
  model = dc.models.SklearnModel(sklearn_model)

  # Fit trained model
  model.fit(dataset)
  model.save()

  # Eval model on train
  scores = model.evaluate(dataset, [classification_metric])
  assert scores[classification_metric.name] > .9


def test_sklearn_skewed_classification_overfit():
  """Test sklearn models can overfit 0/1 datasets with few actives."""
  n_samples = 100
  n_features = 3
  n_tasks = 1

  # Generate dummy dataset
  np.random.seed(123)
  p = .05
  ids = np.arange(n_samples)
  X = np.random.rand(n_samples, n_features)
  y = np.random.binomial(1, p, size=(n_samples, n_tasks))
  w = np.ones((n_samples, n_tasks))

  dataset = dc.data.NumpyDataset(X, y, w, ids)

  classification_metric = dc.metrics.Metric(dc.metrics.roc_auc_score)
  sklearn_model = RandomForestClassifier()
  model = dc.models.SklearnModel(sklearn_model)

  # Fit trained model
  model.fit(dataset)
  model.save()

  # Eval model on train
  scores = model.evaluate(dataset, [classification_metric])
  assert scores[classification_metric.name] > .9


@pytest.mark.torch
def test_regression_overfit():
  """Test that MultitaskRegressor can overfit simple regression datasets."""
  n_samples = 10
  n_features = 3
  n_tasks = 1

  # Generate dummy dataset
  np.random.seed(123)
  ids = np.arange(n_samples)
  X = np.random.rand(n_samples, n_features)
  y = np.zeros((n_samples, n_tasks))
  w = np.ones((n_samples, n_tasks))
  dataset = dc.data.NumpyDataset(X, y, w, ids)

  regression_metric = dc.metrics.Metric(dc.metrics.mean_squared_error)
  # TODO(rbharath): This breaks with optimizer="momentum". Why?
  model = dc.models.MultitaskRegressor(
      n_tasks,
      n_features,
      dropouts=[0.],
      weight_init_stddevs=[np.sqrt(6) / np.sqrt(1000)],
      batch_size=n_samples,
      learning_rate=0.003)

  # Fit trained model
  model.fit(dataset, nb_epoch=100)

  # Eval model on train
  scores = model.evaluate(dataset, [regression_metric])
  assert scores[regression_metric.name] < 0.007


@pytest.mark.torch
def test_classification_overfit():
  """Test that MultitaskClassifier can overfit simple classification datasets."""
  n_samples = 10
  n_features = 3
  n_tasks = 1
  n_classes = 2

  # Generate dummy dataset
  np.random.seed(123)
  ids = np.arange(n_samples)
  X = np.random.rand(n_samples, n_features)
  y = np.zeros((n_samples, n_tasks))
  w = np.ones((n_samples, n_tasks))
  dataset = dc.data.NumpyDataset(X, y, w, ids)

  classification_metric = dc.metrics.Metric(dc.metrics.accuracy_score)
  model = dc.models.MultitaskClassifier(
      n_tasks,
      n_features,
      dropouts=[0.],
      weight_init_stddevs=[.1],
      batch_size=n_samples,
      optimizer=Adam(learning_rate=0.0003, beta1=0.9, beta2=0.999))

  # Fit trained model
  model.fit(dataset, nb_epoch=100)

  # Eval model on train
  scores = model.evaluate(dataset, [classification_metric])
  assert scores[classification_metric.name] > .9


@pytest.mark.torch
def test_residual_classification_overfit():
  """Test that a residual network can overfit simple classification datasets."""
  n_samples = 10
  n_features = 5
  n_tasks = 1
  n_classes = 2

  # Generate dummy dataset
  np.random.seed(123)
  ids = np.arange(n_samples)
  X = np.random.rand(n_samples, n_features)
  y = np.random.randint(2, size=(n_samples, n_tasks))
  w = np.ones((n_samples, n_tasks))
  dataset = dc.data.NumpyDataset(X, y, w, ids)

  classification_metric = dc.metrics.Metric(dc.metrics.accuracy_score)
  model = dc.models.MultitaskClassifier(
      n_tasks,
      n_features,
      layer_sizes=[20] * 10,
      dropouts=0.0,
      batch_size=n_samples,
      residual=True)

  # Fit trained model
  model.fit(dataset, nb_epoch=500)

  # Eval model on train
  scores = model.evaluate(dataset, [classification_metric])
  assert scores[classification_metric.name] > .9


@flaky
@pytest.mark.torch
def test_fittransform_regression_overfit():
  """Test that MultitaskFitTransformRegressor can overfit simple regression datasets."""
  n_samples = 10
  n_features = 3
  n_tasks = 1

  # Generate dummy dataset
  np.random.seed(123)
  ids = np.arange(n_samples)
  X = np.random.rand(n_samples, n_features, n_features)
  y = np.zeros((n_samples, n_tasks))
  w = np.ones((n_samples, n_tasks))
  dataset = dc.data.NumpyDataset(X, y, w, ids)

  fit_transformers = [dc.trans.CoulombFitTransformer(dataset)]
  regression_metric = dc.metrics.Metric(dc.metrics.mean_squared_error)
  model = dc.models.MultitaskFitTransformRegressor(
      n_tasks, [n_features, n_features],
      dropouts=[0.01],
      weight_init_stddevs=[np.sqrt(6) / np.sqrt(1000)],
      batch_size=n_samples,
      fit_transformers=fit_transformers,
      n_evals=1,
      optimizer=Adam(learning_rate=0.003, beta1=0.9, beta2=0.999))

  # Fit trained model
  model.fit(dataset, nb_epoch=100)

  # Eval model on train
  scores = model.evaluate(dataset, [regression_metric])
  assert scores[regression_metric.name] < .1


@pytest.mark.torch
def test_skewed_classification_overfit():
  """Test MultitaskClassifier can overfit 0/1 datasets with few actives."""
  # n_samples = 100
  n_samples = 100
  n_features = 3
  n_tasks = 1
  n_classes = 2

  # Generate dummy dataset
  np.random.seed(123)
  p = .05
  ids = np.arange(n_samples)
  X = np.random.rand(n_samples, n_features)
  y = np.random.binomial(1, p, size=(n_samples, n_tasks))
  w = np.ones((n_samples, n_tasks))

  dataset = dc.data.NumpyDataset(X, y, w, ids)

  classification_metric = dc.metrics.Metric(dc.metrics.roc_auc_score)
  model = dc.models.MultitaskClassifier(
      n_tasks,
      n_features,
      dropouts=[0.],
      weight_init_stddevs=[.1],
      batch_size=n_samples,
      learning_rate=0.003)

  # Fit trained model
  model.fit(dataset, nb_epoch=100)

  # Eval model on train
  scores = model.evaluate(dataset, [classification_metric])
  assert scores[classification_metric.name] > .75


@pytest.mark.torch
def test_skewed_missing_classification_overfit():
  """MultitaskClassifier, skewed data, few actives

  Test MultitaskClassifier overfit 0/1 datasets with missing data and few
  actives. This is intended to be as close to singletask MUV datasets as
  possible.
  """
  n_samples = 5120
  n_features = 6
  n_tasks = 1
  n_classes = 2

  # Generate dummy dataset
  np.random.seed(123)
  p = .002
  ids = np.arange(n_samples)
  X = np.random.rand(n_samples, n_features)
  y = np.random.binomial(1, p, size=(n_samples, n_tasks))
  w = np.ones((n_samples, n_tasks))
  y_flat, w_flat = np.squeeze(y), np.squeeze(w)
  y_nonzero = y_flat[w_flat != 0]
  num_nonzero = np.count_nonzero(y_nonzero)
  weight_nonzero = len(y_nonzero) / num_nonzero
  w_flat[y_flat != 0] = weight_nonzero
  w = np.reshape(w_flat, (n_samples, n_tasks))

  dataset = dc.data.DiskDataset.from_numpy(X, y, w, ids)

  classification_metric = dc.metrics.Metric(dc.metrics.roc_auc_score)
  model = dc.models.MultitaskClassifier(
      n_tasks,
      n_features,
      dropouts=[0.],
      weight_init_stddevs=[1.],
      batch_size=n_samples,
      learning_rate=0.003)

  # Fit trained model
  model.fit(dataset, nb_epoch=100)

  # Eval model on train
  scores = model.evaluate(dataset, [classification_metric])
  assert scores[classification_metric.name] > .7


def test_sklearn_multitask_classification_overfit():
  """Test SKLearn singletask-to-multitask overfits tiny data."""
  n_tasks = 10
  tasks = ["task%d" % task for task in range(n_tasks)]
  n_samples = 10
  n_features = 3

  # Generate dummy dataset
  np.random.seed(123)
  ids = np.arange(n_samples)
  X = np.random.rand(n_samples, n_features)
  y = np.random.randint(2, size=(n_samples, n_tasks))
  w = np.ones((n_samples, n_tasks))
  dataset = dc.data.DiskDataset.from_numpy(X, y, w, ids)

  classification_metric = dc.metrics.Metric(
      dc.metrics.roc_auc_score, task_averager=np.mean)

  def model_builder(model_dir):
    sklearn_model = RandomForestClassifier()
    return dc.models.SklearnModel(sklearn_model, model_dir)

  model = dc.models.SingletaskToMultitask(tasks, model_builder)

  # Fit trained model
  model.fit(dataset)
  model.save()

  # Eval model on train
  scores = model.evaluate(dataset, [classification_metric])
  assert scores[classification_metric.name] > .9


@flaky
@pytest.mark.torch
def test_multitask_classification_overfit():
  """Test MultitaskClassifier overfits tiny data."""
  n_tasks = 10
  n_samples = 10
  n_features = 3
  n_classes = 2

  # Generate dummy dataset
  np.random.seed(123)
  ids = np.arange(n_samples)
  X = np.random.rand(n_samples, n_features)
  y = np.zeros((n_samples, n_tasks))
  w = np.ones((n_samples, n_tasks))
  dataset = dc.data.NumpyDataset(X, y, w, ids)

  classification_metric = dc.metrics.Metric(
      dc.metrics.accuracy_score, task_averager=np.mean, n_tasks=n_tasks)
  model = dc.models.MultitaskClassifier(
      n_tasks,
      n_features,
      dropouts=[0.],
      weight_init_stddevs=[.1],
      batch_size=n_samples,
      optimizer=Adam(learning_rate=0.0003, beta1=0.9, beta2=0.999))

  # Fit trained model
  model.fit(dataset)

  # Eval model on train
  scores = model.evaluate(dataset, [classification_metric])
  assert scores[classification_metric.name] > .9


@flaky
@pytest.mark.torch
def test_multitask_classification_regularization():
  """Test regularizing a MultitaskClassifier."""
  n_tasks = 10
  n_samples = 10
  n_features = 3
  n_classes = 2

  # Generate dummy dataset
  np.random.seed(123)
  ids = np.arange(n_samples)
  X = np.random.rand(n_samples, n_features)
  y = np.zeros((n_samples, n_tasks))
  w = np.ones((n_samples, n_tasks))
  dataset = dc.data.NumpyDataset(X, y, w, ids)

  classification_metric = dc.metrics.Metric(
      dc.metrics.accuracy_score, task_averager=np.mean, n_tasks=n_tasks)
  model = dc.models.MultitaskClassifier(
      n_tasks,
      n_features,
      layer_sizes=[1000],
      dropouts=0,
      weight_decay_penalty=1.0,
      weight_decay_penalty_type='l1',
      batch_size=n_samples,
      learning_rate=0.0003)

  # Fit trained model
  model.fit(dataset, nb_epoch=500)

  # Eval model on train
  scores = model.evaluate(dataset, [classification_metric])
  assert scores[classification_metric.name] > .9

  # Most weights should be close to zero.

  elements = 0.0
  num_nonzero = 0.0
  for p in model.model.parameters():
    if len(p.shape) == 2 and p.shape[0] == 1000:
      elements += p.numel()
      num_nonzero += (p.abs() > 1e-3).sum()
  assert num_nonzero / elements < 0.1


@pytest.mark.tensorflow
def test_robust_multitask_classification_overfit():
  """Test robust multitask overfits tiny data."""
  n_tasks = 10
  n_samples = 10
  n_features = 3
  n_classes = 2

  # Generate dummy dataset
  np.random.seed(123)
  ids = np.arange(n_samples)
  X = np.random.rand(n_samples, n_features)
  y = np.zeros((n_samples, n_tasks))
  w = np.ones((n_samples, n_tasks))
  dataset = dc.data.NumpyDataset(X, y, w, ids)

  classification_metric = dc.metrics.Metric(
      dc.metrics.accuracy_score, task_averager=np.mean)
  model = dc.models.RobustMultitaskClassifier(
      n_tasks,
      n_features,
      layer_sizes=[50],
      bypass_layer_sizes=[10],
      dropouts=[0.],
      learning_rate=0.003,
      weight_init_stddevs=[.1],
      batch_size=n_samples)

  # Fit trained model
  model.fit(dataset, nb_epoch=25)

  # Eval model on train
  scores = model.evaluate(dataset, [classification_metric])
  assert scores[classification_metric.name] > .9


@pytest.mark.tensorflow
def test_IRV_multitask_classification_overfit():
  """Test IRV classifier overfits tiny data."""
  n_tasks = 5
  n_samples = 10
  n_features = 128
  n_classes = 2

  # Generate dummy dataset
  np.random.seed(123)
  ids = np.arange(n_samples)
  X = np.random.randint(2, size=(n_samples, n_features))
  y = np.ones((n_samples, n_tasks))
  w = np.ones((n_samples, n_tasks))
  dataset = dc.data.NumpyDataset(X, y, w, ids)
  IRV_transformer = dc.trans.IRVTransformer(5, n_tasks, dataset)
  dataset_trans = IRV_transformer.transform(dataset)
  classification_metric = dc.metrics.Metric(
      dc.metrics.accuracy_score, task_averager=np.mean)
  model = dc.models.MultitaskIRVClassifier(
      n_tasks, K=5, learning_rate=0.01, batch_size=n_samples)

  # Fit trained model
  model.fit(dataset_trans)

  # Eval model on train
  scores = model.evaluate(dataset_trans, [classification_metric])
  assert scores[classification_metric.name] > .9


def test_sklearn_multitask_regression_overfit():
  """Test SKLearn singletask-to-multitask overfits tiny regression data."""
  n_tasks = 2
  tasks = ["task%d" % task for task in range(n_tasks)]
  n_samples = 10
  n_features = 3

  # Generate dummy dataset
  np.random.seed(123)
  ids = np.arange(n_samples)
  X = np.random.rand(n_samples, n_features)
  y = np.random.rand(n_samples, n_tasks)
  w = np.ones((n_samples, n_tasks))

  dataset = dc.data.DiskDataset.from_numpy(X, y, w, ids)

  regression_metric = dc.metrics.Metric(
      dc.metrics.r2_score, task_averager=np.mean)

  def model_builder(model_dir):
    sklearn_model = RandomForestRegressor()
    return dc.models.SklearnModel(sklearn_model, model_dir)

  model = dc.models.SingletaskToMultitask(tasks, model_builder)

  # Fit trained model
  model.fit(dataset)
  model.save()

  # Eval model on train
  scores = model.evaluate(dataset, [regression_metric])
  assert scores[regression_metric.name] > .7


@pytest.mark.torch
def test_multitask_regression_overfit():
  """Test MultitaskRegressor overfits tiny data."""
  n_tasks = 10
  n_samples = 10
  n_features = 10
  n_classes = 2

  # Generate dummy dataset
  np.random.seed(123)
  ids = np.arange(n_samples)
  X = np.random.rand(n_samples, n_features)
  y = np.random.rand(n_samples, n_tasks)
  w = np.ones((n_samples, n_tasks))

  dataset = dc.data.NumpyDataset(X, y, w, ids)

  regression_metric = dc.metrics.Metric(
      dc.metrics.mean_squared_error, task_averager=np.mean, mode="regression")
  model = dc.models.MultitaskRegressor(
      n_tasks, n_features, dropouts=0.0, batch_size=n_samples)

  # Fit trained model
  model.fit(dataset, nb_epoch=1000)

  # Eval model on train
  scores = model.evaluate(dataset, [regression_metric])
  assert scores[regression_metric.name] < .02


@pytest.mark.torch
def test_multitask_regression_regularization():
  """Test regularizing a MultitaskRegressor."""
  n_tasks = 10
  n_samples = 10
  n_features = 10
  n_classes = 2

  # Generate dummy dataset
  np.random.seed(123)
  ids = np.arange(n_samples)
  X = np.random.rand(n_samples, n_features)
  y = np.random.rand(n_samples, n_tasks)
  w = np.ones((n_samples, n_tasks))

  dataset = dc.data.NumpyDataset(X, y, w, ids)

  regression_metric = dc.metrics.Metric(
      dc.metrics.mean_squared_error, task_averager=np.mean, mode="regression")
  model = dc.models.MultitaskRegressor(
      n_tasks,
      n_features,
      dropouts=0.0,
      batch_size=n_samples,
      weight_decay_penalty=0.01,
      weight_decay_penalty_type='l1')

  # Fit trained model
  model.fit(dataset, nb_epoch=1000)

  # Eval model on train
  scores = model.evaluate(dataset, [regression_metric])
  assert scores[regression_metric.name] < 0.1

  # Most weights should be close to zero.

  elements = 0.0
  num_nonzero = 0.0
  for p in model.model.parameters():
    if len(p.shape) == 2 and p.shape[0] == 1000:
      elements += p.numel()
      num_nonzero += (p.abs() > 1e-3).sum()
  assert num_nonzero / elements < 0.1


@pytest.mark.torch
def test_residual_regression_overfit():
  """Test that a residual multitask network can overfit tiny data."""
  n_tasks = 10
  n_samples = 10
  n_features = 10
  n_classes = 2

  # Generate dummy dataset
  np.random.seed(123)
  ids = np.arange(n_samples)
  X = np.random.rand(n_samples, n_features)
  y = np.random.rand(n_samples, n_tasks)
  w = np.ones((n_samples, n_tasks))

  dataset = dc.data.NumpyDataset(X, y, w, ids)

  regression_metric = dc.metrics.Metric(
      dc.metrics.mean_squared_error, task_averager=np.mean, mode="regression")
  model = dc.models.MultitaskRegressor(
      n_tasks,
      n_features,
      layer_sizes=[20] * 10,
      dropouts=0.0,
      batch_size=n_samples,
      residual=True)

  # Fit trained model
  model.fit(dataset, nb_epoch=1000)

  # Eval model on train
  scores = model.evaluate(dataset, [regression_metric])
  assert scores[regression_metric.name] < .02


@pytest.mark.tensorflow
def test_robust_multitask_regression_overfit():
  """Test robust multitask overfits tiny data."""
  np.random.seed(123)
  tf.random.set_seed(123)
  n_tasks = 10
  n_samples = 10
  n_features = 3
  n_classes = 2

  # Generate dummy dataset
  np.random.seed(123)
  ids = np.arange(n_samples)
  X = np.random.rand(n_samples, n_features)
  y = np.zeros((n_samples, n_tasks))
  w = np.ones((n_samples, n_tasks))

  dataset = dc.data.NumpyDataset(X, y, w, ids)

  regression_metric = dc.metrics.Metric(
      dc.metrics.mean_squared_error, task_averager=np.mean, mode="regression")
  model = dc.models.RobustMultitaskRegressor(
      n_tasks,
      n_features,
      layer_sizes=[50],
      bypass_layer_sizes=[10],
      dropouts=[0.],
      learning_rate=0.003,
      weight_init_stddevs=[.1],
      batch_size=n_samples)

  # Fit trained model
  model.fit(dataset, nb_epoch=25)

  # Eval model on train
  scores = model.evaluate(dataset, [regression_metric])
  assert scores[regression_metric.name] < .2


@pytest.mark.tensorflow
def test_progressive_classification_overfit():
  """Test progressive multitask overfits tiny data."""
  np.random.seed(123)
  n_tasks = 5
  n_samples = 10
  n_features = 6

  # Generate dummy dataset
  np.random.seed(123)
  ids = np.arange(n_samples)
  X = np.random.rand(n_samples, n_features)
  y = np.random.randint(2, size=(n_samples, n_tasks))
  w = np.ones((n_samples, n_tasks))

  dataset = dc.data.NumpyDataset(X, y, w, ids)

  metric = dc.metrics.Metric(dc.metrics.accuracy_score, task_averager=np.mean)
  model = dc.models.ProgressiveMultitaskClassifier(
      n_tasks,
      n_features,
      layer_sizes=[50],
      bypass_layer_sizes=[10],
      dropouts=[0.],
      learning_rate=0.001,
      weight_init_stddevs=[.1],
      alpha_init_stddevs=[.02],
      batch_size=n_samples)

  # Fit trained model
  model.fit(dataset, nb_epoch=300)

  # Eval model on train
  scores = model.evaluate(dataset, [metric])
  assert scores[metric.name] > .9


@pytest.mark.tensorflow
def test_progressive_regression_overfit():
  """Test progressive multitask overfits tiny data."""
  np.random.seed(123)
  n_tasks = 5
  n_samples = 10
  n_features = 6

  # Generate dummy dataset
  np.random.seed(123)
  ids = np.arange(n_samples)
  X = np.random.rand(n_samples, n_features)
  y = np.random.rand(n_samples, n_tasks)
  w = np.ones((n_samples, n_tasks))

  dataset = dc.data.NumpyDataset(X, y, w, ids)

  metric = dc.metrics.Metric(dc.metrics.rms_score, task_averager=np.mean)
  model = dc.models.ProgressiveMultitaskRegressor(
      n_tasks,
      n_features,
      layer_sizes=[50],
      bypass_layer_sizes=[10],
      dropouts=[0.],
      learning_rate=0.002,
      weight_init_stddevs=[.1],
      alpha_init_stddevs=[.02],
      batch_size=n_samples)

  # Fit trained model
  model.fit(dataset, nb_epoch=200)

  # Eval model on train
  scores = model.evaluate(dataset, [metric])
  assert scores[metric.name] < .2


@pytest.mark.torch
def test_multitask_regressor_uncertainty():
  """Test computing uncertainty for a MultitaskRegressor."""
  n_tasks = 1
  n_samples = 30
  n_features = 1
  noise = 0.1

  # Generate dummy dataset
  X = np.random.rand(n_samples, n_features, 1)
  y = 10 * X + np.random.normal(scale=noise, size=(n_samples, n_tasks, 1))
  dataset = dc.data.NumpyDataset(X, y)

  model = dc.models.MultitaskRegressor(
      n_tasks,
      n_features,
      layer_sizes=[200],
      weight_init_stddevs=[.1],
      batch_size=n_samples,
      dropouts=0.1,
      learning_rate=0.003,
      uncertainty=True)

  # Fit trained model
  model.fit(dataset, nb_epoch=2500)

  # Predict the output and uncertainty.
  pred, std = model.predict_uncertainty(dataset)
  assert np.mean(np.abs(y - pred)) < 1.0
  assert noise < np.mean(std) < 1.0


@pytest.mark.torch
def test_multitask_regressor_delaney_uncertainty():
  """Test computing uncertainty on a larger dataset."""
  tasks, datasets, transformers = dc.molnet.load_delaney('ECFP')
  train_dataset, valid_dataset, test_dataset = datasets
  model = dc.models.MultitaskRegressor(len(tasks), 1024, uncertainty=True)
  model.fit(train_dataset, nb_epoch=20)
  metric = dc.metrics.Metric(dc.metrics.pearsonr)
  scores = model.evaluate(test_dataset, [metric], transformers)
  assert scores['pearsonr'] > 0.5


@pytest.mark.slow
@pytest.mark.tensorflow
def test_DAG_singletask_regression_overfit():
  """Test DAG regressor multitask overfits tiny data."""
  np.random.seed(123)
  tf.random.set_seed(123)
  n_tasks = 1
  current_dir = os.path.dirname(os.path.abspath(__file__))

  # Load mini log-solubility dataset.
  featurizer = dc.feat.ConvMolFeaturizer()
  tasks = ["outcome"]
  input_file = os.path.join(current_dir, "example_regression.csv")
  loader = dc.data.CSVLoader(
      tasks=tasks, feature_field="smiles", featurizer=featurizer)
  dataset = loader.create_dataset(input_file)

  regression_metric = dc.metrics.Metric(
      dc.metrics.pearson_r2_score, task_averager=np.mean)

  n_feat = 75
  batch_size = 10
  transformer = dc.trans.DAGTransformer(max_atoms=50)
  dataset = transformer.transform(dataset)

  model = dc.models.DAGModel(
      n_tasks,
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


@pytest.mark.tensorflow
def test_weave_singletask_classification_overfit():
  """Test weave model overfits tiny data."""
  np.random.seed(123)
  tf.random.set_seed(123)
  n_tasks = 1
  current_dir = os.path.dirname(os.path.abspath(__file__))

  # Load mini log-solubility dataset.
  featurizer = dc.feat.WeaveFeaturizer()
  tasks = ["outcome"]
  input_file = os.path.join(current_dir, "example_classification.csv")
  loader = dc.data.CSVLoader(
      tasks=tasks, feature_field="smiles", featurizer=featurizer)
  dataset = loader.create_dataset(input_file)

  classification_metric = dc.metrics.Metric(dc.metrics.accuracy_score)

  batch_size = 10
  model = dc.models.WeaveModel(
      n_tasks,
      batch_size=batch_size,
      learning_rate=0.0003,
      dropout=0.0,
      mode="classification")

  # Fit trained model
  model.fit(dataset, nb_epoch=100)

  # Eval model on train
  scores = model.evaluate(dataset, [classification_metric])

  assert scores[classification_metric.name] > .65


@pytest.mark.slow
@pytest.mark.tensorflow
def test_weave_singletask_regression_overfit():
  """Test weave model overfits tiny data."""
  np.random.seed(123)
  tf.random.set_seed(123)
  n_tasks = 1
  current_dir = os.path.dirname(os.path.abspath(__file__))

  # Load mini log-solubility dataset.
  featurizer = dc.feat.WeaveFeaturizer()
  tasks = ["outcome"]
  input_file = os.path.join(current_dir, "example_regression.csv")
  loader = dc.data.CSVLoader(
      tasks=tasks, feature_field="smiles", featurizer=featurizer)
  dataset = loader.create_dataset(input_file)

  regression_metric = dc.metrics.Metric(
      dc.metrics.pearson_r2_score, task_averager=np.mean)

  batch_size = 10

  model = dc.models.WeaveModel(
      n_tasks,
      batch_size=batch_size,
      learning_rate=0.0003,
      dropout=0.0,
      mode="regression")

  # Fit trained model
  model.fit(dataset, nb_epoch=200)

  # Eval model on train
  scores = model.evaluate(dataset, [regression_metric])

  assert scores[regression_metric.name] > .8


@pytest.mark.slow
@pytest.mark.tensorflow
def test_MPNN_singletask_regression_overfit():
  """Test MPNN overfits tiny data."""
  np.random.seed(123)
  tf.random.set_seed(123)
  n_tasks = 1
  current_dir = os.path.dirname(os.path.abspath(__file__))

  # Load mini log-solubility dataset.
  featurizer = dc.feat.WeaveFeaturizer()
  tasks = ["outcome"]
  input_file = os.path.join(current_dir, "example_regression.csv")
  loader = dc.data.CSVLoader(
      tasks=tasks, feature_field="smiles", featurizer=featurizer)
  dataset = loader.create_dataset(input_file)

  regression_metric = dc.metrics.Metric(
      dc.metrics.pearson_r2_score, task_averager=np.mean)

  n_atom_feat = 75
  n_pair_feat = 14
  batch_size = 10
  model = dc.models.MPNNModel(
      n_tasks,
      n_atom_feat=n_atom_feat,
      n_pair_feat=n_pair_feat,
      T=2,
      M=3,
      batch_size=batch_size,
      learning_rate=0.001,
      use_queue=False,
      mode="regression")

  # Fit trained model
  model.fit(dataset, nb_epoch=50)

  # Eval model on train
  scores = model.evaluate(dataset, [regression_metric])

  assert scores[regression_metric.name] > .8


@pytest.mark.tensorflow
def test_textCNN_singletask_classification_overfit():
  """Test textCNN model overfits tiny data."""
  np.random.seed(123)
  tf.random.set_seed(123)
  n_tasks = 1
  current_dir = os.path.dirname(os.path.abspath(__file__))

  featurizer = dc.feat.RawFeaturizer()
  tasks = ["outcome"]
  input_file = os.path.join(current_dir, "example_classification.csv")
  loader = dc.data.CSVLoader(
      tasks=tasks, feature_field="smiles", featurizer=featurizer)
  dataset = loader.create_dataset(input_file)

  classification_metric = dc.metrics.Metric(dc.metrics.accuracy_score)

  char_dict, length = dc.models.TextCNNModel.build_char_dict(dataset)
  batch_size = 10

  model = dc.models.TextCNNModel(
      n_tasks,
      char_dict,
      seq_length=length,
      batch_size=batch_size,
      learning_rate=0.001,
      use_queue=False,
      mode="classification")

  # Fit trained model
  model.fit(dataset, nb_epoch=200)

  # Eval model on train
  scores = model.evaluate(dataset, [classification_metric])

  assert scores[classification_metric.name] > .8


@flaky()
@pytest.mark.tensorflow
def test_textCNN_singletask_regression_overfit():
  """Test textCNN model overfits tiny data."""
  np.random.seed(123)
  tf.random.set_seed(123)
  n_tasks = 1
  current_dir = os.path.dirname(os.path.abspath(__file__))

  # Load mini log-solubility dataset.
  featurizer = dc.feat.RawFeaturizer()
  tasks = ["outcome"]
  input_file = os.path.join(current_dir, "example_regression.csv")
  loader = dc.data.CSVLoader(
      tasks=tasks, feature_field="smiles", featurizer=featurizer)
  dataset = loader.create_dataset(input_file)

  regression_metric = dc.metrics.Metric(
      dc.metrics.pearson_r2_score, task_averager=np.mean)

  char_dict, length = dc.models.TextCNNModel.build_char_dict(dataset)
  batch_size = 10

  model = dc.models.TextCNNModel(
      n_tasks,
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
