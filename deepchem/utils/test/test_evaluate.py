"""Unit tests for evaluators."""
import pytest
import deepchem as dc
import numpy as np
import sklearn
from deepchem.utils.evaluate import Evaluator
from deepchem.utils.evaluate import GeneratorEvaluator
try:
  import tensorflow as tf  # noqa
  has_tensorflow = True
except:
  has_tensorflow = False

try:
  import torch  # noqa
  has_pytorch = True
except:
  has_pytorch = False


def test_multiclass_threshold_predictions():
  """Check prediction thresholding works correctly."""
  # Construct a random class probability matrix
  y = np.random.rand(10, 5)
  y_sums = np.sum(y, axis=1)
  y = y / y_sums[:, None]
  y_out = dc.metrics.threshold_predictions(y)
  assert y_out.shape == (10,)
  assert np.allclose(y_out, np.argmax(y, axis=1))


def test_binary_threshold_predictions():
  """Check prediction thresholding works correctly."""
  # Construct a random class probability matrix
  y = np.random.rand(10, 2)
  y_sums = np.sum(y, axis=1)
  y = y / y_sums[:, None]
  y_out = dc.metrics.threshold_predictions(y, threshold=0.3)
  assert y_out.shape == (10,)
  assert np.allclose(y_out, np.where(y[:, 1] >= 0.3, np.ones(10), np.zeros(10)))


@pytest.mark.torch
def test_evaluator_dc_metric():
  """Test an evaluator on a dataset."""
  X = np.random.rand(10, 5)
  y = np.random.rand(10, 1)
  dataset = dc.data.NumpyDataset(X, y)
  model = dc.models.MultitaskRegressor(1, 5)
  evaluator = Evaluator(model, dataset, [])
  metric = dc.metrics.Metric(dc.metrics.mae_score)
  multitask_scores = evaluator.compute_model_performance(metric)
  assert isinstance(multitask_scores, dict)
  assert len(multitask_scores) == 1
  assert multitask_scores['mae_score'] > 0


@pytest.mark.torch
def test_multiclass_classification_singletask():
  """Test multiclass classification evaluation."""
  X = np.random.rand(100, 5)
  y = np.random.randint(5, size=(100,))
  dataset = dc.data.NumpyDataset(X, y)
  model = dc.models.MultitaskClassifier(1, 5, n_classes=5)
  evaluator = Evaluator(model, dataset, [])
  multitask_scores = evaluator.compute_model_performance(
      dc.metrics.roc_auc_score, n_classes=5)
  assert len(multitask_scores) == 1
  assert multitask_scores["metric-1"] >= 0


def test_sklearn_multiclass_classification_singletask():
  """Test multiclass classification evaluation."""
  X = np.random.rand(100, 5)
  y = np.random.randint(5, size=(100,))
  dataset = dc.data.NumpyDataset(X, y)
  rf = sklearn.ensemble.RandomForestClassifier(50)
  model = dc.models.SklearnModel(rf)
  model.fit(dataset)
  evaluator = Evaluator(model, dataset, [])
  multitask_scores = evaluator.compute_model_performance(
      dc.metrics.roc_auc_score, n_classes=5)
  assert len(multitask_scores) == 1
  assert multitask_scores["metric-1"] >= 0


@pytest.mark.torch
def test_evaluate_multiclass_classification_singletask():
  """Test multiclass classification evaluation."""
  X = np.random.rand(100, 5)
  y = np.random.randint(5, size=(100,))
  dataset = dc.data.NumpyDataset(X, y)
  model = dc.models.MultitaskClassifier(1, 5, n_classes=5)
  multitask_scores = model.evaluate(
      dataset, dc.metrics.roc_auc_score, n_classes=5)
  assert len(multitask_scores) == 1
  assert multitask_scores["metric-1"] >= 0


@pytest.mark.torch
def test_multitask_evaluator():
  """Test evaluation of a multitask metric."""
  X = np.random.rand(10, 5)
  y = np.random.rand(10, 2, 1)
  dataset = dc.data.NumpyDataset(X, y)
  model = dc.models.MultitaskRegressor(2, 5)
  evaluator = Evaluator(model, dataset, [])
  metric = dc.metrics.Metric(dc.metrics.mae_score)
  multitask_scores, all_task_scores = evaluator.compute_model_performance(
      metric, per_task_metrics=True)
  assert isinstance(multitask_scores, dict)
  assert len(multitask_scores) == 1
  assert multitask_scores['mae_score'] > 0
  assert isinstance(all_task_scores, dict)
  assert len(multitask_scores) == 1


@pytest.mark.torch
def test_model_evaluate_dc_metric():
  """Test a model evaluate on a dataset."""
  X = np.random.rand(10, 5)
  y = np.random.rand(10, 1)
  dataset = dc.data.NumpyDataset(X, y)
  model = dc.models.MultitaskRegressor(1, 5)
  metric = dc.metrics.Metric(dc.metrics.mae_score)
  multitask_scores = model.evaluate(dataset, metric, [])
  assert isinstance(multitask_scores, dict)
  assert len(multitask_scores) == 1
  assert multitask_scores['mae_score'] > 0


@pytest.mark.torch
def test_multitask_model_evaluate_sklearn():
  """Test evaluation of a multitask metric."""
  X = np.random.rand(10, 5)
  y = np.random.rand(10, 2)
  dataset = dc.data.NumpyDataset(X, y)
  model = dc.models.MultitaskRegressor(2, 5)
  evaluator = Evaluator(model, dataset, [])
  multitask_scores, all_task_scores = evaluator.compute_model_performance(
      dc.metrics.mean_absolute_error, per_task_metrics=True)
  assert isinstance(multitask_scores, dict)
  assert len(multitask_scores) == 1
  assert multitask_scores['metric-1'] > 0
  assert isinstance(all_task_scores, dict)
  assert len(multitask_scores) == 1


@pytest.mark.torch
def test_multitask_model_evaluate():
  """Test evaluation of a multitask metric."""
  X = np.random.rand(10, 5)
  y = np.random.rand(10, 2)
  dataset = dc.data.NumpyDataset(X, y)
  model = dc.models.MultitaskRegressor(2, 5)
  multitask_scores, all_task_scores = model.evaluate(
      dataset, dc.metrics.mean_absolute_error, per_task_metrics=True)
  assert isinstance(multitask_scores, dict)
  assert len(multitask_scores) == 1
  assert multitask_scores["metric-1"] > 0
  assert isinstance(all_task_scores, dict)


@pytest.mark.torch
def test_evaluator_dc_multi_metric():
  """Test an evaluator on a dataset."""
  X = np.random.rand(10, 5)
  y = np.random.rand(10, 1)
  dataset = dc.data.NumpyDataset(X, y)
  model = dc.models.MultitaskRegressor(1, 5)
  evaluator = Evaluator(model, dataset, [])
  metric1 = dc.metrics.Metric(dc.metrics.mae_score, n_tasks=2)
  metric2 = dc.metrics.Metric(dc.metrics.r2_score, n_tasks=2)
  multitask_scores = evaluator.compute_model_performance([metric1, metric2])
  assert isinstance(multitask_scores, dict)
  assert len(multitask_scores) == 2
  assert multitask_scores['mae_score'] > 0
  assert "r2_score" in multitask_scores


@pytest.mark.torch
def test_model_evaluate_dc_multi_metric():
  """Test an evaluator on a dataset."""
  X = np.random.rand(10, 5)
  y = np.random.rand(10, 1)
  dataset = dc.data.NumpyDataset(X, y)
  model = dc.models.MultitaskRegressor(1, 5)
  metric1 = dc.metrics.Metric(dc.metrics.mae_score)
  metric2 = dc.metrics.Metric(dc.metrics.r2_score)
  multitask_scores = model.evaluate(dataset, [metric1, metric2])
  assert isinstance(multitask_scores, dict)
  assert len(multitask_scores) == 2
  assert multitask_scores['mae_score'] > 0
  assert "r2_score" in multitask_scores


@pytest.mark.torch
def test_generator_evaluator_dc_metric_multitask_single_point():
  """Test generator evaluator on a generator."""
  X = np.random.rand(10, 5)
  y = np.random.rand(10, 1)
  dataset = dc.data.NumpyDataset(X, y)
  model = dc.models.MultitaskRegressor(1, 5)
  generator = model.default_generator(dataset, pad_batches=False)
  evaluator = GeneratorEvaluator(model, generator, [])
  metric = dc.metrics.Metric(dc.metrics.mae_score)
  multitask_scores = evaluator.compute_model_performance(metric)
  assert isinstance(multitask_scores, dict)
  assert len(multitask_scores) == 1
  assert multitask_scores['mae_score'] > 0
  assert len(multitask_scores) == 1


@pytest.mark.torch
def test_evaluator_sklearn_metric():
  """Test an evaluator on a dataset."""
  X = np.random.rand(10, 5)
  y = np.random.rand(10, 1)
  dataset = dc.data.NumpyDataset(X, y)
  model = dc.models.MultitaskRegressor(1, 5)
  evaluator = Evaluator(model, dataset, [])
  multitask_scores = evaluator.compute_model_performance(
      dc.metrics.mean_absolute_error)
  assert isinstance(multitask_scores, dict)
  assert len(multitask_scores) == 1
  # Note that since no name as provided, metrics are index by order
  # given.
  assert multitask_scores['metric-1'] > 0


@pytest.mark.torch
def test_generator_evaluator_dc_metric_multitask():
  """Test generator evaluator on a generator."""
  X = np.random.rand(10, 5)
  y = np.random.rand(10, 1)
  dataset = dc.data.NumpyDataset(X, y)
  model = dc.models.MultitaskRegressor(1, 5)
  generator = model.default_generator(dataset, pad_batches=False)
  evaluator = GeneratorEvaluator(model, generator, [])
  metric = dc.metrics.Metric(dc.metrics.mae_score)
  multitask_scores = evaluator.compute_model_performance(metric)
  assert isinstance(multitask_scores, dict)
  assert len(multitask_scores) == 1
  assert multitask_scores['mae_score'] > 0


@pytest.mark.torch
def test_model_evaluate_sklearn_metric():
  """Test a model evaluate on a dataset."""
  X = np.random.rand(10, 5)
  y = np.random.rand(10, 1)
  dataset = dc.data.NumpyDataset(X, y)
  model = dc.models.MultitaskRegressor(1, 5)
  multitask_scores = model.evaluate(dataset, dc.metrics.mean_absolute_error)
  assert isinstance(multitask_scores, dict)
  assert len(multitask_scores) == 1
  # Note that since no name as provided, metrics are index by order
  # given.
  assert multitask_scores['metric-1'] > 0


@pytest.mark.torch
def test_evaluator_sklearn_multi_metric():
  """Test an evaluator on a dataset."""
  X = np.random.rand(10, 5)
  y = np.random.rand(10, 1)
  dataset = dc.data.NumpyDataset(X, y)
  model = dc.models.MultitaskRegressor(1, 5)
  evaluator = Evaluator(model, dataset, [])
  multitask_scores = evaluator.compute_model_performance(
      [dc.metrics.mean_absolute_error, dc.metrics.r2_score])
  assert isinstance(multitask_scores, dict)
  assert len(multitask_scores.keys()) == 2
  # Note that since no name as provided, metrics are index by order
  # given.
  assert multitask_scores['metric-1'] > 0
  assert "metric-2" in multitask_scores


@pytest.mark.torch
def test_model_evaluate_sklearn_multi_metric():
  """Test an evaluator on a dataset."""
  X = np.random.rand(10, 5)
  y = np.random.rand(10, 1)
  dataset = dc.data.NumpyDataset(X, y)
  model = dc.models.MultitaskRegressor(1, 5)
  multitask_scores = model.evaluate(
      dataset, [dc.metrics.mean_absolute_error, dc.metrics.r2_score])
  assert isinstance(multitask_scores, dict)
  assert len(multitask_scores.keys()) == 2
  # Note that since no name as provided, metrics are index by order
  # given.
  assert multitask_scores['metric-1'] > 0
  assert "metric-2" in multitask_scores


@pytest.mark.tensorflow
def test_gc_binary_classification():
  """Test multiclass classification evaluation."""
  smiles = ["C", "CC"]
  featurizer = dc.feat.ConvMolFeaturizer()
  X = featurizer.featurize(smiles)
  y = np.random.randint(2, size=(len(smiles),))
  dataset = dc.data.NumpyDataset(X, y)
  model = dc.models.GraphConvModel(1, mode="classification")
  # TODO: Fix this case with correct thresholding
  evaluator = Evaluator(model, dataset, [])
  multitask_scores = evaluator.compute_model_performance(
      dc.metrics.accuracy_score, n_classes=2)
  assert len(multitask_scores) == 1
  assert multitask_scores["metric-1"] >= 0


@pytest.mark.tensorflow
def test_gc_binary_kappa_classification():
  """Test multiclass classification evaluation."""
  np.random.seed(1234)
  smiles = ["C", "CC", "CO", "CCC", "CCCC"]
  featurizer = dc.feat.ConvMolFeaturizer()
  X = featurizer.featurize(smiles)
  y = np.random.randint(2, size=(len(smiles),))
  dataset = dc.data.NumpyDataset(X, y)
  model = dc.models.GraphConvModel(1, mode="classification")
  # TODO: Fix this case with correct thresholding
  evaluator = Evaluator(model, dataset, [])
  multitask_scores = evaluator.compute_model_performance(
      dc.metrics.kappa_score, n_classes=2)
  assert len(multitask_scores) == 1
  assert multitask_scores["metric-1"] <= 1
  assert multitask_scores["metric-1"] >= -1


@pytest.mark.tensorflow
def test_gc_multiclass_classification():
  """Test multiclass classification evaluation."""
  np.random.seed(1234)
  smiles = ["C", "CC"]
  featurizer = dc.feat.ConvMolFeaturizer()
  X = featurizer.featurize(smiles)
  y = np.random.randint(5, size=(len(smiles),))
  dataset = dc.data.NumpyDataset(X, y)
  model = dc.models.GraphConvModel(1, mode="classification", n_classes=5)
  evaluator = Evaluator(model, dataset, [])
  multitask_scores = evaluator.compute_model_performance(
      dc.metrics.accuracy_score, n_classes=5)
  assert len(multitask_scores) == 1
  assert multitask_scores["metric-1"] >= 0
