"""Unit tests for evaluators."""
import deepchem as dc
import numpy as np
import unittest
import sklearn
from deepchem.utils.evaluate import Evaluator
from deepchem.utils.evaluate import GeneratorEvaluator

class TestEvaluator(unittest.TestCase):

  def setUp(self):
    """Perform common setup for tests."""
    X = np.random.rand(10, 5)
    y = np.random.rand(10, 1)
    self.dataset = dc.data.NumpyDataset(X, y)
    self.model = dc.models.MultitaskRegressor(1, 5)

  def test_evaluator_dc_metric(self):
    """Test an evaluator on a dataset."""
    evaluator = Evaluator(self.model, self.dataset, [])
    metric = dc.metrics.Metric(dc.metrics.mae_score)
    multitask_scores = evaluator.compute_model_performance(metric)
    assert isinstance(multitask_scores, dict)
    assert len(multitask_scores) == 1
    assert multitask_scores['mae_score'] > 0

  def test_evaluator_dc_multi_metric(self):
    """Test an evaluator on a dataset."""
    evaluator = Evaluator(self.model, self.dataset, [])
    metric1 = dc.metrics.Metric(dc.metrics.mae_score)
    metric2 = dc.metrics.Metric(dc.metrics.r2_score)
    multitask_scores = evaluator.compute_model_performance(
      [metric1, metric2])
    assert isinstance(multitask_scores, dict)
    assert len(multitask_scores) == 2
    assert multitask_scores['mae_score'] > 0
    assert "r2_score" in multitask_scores
    
    
  def test_evaluator_sklearn_metric(self):
    """Test an evaluator on a dataset."""
    evaluator = Evaluator(self.model, self.dataset, [])
    multitask_scores = evaluator.compute_model_performance(
      sklearn.metrics.mean_absolute_error)
    assert isinstance(multitask_scores, dict)
    assert len(multitask_scores) == 1
    # Note that since no name as provided, metrics are index by order
    # given.
    assert multitask_scores['metric-1'] > 0

  def test_evaluator_sklearn_multi_metric(self):
    """Test an evaluator on a dataset."""
    evaluator = Evaluator(self.model, self.dataset, [])
    multitask_scores = evaluator.compute_model_performance(
      [sklearn.metrics.mean_absolute_error,
       sklearn.metrics.r2_score])
    assert isinstance(multitask_scores, dict)
    assert len(multitask_scores.keys()) == 2
    # Note that since no name as provided, metrics are index by order
    # given.
    assert multitask_scores['metric-1'] > 0
    assert "metric-2" in multitask_scores

  def test_generator_evaluator_dc_metric_multitask(self):
    """Test generator evaluator on a generator."""
    generator = self.model.default_generator(
      self.dataset, pad_batches=False)
    evaluator = GeneratorEvaluator(self.model, generator, [])
    metric = dc.metrics.Metric(dc.metrics.mae_score)
    multitask_scores = evaluator.compute_model_performance(metric)
    assert isinstance(multitask_scores, dict)
    assert len(multitask_scores) == 1
    assert multitask_scores['mae_score'] > 0

  def test_generator_evaluator_dc_metric_multitask_single_point(self):
    """Test generator evaluator on a generator."""
    generator = self.model.default_generator(
      self.dataset, pad_batches=False)
    evaluator = GeneratorEvaluator(self.model, generator, [])
    metric = dc.metrics.Metric(dc.metrics.mae_score)
    multitask_scores = evaluator.compute_model_performance(metric)
    assert isinstance(multitask_scores, dict)
    assert len(multitask_scores) == 1
    print("multitask_scores")
    print(multitask_scores)
    assert multitask_scores['mae_score'] > 0

  def test_evaluator_dc_metric_singletask(self):
    """Test an evaluator on a dataset."""
    evaluator = Evaluator(self.model, self.dataset, [])
    metric = dc.metrics.Metric(dc.metrics.mae_score)
    multitask_scores = evaluator.compute_model_performance(metric)
    assert isinstance(multitask_scores, dict)
    assert len(multitask_scores) == 1
    assert multitask_scores['mae_score'] > 0

  def test_multiclass_classification_singletask(self):
    """Test multiclass classification evaluation."""
    X = np.random.rand(100, 5)
    y = np.random.randint(5, size=(100,))
    dataset = dc.data.NumpyDataset(X, y)
    model = dc.models.MultitaskClassifier(1, 5, n_classes=5)
    evaluator = Evaluator(model, dataset, [])
    multitask_scores = evaluator.compute_model_performance(
      sklearn.metrics.accuracy_score, n_classes=5)
    assert len(multitask_scores) == 1
    assert multitask_scores["metric-1"] >= 0

# TODO: Add a multtiask metrics example
# TODO: Add a multitask per-task metric example
# TODO: Add metrics for images here as a test

