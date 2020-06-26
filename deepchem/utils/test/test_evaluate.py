"""Unit tests for evaluators."""
import deepchem as dc
import numpy as np
import unittest
from deepchem.utils.evaluate import Evaluator
from deepchem.utils.evaluate import GeneratorEvaluator

class TestEvaluator(unittest.TestCase):

  def setUp(self):
    """Perform common setup for tests."""
    X = np.random.rand(10, 5)
    y = np.random.rand(10, 1)
    self.dataset = dc.data.NumpyDataset(X, y)

  def test_evaluator_dc_metric(self):
    """Test an evaluator on a dataset."""
    model = dc.models.MultitaskRegressor(1, 5)
    transformers = []
    evaluator = Evaluator(model, self.dataset, transformers)
    metric = dc.metrics.Metric(dc.metrics.mae_score)
    multitask_scores = evaluator.compute_model_performance([metric])
    assert isinstance(multitask_scores, dict)
    assert len(multitask_scores) == 1
    assert multitask_scores['mae_score'] > 0

  def test_generator_evaluator_dc_metric_multitask(self):
    """Test generator evaluator on a generator."""
    model = dc.models.MultitaskRegressor(1, 5)
    generator = model.default_generator(
      self.dataset, pad_batches=False)
    transformers = []
    evaluator = GeneratorEvaluator(model, generator, transformers)
    metric = dc.metrics.Metric(dc.metrics.mae_score)
    multitask_scores = evaluator.compute_model_performance([metric])
    assert isinstance(multitask_scores, dict)
    assert len(multitask_scores) == 1
    assert multitask_scores['mae_score'] > 0

  def test_generator_evaluator_dc_metric_multitask_single_point(self):
    """Test generator evaluator on a generator."""
    model = dc.models.MultitaskRegressor(1, 5)
    generator = model.default_generator(
      self.dataset, pad_batches=False)
    transformers = []
    evaluator = GeneratorEvaluator(model, generator, transformers)
    metric = dc.metrics.Metric(dc.metrics.mae_score)
    multitask_scores = evaluator.compute_model_performance([metric])
    assert isinstance(multitask_scores, dict)
    assert len(multitask_scores) == 1
    print("multitask_scores")
    print(multitask_scores)
    assert multitask_scores['mae_score'] > 0

  def test_evaluator_dc_metric_singletask(self):
    """Test an evaluator on a dataset."""
    model = dc.models.MultitaskRegressor(1, 5)
    transformers = []
    evaluator = Evaluator(model, self.dataset, transformers)
    metric = dc.metrics.Metric(dc.metrics.mae_score)
    multitask_scores = evaluator.compute_model_performance([metric])
    assert isinstance(multitask_scores, dict)
    assert len(multitask_scores) == 1
    assert multitask_scores['mae_score'] > 0
