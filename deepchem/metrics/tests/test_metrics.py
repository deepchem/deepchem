"""
Tests for metricsT.
"""
import numpy as np
import deepchem as dc
from tensorflow.python.platform import googletest
from deepchem import metrics


class MetricsTest(googletest.TestCase):

  def test_kappa_score(self):
    y_true = [1, 0, 1, 0]
    y_pred = [0.8, 0.2, 0.3, 0.4]  # [1, 0, 0, 0] with 0.5 threshold
    kappa = dc.metrics.kappa_score(y_true, np.greater(y_pred, 0.5))
    observed_agreement = 3.0 / 4.0
    expected_agreement = ((2 * 1) + (2 * 3)) / 4.0**2
    expected_kappa = np.true_divide(observed_agreement - expected_agreement,
                                    1.0 - expected_agreement)
    self.assertAlmostEqual(kappa, expected_kappa)

  def test_one_sample(self):
    """Test that the metrics won't raise error even in an extreme condition
    where there is only one sample with w > 0.
    """
    np.random.seed(123)
    n_samples = 2
    y_true = np.array([0, 0])
    y_pred = np.random.rand(n_samples, 2)
    w = np.array([0, 1])
    all_metrics = [
        dc.metrics.Metric(dc.metrics.recall_score),
        dc.metrics.Metric(dc.metrics.matthews_corrcoef),
        dc.metrics.Metric(dc.metrics.roc_auc_score)
    ]
    for metric in all_metrics:
      score = metric.compute_singletask_metric(y_true, y_pred, w)
      self.assertTrue(np.isnan(score) or score == 0)

  def test_r2_score(self):
    """Test that R^2 metric passes basic sanity tests"""
    np.random.seed(123)
    n_samples = 10
    y_true = np.random.rand(n_samples,)
    y_pred = np.random.rand(n_samples,)
    regression_metric = dc.metrics.Metric(dc.metrics.r2_score)
    assert np.isclose(
        dc.metrics.r2_score(y_true, y_pred),
        regression_metric.compute_metric(y_true, y_pred))

  def test_one_hot(self):
    y = np.array([0, 0, 1, 0, 1, 1, 0])
    y_hot = metrics.to_one_hot(y)
    expected = np.array([[1, 0], [1, 0], [0, 1], [1, 0], [0, 1], [0, 1], [1,
                                                                          0]])
    yp = metrics.from_one_hot(y_hot)
    assert np.array_equal(expected, y_hot)
    assert np.array_equal(y, yp)

  def test_bedroc_score(self):

    num_actives = 20
    num_total = 400

    y_true_actives = np.ones(num_actives)
    y_true_inactives = np.zeros(num_total - num_actives)
    y_true = np.concatenate([y_true_actives, y_true_inactives])

    # Best score case
    y_pred_best = dc.metrics.to_one_hot(
        np.concatenate([y_true_actives, y_true_inactives]))
    best_score = dc.metrics.bedroc_score(y_true, y_pred_best)
    self.assertAlmostEqual(best_score, 1.0)

    # Worst score case
    worst_pred_actives = np.zeros(num_actives)
    worst_pred_inactives = np.ones(num_total - num_actives)
    y_pred_worst = dc.metrics.to_one_hot(
        np.concatenate([worst_pred_actives, worst_pred_inactives]))
    worst_score = dc.metrics.bedroc_score(y_true, y_pred_worst)
    self.assertAlmostEqual(worst_score, 0.0, 4)


if __name__ == '__main__':
  googletest.main()
