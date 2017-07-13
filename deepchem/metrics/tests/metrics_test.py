"""
Tests for metricsT.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "MIT"

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
    self.assertAlmostEquals(kappa, expected_kappa)

  def test_r2_score(self):
    """Test that R^2 metric passes basic sanity tests"""
    np.random.seed(123)
    n_samples = 10
    y_true = np.random.rand(
        n_samples,)
    y_pred = np.random.rand(
        n_samples,)
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


if __name__ == '__main__':
  googletest.main()
