#!/usr/bin/python
#
# Copyright 2015 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for metrics."""


import numpy as np
from tensorflow.python.platform import googletest
from deepchem.metrics import kappa_score 
from deepchem.metrics import Metric
from deepchem import metrics


class MetricsTest(googletest.TestCase):

  def test_kappa_score(self):
    y_true = [1, 0, 1, 0]
    y_pred = [0.8, 0.2, 0.3, 0.4]  # [1, 0, 0, 0] with 0.5 threshold
    kappa = kappa_score(y_true, np.greater(y_pred, 0.5))
    observed_agreement = 3.0 / 4.0
    expected_agreement = ((2 * 1) + (2 * 3)) / 4.0 ** 2
    expected_kappa = np.true_divide(observed_agreement - expected_agreement,
                                    1.0 - expected_agreement)
    self.assertAlmostEquals(kappa, expected_kappa)

  def test_r2_score(self):
    """Test that R^2 metric passes basic sanity tests"""
    verbosity = "high"
    np.random.seed(123)
    n_samples = 10
    y_true = np.random.rand(n_samples,)
    y_pred = np.random.rand(n_samples,)
    regression_metric = Metric(metrics.r2_score, verbosity=verbosity)
    assert np.isclose(metrics.r2_score(y_true, y_pred),
                      regression_metric.compute_metric(y_true, y_pred))
  

if __name__ == '__main__':
  googletest.main()
