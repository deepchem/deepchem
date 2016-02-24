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
"""Tests for model."""


import numpy as np

from tensorflow.python.platform import googletest

from biology import model
from biology import model_config


class ClassifierTest(googletest.TestCase):

  def setUp(self):
    self.config = model_config.ModelConfig({
        'batch_size': 2,
        'num_classification_tasks': 1,
    })
    self.model = model.Classifier(self.config, train=True,
                                  logdir='/tmp/classifier_test')

  def testParseModelOutput(self):
    # standard 2-class output; some weights are zero
    output = np.asarray([[[0.1, 0.9]],
                         [[0.2, 0.8]]], dtype=float)
    labels = np.asarray([[[0, 1]],
                         [[1, 0]]], dtype=float)
    weights = np.asarray([[0],
                          [1]], dtype=float)
    expected_y_true = [[0]]
    expected_y_pred = [[0.8]]
    y_true, y_pred = self.model.ParseModelOutput(output, labels, weights)
    self.assertListEqual(y_true, expected_y_true)
    self.assertListEqual(y_pred, expected_y_pred)

if __name__ == '__main__':
  googletest.main()
