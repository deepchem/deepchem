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
"""Train a model from the ICML-2015 paper.
"""
# pylint: disable=line-too-long
# pylint: enable=line-too-long

import os


from tensorflow.python.platform import app
from tensorflow.python.platform import flags
from tensorflow.python.platform import gfile

from biology import model_config
from biology.icml import icml_models

flags.DEFINE_string('config', None, 'Serialized ModelConfig proto.')
flags.DEFINE_string('master', '', 'BNS name of the TensorFlow master.')
flags.DEFINE_string('logdir', None, 'Directory for output files.')
flags.DEFINE_integer('replica_id', 0, 'Task ID of this replica.')
flags.DEFINE_integer('ps_tasks', 0, 'Number of parameter server tasks.')
flags.DEFINE_integer('num_folds', 5, 'Number of cross-validation folds.')
flags.DEFINE_integer('fold', None, 'Fold index for this model.')

FLAGS = flags.FLAGS

def kfold_pattern(input_pattern, num_folds, fold=None):
  """Generator for train/test filename splits.

  The pattern is not expanded except for the %d being replaced by the fold
  index.

  Args:
    input_pattern: Input filename pattern. Should contain %d for fold index.
    num_folds: Number of folds.
    fold: If not None, the generator only yields the train/test split for the
      given fold.

  Yields:
    train_filenames: A list of file patterns in training set.
    test_filenames: A list of file patterns in test set.
  """
  # get filenames associated with each fold
  fold_filepatterns = [input_pattern % i for i in range(num_folds)]

  # create train/test splits
  for i in range(num_folds):
    if fold is not None and i != fold:
      continue
    train = fold_filepatterns[:i] + fold_filepatterns[i+1:]
    test = [fold_filepatterns[i]]
    if any([f in test for f in train]):
      logging.fatal('Train/test split is not complete.')
    if set(train + test) != set(fold_filepatterns):
      logging.fatal('Not all input files are accounted for.')
    yield train, test


def main(unused_argv=None):
  Run()


if __name__ == '__main__':
  flags.MarkFlagAsRequired('config')
  flags.MarkFlagAsRequired('logdir')
  flags.MarkFlagAsRequired('fold')
  app.run()
