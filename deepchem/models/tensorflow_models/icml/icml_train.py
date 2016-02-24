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


def Run(input_data_types=None):
  """Trains the model with specified parameters.

  Args:
    input_data_types: List of legacy_types_pb2 constants or None.
  """
  config = model_config.ModelConfig({
      'input_pattern': '',  # Should have %d for fold index substitution.
      'num_classification_tasks': 259,
      'tasks_in_input': 259,  # Dimensionality of sstables
      'max_steps': 50000000,
      'summaries': False,
      'batch_size': 128,
      'learning_rate': 0.0003,
      'num_classes': 2,
      'optimizer': 'sgd',
      'penalty': 0.0,
      'num_features': 1024,
      'layer_sizes': [1200],
      'weight_init_stddevs': [0.01],
      'bias_init_consts': [0.5],
      'dropouts': [0.0],
  })
  config.ReadFromFile(FLAGS.config,
                      overwrite='required')

  if FLAGS.replica_id == 0:
    gfile.MakeDirs(FLAGS.logdir)
    config.WriteToFile(os.path.join(FLAGS.logdir, 'config.pbtxt'))

  model = icml_models.IcmlModel(config,
                                train=True,
                                logdir=FLAGS.logdir,
                                master=FLAGS.master)

  if FLAGS.num_folds is not None and FLAGS.fold is not None:
    folds = kfold_pattern(config.input_pattern, FLAGS.num_folds,
                          FLAGS.fold)
    train_pattern, _ = folds.next()
    train_pattern = ','.join(train_pattern)
  else:
    train_pattern = config.input_pattern

  with model.graph.as_default():
    model.Train(model.ReadInput(train_pattern,
                                input_data_types=input_data_types),
                max_steps=config.max_steps,
                summaries=config.summaries,
                replica_id=FLAGS.replica_id,
                ps_tasks=FLAGS.ps_tasks)


def main(unused_argv=None):
  Run()


if __name__ == '__main__':
  flags.MarkFlagAsRequired('config')
  flags.MarkFlagAsRequired('logdir')
  flags.MarkFlagAsRequired('fold')
  app.run()
