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
"""Evaluate a model from the ICML-2015 paper.

This script requires a trained model with its associated config and checkpoint.
If you don't have a trained model, run icml_train.py first.

"""
# pylint: disable=line-too-long
# pylint: enable=line-too-long


from nowhere.research.biology.collaborations.pande.py import utils

from tensorflow.python.platform import app
from tensorflow.python.platform import flags
from tensorflow.python.platform import gfile

from biology import model_config
from biology.icml import icml_models

flags.DEFINE_string('config', None, 'Serialized ModelConfig proto.')
flags.DEFINE_string('checkpoint', None,
                    'Model checkpoint file. File can contain either an '
                    'absolute checkpoint (e.g. model.ckpt-{step}) or a '
                    'serialized CheckpointState proto.')
flags.DEFINE_string('input_pattern', None, 'Input file pattern; '
                    'It should include %d for fold index substitution.')
flags.DEFINE_string('master', 'local', 'BNS name of the TensorFlow master.')
flags.DEFINE_string('logdir', None, 'Directory for output files.')
flags.DEFINE_integer('num_folds', 5, 'Number of cross-validation folds.')
flags.DEFINE_integer('fold', None, 'Fold index for this model.')
flags.DEFINE_enum('model_type', 'single', ['single', 'deep', 'deepaux', 'py',
                                           'pydrop1', 'pydrop2'],
                  'Which model from the ICML paper should be trained/evaluated')
FLAGS = flags.FLAGS


def main(unused_argv=None):
  config = model_config.ModelConfig()
  config.ReadFromFile(FLAGS.config, overwrite='allowed')
  gfile.MakeDirs(FLAGS.logdir)
  model = icml_models.CONSTRUCTORS[FLAGS.model_type](config,
                                                     train=False,
                                                     logdir=FLAGS.logdir,
                                                     master=FLAGS.master)

  if FLAGS.num_folds is not None and FLAGS.fold is not None:
    folds = utils.kfold_pattern(FLAGS.input_pattern, FLAGS.num_folds,
                                FLAGS.fold)
    _, test_pattern = folds.next()
    test_pattern = ','.join(test_pattern)
  else:
    test_pattern = FLAGS.input_pattern

  with model.graph.as_default():
    model.Eval(model.ReadInput(test_pattern), FLAGS.checkpoint)


if __name__ == '__main__':
  flags.MarkFlagAsRequired('config')
  flags.MarkFlagAsRequired('checkpoint')
  flags.MarkFlagAsRequired('input_pattern')
  flags.MarkFlagAsRequired('logdir')
  app.run()
