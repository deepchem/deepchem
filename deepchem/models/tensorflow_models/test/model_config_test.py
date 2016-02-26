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


import os
import tempfile

from tensorflow.python.platform import flags
from tensorflow.python.platform import gfile
from tensorflow.python.platform import googletest

from deepchem.models.tensorflow_models import model_config

EXAMPLE_DICT = {
    'hello': 'world',
    'pi': 3.14159,
    'Forty_Two': 42,
    'great': True,
    'spells': ['alohamora', 'expelliarmus'],
    'scores': [9.8, 10.0],
    'sizes': [2000, 100],
    'waver': [True, False, True],
    }

EXAMPLE_DEFAULTS = {
    'hello': 'there',
    'pi': 3.14,
    'Forty_Two': 24,
    'great': False,
    'spells': ['abracadabra', 'cruciatus'],
    'scores': [1.8, 1.0],
    'sizes': [1200, 10],
    'waver': [False, True, False],
}

EXAMPLE_FILE_CONTENTS = """parameter {
  name: "Forty_Two"
  int_value: 42
}
parameter {
  name: "great"
  bool_value: true
}
parameter {
  name: "hello"
  string_value: "world"
}
parameter {
  name: "pi"
  float_value: 3.14159
}
parameter {
  name: "scores"
  float_list: 9.8
  float_list: 10.0
}
parameter {
  name: "sizes"
  int_list: 2000
  int_list: 100
}
parameter {
  name: "spells"
  string_list: "alohamora"
  string_list: "expelliarmus"
}
parameter {
  name: "waver"
  bool_list: true
  bool_list: false
  bool_list: true
}
"""


class ModelConfigTest(googletest.TestCase):

  def setUp(self):
    super(ModelConfigTest, self).setUp()
    self.root = tempfile.mkdtemp()

  def _assertMatchesExample(self, config):
    self.assertEqual(config.hello, 'world')
    self.assertEqual(config.pi, 3.14159)
    self.assertEqual(config.Forty_Two, 42)
    self.assertTrue(config.great)
    self.assertEqual(config.scores, [9.8, 10.0])
    self.assertEqual(config.sizes, [2000, 100])
    self.assertEqual(config.spells, ['alohamora', 'expelliarmus'])
    self.assertEqual(config.waver, [True, False, True])

  def testCreatesAttributes(self):
    config = model_config.ModelConfig(EXAMPLE_DICT)
    self._assertMatchesExample(config)

  def testGetOptionalParam(self):
    config = model_config.ModelConfig(EXAMPLE_DICT)
    self.assertEqual('world', config.GetOptionalParam('hello', 'everybody'))
    self.assertEqual('default', config.GetOptionalParam('otherkey', 'default'))

  def testOnlyValidAttributeNamesAllowed(self):
    config = model_config.ModelConfig()
    with self.assertRaises(ValueError):
      config.AddParam('spaces not allowed',
                      'blah',
                      overwrite='forbidden')

    with self.assertRaises(ValueError):
      config.AddParam('42_must_start_with_letter',
                      'blah',
                      overwrite='forbidden')

    with self.assertRaises(ValueError):
      config.AddParam('hyphens-not-allowed',
                      'blah',
                      overwrite='forbidden')

    with self.assertRaises(ValueError):
      config.AddParam('',
                      'empty string no good',
                      overwrite='forbidden')

  def testDuplicateKeysNotAllowed(self):
    config = model_config.ModelConfig(EXAMPLE_DICT)
    with self.assertRaises(ValueError):
      config.AddParam('hello',
                      'everybody',
                      overwrite='forbidden')

  def testRequireDefault(self):
    config = model_config.ModelConfig(EXAMPLE_DICT)
    config.AddParam('hello',
                    'everybody',
                    overwrite='required')
    with self.assertRaises(ValueError):
      config.AddParam('not',
                      'present',
                      overwrite='required')

  def testSilentOverwrite(self):
    config = model_config.ModelConfig(EXAMPLE_DICT)
    config.AddParam('not', 'present', overwrite='allowed')
    config.AddParam('not', 'anymore', overwrite='allowed')

  def testHeterogeneousList(self):
    config = model_config.ModelConfig()
    with self.assertRaises(ValueError):
      config.AddParam('different',
                      ['types for', 'different', 0xF, 0x0, 'lks'],
                      overwrite='forbidden')

  def testWritesFile(self):
    config = model_config.ModelConfig(EXAMPLE_DICT)
    filename = os.path.join(self.root, 'config.pbtxt')
    config.WriteToFile(filename)

    with open(filename) as pbtxt_file:
      self.assertEqual(EXAMPLE_FILE_CONTENTS, pbtxt_file.read())

  def testReadsFile_NoDuplicates(self):
    filename = os.path.join(self.root, 'config.pbtxt')
    with open(filename, 'w') as pbtxt_file:
      pbtxt_file.write(EXAMPLE_FILE_CONTENTS)

    config = model_config.ModelConfig()
    config.ReadFromFile(filename, overwrite='forbidden')
    self._assertMatchesExample(config)

  def testReadsFile_RequireDefaults(self):
    filename = os.path.join(self.root, 'config.pbtxt')
    with open(filename, 'w') as pbtxt_file:
      pbtxt_file.write(EXAMPLE_FILE_CONTENTS)

    self.assertEqual(set(EXAMPLE_DEFAULTS.keys()), set(EXAMPLE_DICT.keys()))
    config = model_config.ModelConfig(EXAMPLE_DEFAULTS)
    config.ReadFromFile(filename, overwrite='required')
    self._assertMatchesExample(config)


if __name__ == '__main__':
  googletest.main()
