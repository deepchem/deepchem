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
"""Wrapper of key-value pairs, which can be de/serialized from/to disk.
"""
import re

from google.protobuf import text_format
from tensorflow.python.platform import gfile

from deepchem.models.tensorflow_models import model_config_pb2


class ModelConfig(object):
  """Wrapper of key-value pairs which can be de/serialized from/to disk.

  A given key-value pair cannot be removed once added.
  This wrapper is mostly meant to read
  a config from disk or a python dict once, and subsequently the
  values are read through the object's attributes.

  De/Serialization is done through a protocol buffer with a text format,
  so files on disk are human readable and editable. See the unittest
  for an example of the protocol buffer text format.
  """

  _supported_types = [bool, int, float, str, unicode, list]
  _supported_overwrite_modes = ['forbidden', 'required', 'allowed']

  def __init__(self, defaults=None):
    """Creates a config object.

    Args:
      defaults: An optional dictionary with string keys and
          possibly heterogenously typed values;
          see class attribute _supported_types for supported types.
          The newly constructed object will gain attributes matching
          the dict's keys and values.
    """
    self._config_dict = {}
    if defaults:
      for key, value in defaults.iteritems():
        self.AddParam(key, value, overwrite='forbidden')

  def _ValidateParam(self, key, value, overwrite):
    """Checks param has a valid type, name, and enforces duplicate key handling.

    Args:
      key: str or unicode. Must be an allowable python attribute name,
          (specifically, must match r'^[a-zA-Z][a-zA-Z_0-9]+$')
      value: bool, int, float, str, unicode or homogeneous list thereof.
          The value to be stored.
      overwrite: String, how to handle duplicate keys.
        'forbidden': raise ValueError if key is already present.
        'required': raise ValueError if key is *not* already present.
        'allowed': key will be added or updated silently.

    Raises:
      ValueError: if parameters are not valid types,
          or if the key is not an allowable python attribute name,
          or if duplicate key validation failed.
    """
    if overwrite not in self._supported_overwrite_modes:
      raise ValueError(
          'overwrite mode "{}" not allowed, must be one of {}'.format(
              overwrite, ','.join(self._supported_overwrite_modes)))

    if type(key) not in [str, unicode]:
      raise ValueError('Key must but a string, but is: {}'.format(type(key)))

    if re.match(r'^[a-zA-Z][a-zA-Z_0-9]+$', key) is None:
      raise ValueError('Key is a bad attribute name: {}'.format(key))

    if key in self._config_dict:
      if overwrite == 'forbidden':
        raise ValueError('Not allowed to specify same key twice: {}'.format(
            key))
      if (not isinstance(value, type(self._config_dict[key])) and
          {str, unicode} != {type(value), type(self._config_dict[key])}):
        raise ValueError(
            'Not allowed to change value type ({} -> {}) for a key: {}'.format(
                type(self._config_dict[key]), type(value), key))
    else:
      if overwrite == 'required':
        raise ValueError('Must specify default for {}'.format(key))

    if type(value) not in self._supported_types:
      raise ValueError(
          'Only {} values allowed: {}'.format(
              ','.join([str(t) for t in self._supported_types]),
              type(value)))

    if type(value) is list:
      if not value:
        raise ValueError('Only non-empty lists supported: {}'.format(key))
      type_set = {type(v) for v in value}
      if len(type_set) > 1:
        raise ValueError('Only homogenous lists supported, found: {}={}'.format(
            key, ','.join(str(t) for t in type_set)))

  def AddParam(self, key, value, overwrite):
    """Adds one key-value pair to the dict being stored.

    Args:
      key: str or unicode. Must be an allowable python attribute name,
          (specifically, must match r'^[a-zA-Z][a-zA-Z_0-9]+$')
      value: bool, int, float, str, unicode or homogeneous list thereof.
          The value to be stored.
      overwrite: String, how to handle duplicate keys.
        See _ValidateParam for allowed values and descriptions.

    Raises:
      ValueError: see _ValidateParam for raising conditions.
    """
    self._ValidateParam(key, value, overwrite)
    self._config_dict[key] = value
    setattr(self, key, value)

  def GetOptionalParam(self, key, default_value):
    """Returns the param value or the default_value if not present.

    Typically you should directly read the object attribute for the
    key, but if the key is optionally present this method can be convenient.

    Args:
      key: String of the parameter name.
      default_value: Value to return if key is not present in this config.
          May be int, float or string.

    Returns:
      Value of the parameter named by key or default_value if key isn't present.
    """
    return getattr(self, key, default_value)

  def WriteToFile(self, filename):
    """Writes this ModelConfig object to disk.

    Args:
      filename: Path to write config to on disk.

    Raises:
      IOError: in case of error while writing.
      ValueError: in case of unsupported key or value type.
    """
    config_proto = model_config_pb2.ModelConfig()
    for key, value in sorted(self._config_dict.iteritems()):
      proto_param = config_proto.parameter.add()
      proto_param.name = key
      if type(value) is int:
        proto_param.int_value = value
      elif type(value) is float:
        proto_param.float_value = value
      elif type(value) in [str, unicode]:
        proto_param.string_value = value
      elif type(value) is bool:
        proto_param.bool_value = value
      elif type(value) is list:
        list_type = type(value[0])
        if list_type is int:
          proto_param.int_list.extend(value)
        elif list_type is float:
          proto_param.float_list.extend(value)
        elif list_type in [str, unicode]:
          proto_param.string_list.extend(value)
        elif list_type is bool:
          proto_param.bool_list.extend(value)
        else:
          raise ValueError('Unsupported list type: {}'.format(list_type))
      else:
        raise ValueError('Unsupported value type: {}'.format(type(value)))

    with open(filename, mode='w') as config_file:
      config_file.write(text_format.MessageToString(config_proto))

  def ReadFromFile(self, filename, overwrite='required'):
    """Reads into this ModelConfig object from disk.

    Args:
      filename: Path to serialized config file.
      overwrite: String, how to handle duplicate keys.
          See _ValidateParam for allowed values and descriptions.

    Raises:
      IOError: in case of error while reading.
      ValueError: if no value is set in a parameter.
    """
    config_proto = model_config_pb2.ModelConfig()
    with open(filename) as config_file:
      text_format.Merge(config_file.read(), config_proto)

    for p in config_proto.parameter:
      value_name = p.WhichOneof('value')
      if value_name:
        value = getattr(p, value_name)
      elif p.int_list:
        value = list(p.int_list)
      elif p.float_list:
        value = list(p.float_list)
      elif p.string_list:
        value = list(p.string_list)
      elif p.bool_list:
        value = list(p.bool_list)
      else:
        raise ValueError('No value set for key: {}'.format(p.name))
      self.AddParam(p.name, value, overwrite)
