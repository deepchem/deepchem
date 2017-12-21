"""Manages Placeholders for Graph convolution networks.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Han Altae-Tran and Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "MIT"

import warnings
import numpy as np
import tensorflow as tf
from deepchem.feat.mol_graphs import ConvMol


def merge_two_dicts(x, y):
  z = x.copy()
  z.update(y)
  return z


def merge_dicts(l):
  """Convenience function to merge list of dictionaries."""
  merged = {}
  for dict in l:
    merged = merge_two_dicts(merged, dict)
  return merged
