"""
Tests for getting featurizer, transformer, and splitter classes.
"""
import csv
import tempfile
import unittest

import numpy as np
import os
import pytest

import deepchem as dc
from deepchem.feat.base_classes import Featurizer
from deepchem.trans.transformers import Transformer
from deepchem.splits.splitters import Splitter
from deepchem.molnet.defaults import get_defaults


class TestDefaults(unittest.TestCase):
  """
  Tests for getting featurizer, transformer, and splitter classes.
  """

  def test_defaults(self):
    """Test getting defaults for MolNet loaders."""
    feats = get_defaults("feat")
    trans = get_defaults("trans")
    splits = get_defaults("splits")

    fkey = next(iter(feats))
    assert type(fkey) == str
    assert issubclass(feats[fkey], Featurizer)

    tkey = next(iter(trans))
    assert type(tkey) == str
    assert issubclass(trans[tkey], Transformer)

    skey = next(iter(splits))
    assert type(skey) == str
    assert issubclass(splits[skey], Splitter)
