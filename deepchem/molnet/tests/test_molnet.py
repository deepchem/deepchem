"""
Tests for molnet function 
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import unittest
import numpy as np
import pandas as pd
import deepchem as dc
import tempfile
import csv


class TestMolnet(unittest.TestCase):
  """
  Test basic function of molnet
  """

  def setUp(self):
    super(TestMolnet, self).setUp()
    self.current_dir = os.path.dirname(os.path.abspath(__file__))

  def test_delaney_graphconvreg(self):
    """Tests molnet benchmarking on delaney with graphconvreg."""
    datasets = ['delaney']
    model = 'graphconvreg'
    split = 'random'
    out_path = tempfile.mkdtemp()
    dc.molnet.run_benchmark(
        datasets, str(model), split=split, out_path=out_path)
    with open(os.path.join(out_path, 'results.csv'), 'r') as f:
      reader = csv.reader(f)
      for lastrow in reader:
        pass
      assert lastrow[-4] == model
      assert lastrow[-5] == 'valid'
      assert float(lastrow[-3]) > 0.75
    os.remove(os.path.join(out_path, 'results.csv'))

  def test_qm7_multitask(self):
    """Tests molnet benchmarking on qm7 with multitask network."""
    datasets = ['qm7']
    model = 'tf_regression_ft'
    split = 'random'
    out_path = tempfile.mkdtemp()
    dc.molnet.run_benchmark(
        datasets, str(model), split=split, out_path=out_path)
    with open(os.path.join(out_path, 'results.csv'), 'r') as f:
      reader = csv.reader(f)
      for lastrow in reader:
        pass
      assert lastrow[-4] == model
      assert lastrow[-5] == 'valid'
      assert float(lastrow[-3]) > 0.95
    os.remove(os.path.join(out_path, 'results.csv'))

  def test_clintox_multitask(self):
    """Tests molnet benchmarking on clintox with multitask network."""
    datasets = ['clintox']
    model = 'tf'
    split = 'random'
    out_path = tempfile.mkdtemp()
    dc.molnet.run_benchmark(
        datasets, str(model), split=split, out_path=out_path, test=True)
    with open(os.path.join(out_path, 'results.csv'), 'r') as f:
      reader = csv.reader(f)
      for lastrow in reader:
        pass
      assert lastrow[-4] == model
      assert lastrow[-5] == 'test'
      assert float(lastrow[-3]) > 0.7
    os.remove(os.path.join(out_path, 'results.csv'))
