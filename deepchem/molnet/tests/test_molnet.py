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
    out_path = self.current_dir
    dc.molnet.run_benchmark(datasets, model, split=split, out_path=out_path)
    with open(os.path.join(out_path, 'results.csv'), 'r') as f:
      reader = csv.reader(f)
      for lastrow in reader:
        pass
      assert lastrow[-4] == model
      assert lastrow[-5] == 'valid'
      assert lastrow[-3] > 0.75

  def test_qm7_multitask(self):
    """Tests molnet benchmarking on qm7 with multitask network."""
    datasets = ['qm7']
    model = 'tf_regression'
    split = 'random'
    out_path = self.current_dir
    dc.molnet.run_benchmark(datasets, model, split=split, out_path=out_path)
    with open(os.path.join(out_path, 'results.csv'), 'r') as f:
      reader = csv.reader(f)
      for lastrow in reader:
        pass
      assert lastrow[-4] == model + '_ft'
      assert lastrow[-5] == 'valid'
      assert lastrow[-3] > 0.95

  def test_tox21_multitask(self):
    """Tests molnet benchmarking on tox21 with multitask network."""
    datasets = ['tox21']
    model = 'tf'
    split = 'random'
    out_path = self.current_dir
    dc.molnet.run_benchmark(datasets, model, split=split, out_path=out_path)
    with open(os.path.join(out_path, 'results.csv'), 'r') as f:
      reader = csv.reader(f)
      for lastrow in reader:
        pass
      assert lastrow[-4] == model
      assert lastrow[-5] == 'valid'
      assert lastrow[-3] > 0.75
