"""
Tests for swissprot loader.
"""

import os
import deepchem as dc

from deepchem.molnet import load_swissprot


def test_swissprot_loader():
  current_dir = os.path.dirname(os.path.abspath(__file__))
  codes = [
      'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R',
      'S', 'T', 'V', 'W', 'Y'
  ]
  tasks, datasets, transformers = load_swissprot(
      splitter='random',
      data_dir=current_dir,
      save_dir=current_dir,
      reload=False,
      featurizer=dc.feat.OneHotFeaturizer(codes, max_length=1000))
  train, val, test = datasets
  assert len(tasks) == 1
  assert train.X.shape == (45796, 100, 21)


test_swissprot_loader()
