"""
Tests for swissprot loader.
"""

import os
import numpy as np
from deepchem.molnet import load_swissprot
def test_swissprot_loader():
  current_dir = os.path.dirname(os.path.abspath(__file__))
  codes = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
         'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
  tasks, datasets = load_swissprot(
      reload=False,
      data_dir=current_dir,
      featurizer=dc.featurizer.OneHoteFeatureizer(codes),
      splitter_kwargs={
          'seed': 42,
          'frac_train': 0.6,
          'frac_valid': 0.2,
          'frac_test': 0.2
      })
  assert len(tasks) == 0
  assert datasets.X.shape == (564638, 100, 35)
