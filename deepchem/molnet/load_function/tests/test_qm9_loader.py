"""
Tests for qm9 loader.
"""

import os
import numpy as np
from deepchem.molnet import load_qm9


def test_qm9_loader():
  current_dir = os.path.dirname(os.path.abspath(__file__))
  tasks, datasets, transformers = load_qm9(
      reload=False,
      data_dir=current_dir,
      featurizer='ECFP',
      splitter_kwargs={
          'seed': 42,
          'frac_train': 0.6,
          'frac_valid': 0.2,
          'frac_test': 0.2
      })

  assert len(tasks) == 12
  assert tasks[0] == 'mu'
  assert datasets[0].X.shape == (8, 1024)
