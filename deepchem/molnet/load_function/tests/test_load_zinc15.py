"""
Tests for zinc15 loader.
"""

import os
import numpy as np
from deepchem.molnet import load_zinc15


def test_zinc15_loader():
  current_dir = os.path.dirname(os.path.abspath(__file__))

  tasks, datasets, transformers = load_zinc15(
      reload=False,
      data_dir=current_dir,
      splitter_kwargs={
          'seed': 42,
          'frac_train': 0.6,
          'frac_valid': 0.2,
          'frac_test': 0.2
      })

  test_vec = np.array([
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, -1.224744871391589, 0.0, 0.0, 0.0, 0.0, 2.0, -0.5, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
  ])

  train, val, test = datasets
  assert tasks == ['mwt', 'logp', 'reactive']
  assert train.X.shape == (3, 100, 35)
  assert np.allclose(train.X[0][0], test_vec, atol=0.01)

  if os.path.exists(os.path.join(current_dir, 'zinc15_250K_2D.csv')):
    os.remove(os.path.join(current_dir, 'zinc15_250K_2D.csv'))
