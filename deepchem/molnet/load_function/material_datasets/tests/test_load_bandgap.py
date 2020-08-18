"""
Tests for bandgap loader.
"""

import os
import numpy as np
from deepchem.molnet import load_bandgap


def test_bandgap_loader():
  current_dir = os.path.dirname(os.path.abspath(__file__))
  tasks, datasets, transformers = load_bandgap(
      reload=False,
      data_dir=current_dir,
      splitter_kwargs={
          'seed': 42,
          'frac_train': 0.6,
          'frac_valid': 0.2,
          'frac_test': 0.2
      })

  assert tasks[0] == 'experimental_bandgap'
  assert datasets[0].X.shape == (3, 65)
  assert datasets[1].X.shape == (1, 65)
  assert datasets[2].X.shape == (1, 65)
  assert np.allclose(
      datasets[0].X[0][:5],
      np.array([0., 1.22273676, 1.22273676, 1.79647628, 0.82919516]),
      atol=0.01)

  if os.path.exists(os.path.join(current_dir, 'expt_gap.json')):
    os.remove(os.path.join(current_dir, 'expt_gap.json'))
