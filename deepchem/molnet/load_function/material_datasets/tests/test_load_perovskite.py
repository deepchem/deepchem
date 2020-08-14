"""
Tests for perovskite loader.
"""

import os
import numpy as np
from deepchem.molnet import load_perovskite


def test_perovskite_loader():
  current_dir = os.path.dirname(os.path.abspath(__file__))

  tasks, datasets, transformers = load_perovskite(
      reload=False,
      data_dir=current_dir,
      featurizer_kwargs={'max_atoms': 5},
      splitter_kwargs={
          'seed': 42,
          'frac_train': 0.6,
          'frac_valid': 0.2,
          'frac_test': 0.2
      })

  assert tasks[0] == 'formation_energy'
  assert datasets[0].X.shape == (3, 5)
  assert datasets[1].X.shape == (1, 5)
  assert datasets[2].X.shape == (1, 5)
  assert np.allclose(
      datasets[0].X[0],
      [0.02444208, -0.4804022, -0.51238194, -0.20286038, 0.53483076],
      atol=0.01)

  if os.path.exists(os.path.join(current_dir, 'perovskite.json')):
    os.remove(os.path.join(current_dir, 'perovskite.json'))
