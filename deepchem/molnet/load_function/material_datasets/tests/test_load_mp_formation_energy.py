"""
Tests for materials project formation energy loader.
"""

import os
import numpy as np
from deepchem.molnet import load_mp_formation_energy


def test_mp_formation_energy_loader():
  current_dir = os.path.dirname(os.path.abspath(__file__))

  tasks, datasets, transformers = load_mp_formation_energy(
      reload=False,
      data_dir=current_dir,
      featurizer_kwargs={'max_atoms': 2},
      splitter_kwargs={
          'seed': 42,
          'frac_train': 0.6,
          'frac_valid': 0.2,
          'frac_test': 0.2
      })

  assert tasks[0] == 'formation_energy'
  assert datasets[0].X.shape == (3, 2)
  assert datasets[1].X.shape == (1, 2)
  assert datasets[2].X.shape == (1, 2)
  assert np.allclose(datasets[0].X[0], [-0.80130437, -0.51393296], atol=0.01)

  if os.path.exists(os.path.join(current_dir, 'mp_formation_energy.json')):
    os.remove(os.path.join(current_dir, 'mp_formation_energy.json'))
