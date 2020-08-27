"""
Tests for materials project metallicity loader.
"""

import os
import numpy as np
from deepchem.molnet import load_mp_metallicity


def test_mp_metallicity_loader():
  current_dir = os.path.dirname(os.path.abspath(__file__))

  tasks, datasets, transformers = load_mp_metallicity(
      reload=False,
      data_dir=current_dir,
      featurizer_kwargs={'max_atoms': 8},
      splitter_kwargs={
          'seed': 42,
          'frac_train': 0.6,
          'frac_valid': 0.2,
          'frac_test': 0.2
      })

  assert tasks[0] == 'is_metal'
  assert datasets[0].X.shape == (3, 8)
  assert datasets[1].X.shape == (1, 8)
  assert datasets[2].X.shape == (1, 8)
  assert np.allclose(
      datasets[0].X[0], [
          0.80428488, -0.70720997, 1.29101261, 0.61631094, 0.84184489,
          -0.28273997, -1.10252907, -1.23500371
      ],
      atol=0.01)

  if os.path.exists(os.path.join(current_dir, 'mp_is_metal.json')):
    os.remove(os.path.join(current_dir, 'mp_is_metal.json'))
