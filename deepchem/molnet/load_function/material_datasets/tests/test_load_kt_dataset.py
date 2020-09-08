"""
Tests for bandgap loader.
"""

import os
import numpy as np
from deepchem.molnet import load_kt_dataset


def test_bandgap_loader():
  current_dir = os.path.dirname(os.path.abspath(__file__))
  tasks, datasets, transformers = load_kt_dataset(
      reload=False,
      data_dir=current_dir,
      splitter_kwargs={
          'seed': 42,
          'frac_train': 0.6,
          'frac_valid': 0.2,
          'frac_test': 0.2
      })

  assert tasks[0] == 'alpha'
  assert tasks[1] == 'beta'
  assert datasets[0].X.shape == (177, 1024)
  assert datasets[1].X.shape == (22, 1024)
  assert datasets[2].X.shape == (23, 1024)
  assert np.allclose(
      datasets[0].X[0][:2],
      np.array([0., 0.]),
      atol=0.01)

  if os.path.exists(os.path.join(current_dir, 'KTparameterDataset.csv')):
    os.remove(os.path.join(current_dir, 'KTparameterDataset.csv'))
