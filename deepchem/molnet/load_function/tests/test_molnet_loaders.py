"""
Tests for MolNet loader functions.
"""

import os
import tempfile
import shutil
import numpy as np
import deepchem as dc
from deepchem.molnet import load_bandgap, load_perovskite

# TODO: add unit tests for other dataset loaders that comply with
# MolNet loader contribution template


def test_material_dataset_loaders():
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

  assert tasks[0] == 'gap expt'
  assert datasets[0].X.shape == (3, 65)
  assert datasets[1].X.shape == (1, 65)
  assert datasets[2].X.shape == (1, 65)
  assert np.allclose(
      datasets[0].X[0][:5],
      np.array([0., 1.22273676, 1.22273676, 1.79647628, 0.82919516]),
      atol=0.01)

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

  assert tasks[0] == 'e_form'
  assert datasets[0].X.shape == (3, 1, 5)
  assert datasets[1].X.shape == (1, 1, 5)
  assert datasets[2].X.shape == (1, 1, 5)
  assert np.allclose(
      datasets[0].X[0][0],
      [0.02444208, -0.4804022, -0.51238194, -0.20286038, 0.53483076],
      atol=0.01)

  if os.path.exists(os.path.join(current_dir, 'expt_gap.json')):
    os.remove(os.path.join(current_dir, 'expt_gap.json'))

  if os.path.exists(os.path.join(current_dir, 'perovskite.json')):
    os.remove(os.path.join(current_dir, 'perovskite.json'))
