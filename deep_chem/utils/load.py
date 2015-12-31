"""
Utility functions to load datasets.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import numpy as np
from deep_chem.utils.preprocess import transform_outputs
from deep_chem.utils.preprocess import transform_inputs

__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2015, Stanford University"
__license__ = "LGPL"

def ensure_balanced(y, W):
  """Helper function that ensures postives and negatives are balanced."""
  n_samples, n_targets = np.shape(y)
  for target_ind in range(n_targets):
    pos_weight, neg_weight = 0, 0
    for sample_ind in range(n_samples):
      if y[sample_ind, target_ind] == 0:
        neg_weight += W[sample_ind, target_ind]
      elif y[sample_ind, target_ind] == 1:
        pos_weight += W[sample_ind, target_ind]
    assert np.isclose(pos_weight, neg_weight)

#TODO(rbharath/enf): THIS IS SUPER BROKEN and probably needs complete rewrite.
def transform_data(data, input_transforms, output_transforms):
  """Transform data labels as specified

  Parameters
  ----------
  paths: list
    List of paths to Google vs datasets.
  output_transforms: dict
    dict mapping target names to list of label transforms. Each list element
    must be None, "log", "normalize", or "log-normalize". The transformations
    are performed in the order specified. An empty list corresponds to no
    transformations. Only for regression outputs.
  """
  trans_dict = {}
  X = transform_inputs(data["features"], input_transforms)
  trans_dict["mol_ids"], trans_dict["features"] = data["mol_ids"], X
  trans_dict["sorted_tasks"] = data["sorted_tasks"]
  for task in data["sorted_tasks"]:
    y, W = data[task]
    y = transform_outputs(y, W, output_transforms)
    trans_dict[task] = (y, W)
  assert trans_dict.keys() == data.keys()
  return trans_dict
