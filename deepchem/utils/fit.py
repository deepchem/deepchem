"""
Fit model. To be incorporated into Model class.
"""

import os
import sys
import numpy as np
import deepchem.models.deep
from deepchem.utils.dataset import ShardedDataset
from deepchem.models import Model
from deepchem.utils.dataset import load_from_disk
from deepchem.utils.preprocess import get_task_type

def fit_model(model_name, model_params, model_dir, data_dir):
  """Builds model from featurized data."""
  task_type = get_task_type(model_name)
  train = ShardedDataset(os.path.join(data_dir, "train"))

  task_types = {task: task_type for task in train.get_task_names()}
  model_params["data_shape"] = train.get_data_shape()

  model = Model.model_builder(model_name, task_types, model_params)
  model.fit(train)
  model.save(model_dir)
