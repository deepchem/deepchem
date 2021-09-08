import pytest
import tempfile
import numpy as np
import deepchem as dc
from deepchem.feat import MATFeaturizer
from deepchem.models.torch_models import MATModel

try:
  import torch
  has_torch = True
except:
  has_torch = False


@pytest.mark.torch
def test_mat_regression():
  # load datasets
  task, df, trans = dc.molnet.load_freesolv()
  train, valid, test = df

  # initialize model
  model = model = dc.models.torch_models.MATModel(
      batch_size=100, learning_rate=0.01)
  # overfit test
  model.fit(valid, nb_epoch=400)
  metric = dc.metrics.Metric(dc.metrics.mean_absolute_error, mode="regression")
  scores = model.evaluate(valid, [metric], trans)
  assert scores['mean_absolute_error'] < 1.0


@pytest.mark.torch
def test_mat_reload():
  model_dir = tempfile.mkdtemp()
  _, df, trans = dc.molnet.load_freesolv()
  _, valid, _ = df
  model = MATModel(batch_size=100, learning_rate=0.1, model_dir=model_dir)
  model.fit(valid, nb_epoch=400)
  metric = dc.metrics.Metric(dc.metrics.mean_absolute_error, mode="regression")
  scores = model.evaluate(valid, [metric], trans)
  assert scores['mean_absolute_error'] < 1.0
  reloaded_model = MATModel(
      batch_size=100, learning_rate=0.1, model_dir=model_dir)
  reloaded_model.restore()
  pred_mols = ["CCCC", "CCCCCO", "CCCCC"]
  X_pred = MATFeaturizer()(pred_mols)
  random_dataset = dc.data.NumpyDataset(X_pred)
  original_pred = model.predict(random_dataset)
  reload_pred = reloaded_model.predict(random_dataset)
  assert np.all(original_pred == reload_pred)
