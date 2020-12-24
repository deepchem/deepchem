import pytest
import numpy as np
import tensorflow as tf

import deepchem as dc
from deepchem.molnet import load_tox21


def test_tox21_tf_progressive():
  np.random.seed(123)
  tf.random.set_seed(124)

  n_features = 1024
  tox21_tasks, tox21_datasets, transformers = load_tox21()
  _, valid_dataset, _ = tox21_datasets

  metric = dc.metrics.Metric(dc.metrics.roc_auc_score, np.mean)

  def init(**kwargs):
    tf.random.set_seed(572)
    return dc.models.ProgressiveMultitaskClassifier(
        len(tox21_tasks),
        n_features,
        layer_sizes=[1000],
        dropouts=[.25],
        learning_rate=0.001,
        batch_size=50,
        **kwargs)

  ref_model = init()
  model = init(use_openvino=True)

  ref_scores = ref_model.evaluate(valid_dataset, [metric], transformers)
  scores = model.evaluate(valid_dataset, [metric], transformers)
  assert scores['mean-roc_auc_score'] == pytest.approx(
      ref_scores['mean-roc_auc_score'], 1e-5)
  assert model._openvino_model.is_available()
