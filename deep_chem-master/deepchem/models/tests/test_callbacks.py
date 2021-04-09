import unittest
import tempfile
import deepchem as dc
import numpy as np
import tensorflow as tf
try:
  from StringIO import StringIO
except ImportError:
  from io import StringIO


class TestCallbacks(unittest.TestCase):

  def test_validation(self):
    """Test ValidationCallback."""
    tasks, datasets, transformers = dc.molnet.load_clintox()
    train_dataset, valid_dataset, test_dataset = datasets
    n_features = 1024
    model = dc.models.MultitaskClassifier(len(tasks), n_features, dropouts=0.5)

    # Train the model while logging the validation ROC AUC.

    metric = dc.metrics.Metric(dc.metrics.roc_auc_score, np.mean)
    log = StringIO()
    save_dir = tempfile.mkdtemp()
    callback = dc.models.ValidationCallback(
        valid_dataset,
        30, [metric],
        log,
        save_dir=save_dir,
        save_on_minimum=False)
    model.fit(train_dataset, callbacks=callback)

    # Parse the log to pull out the AUC scores.

    log.seek(0)
    scores = []
    for line in log:
      score = float(line.split('=')[-1])
      scores.append(score)

    # The last reported score should match the current performance of the model.

    valid_score = model.evaluate(valid_dataset, [metric], transformers)
    self.assertAlmostEqual(
        valid_score['mean-roc_auc_score'], scores[-1], places=5)

    # Reload the save model and confirm that it matches the best logged score.

    model.restore(model_dir=save_dir)
    valid_score = model.evaluate(valid_dataset, [metric], transformers)
    self.assertAlmostEqual(
        valid_score['mean-roc_auc_score'], max(scores), places=5)
