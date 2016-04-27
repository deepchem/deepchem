"""
Tests to make sure TF models can overfit on tiny datasets.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "LGPL"

import tempfile
import numpy as np
import unittest
import sklearn
from deepchem import metrics
from deepchem.datasets import Dataset
from deepchem.metrics import Metric
from deepchem.models.test import TestAPI
from deepchem.utils.evaluate import Evaluator
from deepchem.models.sklearn_models import SklearnModel
from sklearn.ensemble import RandomForestClassifier

class TestOverfitAPI(TestAPI):
  """
  Test that sklearn and keras models can overfit simple datasets.
  """

  def test_classification_overfit(self):
    """Test that data associated with a tasks stays associated with it."""
    tasks = ["task0"]
    task_types = {task: "classification" for task in tasks}
    n_samples = 10
    n_features = 3
    n_tasks = len(tasks)
    
    # Generate dummy dataset
    ids = np.arange(n_samples)
    #X = np.random.rand(n_samples, n_features)
    X = np.ones((n_samples, n_features))
    #y = np.random.randint(2, size=(n_samples, n_tasks))
    y = np.zeros((n_samples, n_tasks))
    w = np.ones((n_samples, n_tasks))
  
    dataset = Dataset.from_numpy(self.train_dir, tasks, X, y, w, ids)

    model_params = {
      "batch_size": None,
      "data_shape": dataset.get_data_shape()
    }
    np.set_printoptions(precision=5)

    verbosity = "high"
    classification_metric = Metric(metrics.accuracy_score, verbosity=verbosity)
    model = SklearnModel(tasks, task_types, model_params, self.model_dir,
                         mode="classification",
                         model_instance=RandomForestClassifier())

    cl = RandomForestClassifier()
    y, w = y.flatten(), w.flatten()
    cl.fit(X, y, w)

    y_pred = cl.predict(X)
    np.set_printoptions(precision=5)
    y, y_pred = y.flatten(), y_pred.flatten()
    np.testing.assert_array_almost_equal(y, y_pred)

    # Fit trained model
    model.fit(dataset)
    model.save()
    X_dataset, y_dataset, _, _ = dataset.to_numpy()
    np.testing.assert_array_almost_equal(X, X_dataset)
    np.testing.assert_array_almost_equal(y.flatten(), y_dataset.flatten())

    y_pred_model = model.predict(dataset, transformers=[])
    print("y_pred_model")
    print(y_pred_model)
    y_pred_proba_model = model.predict_proba(dataset, transformers=[])
    print("y_pred_proba_model")
    print(y_pred_proba_model)

    # Eval model on train
    transformers = []
    evaluator = Evaluator(model, dataset, transformers, verbosity=verbosity)
    with tempfile.NamedTemporaryFile() as csv_out:
      with tempfile.NamedTemporaryFile() as stats_out:
        scores = evaluator.compute_model_performance(
            [classification_metric], csv_out.name, stats_out)

    print("sklearn.metrics.accuracy_score(y, y_pred)")
    print(sklearn.metrics.accuracy_score(y, y_pred))

    print("metrics.compute_roc_auc_scores(y, y_pred_proba_model)")
    print(metrics.compute_roc_auc_scores(y, y_pred_proba_model))

    print("scores")
    print(scores)
    assert scores[classification_metric.name] > .9
