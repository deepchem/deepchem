"""
Test reload for trained models.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "GPL"

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from deepchem.models.tests import TestAPI
from deepchem import metrics
from deepchem.metrics import Metric
from deepchem.datasets import Dataset
from deepchem.utils.evaluate import Evaluator
from deepchem.models.sklearn_models import SklearnModel
from deepchem.models.keras_models import KerasModel
from deepchem.models.keras_models.fcnet import MultiTaskDNN
from deepchem.models.tensorflow_models import TensorflowModel
from deepchem.models.tensorflow_models.fcnet import TensorflowMultiTaskClassifier
import tensorflow as tf
from keras import backend as K

class TestModelReload(TestAPI):

  def test_sklearn_reload(self):
    """Test that trained model can be reloaded correctly."""
    tasks = ["task0"]
    task_types = {task: "classification" for task in tasks}
    n_samples = 10
    n_features = 3
    n_tasks = len(tasks)
    
    # Generate dummy dataset
    np.random.seed(123)
    ids = np.arange(n_samples)
    X = np.random.rand(n_samples, n_features)
    y = np.random.randint(2, size=(n_samples, n_tasks))
    w = np.ones((n_samples, n_tasks))
  
    dataset = Dataset.from_numpy(self.train_dir, X, y, w, ids, tasks)

    verbosity = "high"
    classification_metric = Metric(metrics.roc_auc_score, verbosity=verbosity)

    sklearn_model = RandomForestClassifier()
    model = SklearnModel(sklearn_model, self.model_dir)

    # Fit trained model
    model.fit(dataset)
    model.save()

    # Load trained model
    reloaded_model = SklearnModel(None, self.model_dir)
    reloaded_model.reload()

    # Eval model on train
    transformers = []
    evaluator = Evaluator(reloaded_model, dataset, transformers, verbosity=verbosity)
    scores = evaluator.compute_model_performance([classification_metric])

    assert scores[classification_metric.name] > .9

  def test_keras_reload(self):
    """Test that trained keras models can be reloaded correctly."""
    g = tf.Graph()
    sess = tf.Session(graph=g)
    K.set_session(sess)
    with g.as_default():
      tasks = ["task0"]
      task_types = {task: "classification" for task in tasks}
      n_samples = 10
      n_features = 3
      n_tasks = len(tasks)
      
      # Generate dummy dataset
      np.random.seed(123)
      ids = np.arange(n_samples)
      X = np.random.rand(n_samples, n_features)
      y = np.random.randint(2, size=(n_samples, n_tasks))
      w = np.ones((n_samples, n_tasks))
    
      dataset = Dataset.from_numpy(self.train_dir, X, y, w, ids, tasks)

      verbosity = "high"
      classification_metric = Metric(metrics.roc_auc_score, verbosity=verbosity)
      keras_model = MultiTaskDNN(n_tasks, n_features, "classification",
                                 dropout=0.)
      model = KerasModel(keras_model, self.model_dir)

      # Fit trained model
      model.fit(dataset)
      model.save()

      # Load trained model
      reloaded_keras_model = MultiTaskDNN(
          n_tasks, n_features, "classification", dropout=0.)
      reloaded_model = KerasModel(reloaded_keras_model, self.model_dir)
      reloaded_model.reload(custom_objects={"MultiTaskDNN": MultiTaskDNN})
      

      # Eval model on train
      transformers = []
      evaluator = Evaluator(reloaded_model, dataset, transformers,
                            verbosity=verbosity)
      scores = evaluator.compute_model_performance([classification_metric])

      assert scores[classification_metric.name] > .6

  def test_tf_reload(self):
    """Test that tensorflow models can overfit simple classification datasets."""
    n_samples = 10
    n_features = 3
    n_tasks = 1 
    n_classes = 2
    
    # Generate dummy dataset
    np.random.seed(123)
    ids = np.arange(n_samples)
    X = np.random.rand(n_samples, n_features)
    y = np.random.randint(n_classes, size=(n_samples, n_tasks))
    w = np.ones((n_samples, n_tasks))
  
    dataset = Dataset.from_numpy(self.train_dir, X, y, w, ids)

    verbosity = "high"
    classification_metric = Metric(metrics.accuracy_score, verbosity=verbosity)

    tensorflow_model = TensorflowMultiTaskClassifier(
          n_tasks, n_features, self.model_dir, dropouts=[0.],
          verbosity=verbosity)
    model = TensorflowModel(tensorflow_model, self.model_dir)

    # Fit trained model
    model.fit(dataset)
    model.save()

    # Load trained model
    reloaded_tensorflow_model = TensorflowMultiTaskClassifier(
          n_tasks, n_features, self.model_dir, dropouts=[0.],
          verbosity=verbosity)
    reloaded_model = TensorflowModel(reloaded_tensorflow_model, self.model_dir)
    reloaded_model.reload()

    # Eval model on train
    transformers = []
    evaluator = Evaluator(reloaded_model, dataset, transformers, verbosity=verbosity)
    scores = evaluator.compute_model_performance([classification_metric])

    assert scores[classification_metric.name] > .6
