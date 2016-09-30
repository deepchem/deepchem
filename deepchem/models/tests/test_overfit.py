"""
Tests to make sure deepchem models can overfit on tiny datasets.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "GPL"

import tempfile
import numpy as np
import unittest
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from deepchem import metrics
from deepchem.datasets import Dataset
from deepchem.metrics import Metric
from deepchem.models.tests import TestAPI
from deepchem.utils.evaluate import Evaluator
from deepchem.models.sklearn_models import SklearnModel
from deepchem.models.keras_models import KerasModel
from deepchem.models.keras_models.fcnet import MultiTaskDNN
from deepchem.models.tensorflow_models import TensorflowModel
from deepchem.models.tensorflow_models.fcnet import TensorflowMultiTaskRegressor
from deepchem.models.tensorflow_models.fcnet import TensorflowMultiTaskClassifier
from deepchem.models.multitask import SingletaskToMultitask
import tensorflow as tf
from keras import backend as K

class TestOverfitAPI(TestAPI):
  """
  Test that models can overfit simple datasets.
  """

  def test_sklearn_regression_overfit(self):
    """Test that sklearn models can overfit simple regression datasets."""
    n_samples = 10
    n_features = 3
    n_tasks = 1
    
    # Generate dummy dataset
    np.random.seed(123)
    ids = np.arange(n_samples)
    X = np.random.rand(n_samples, n_features)
    y = np.random.rand(n_samples, n_tasks)
    w = np.ones((n_samples, n_tasks))
    dataset = Dataset.from_numpy(self.train_dir, X, y, w, ids)

    verbosity = "high"
    regression_metric = Metric(metrics.r2_score, verbosity=verbosity)
    sklearn_model = RandomForestRegressor()
    model = SklearnModel(sklearn_model, self.model_dir)

    # Fit trained model
    model.fit(dataset)
    model.save()

    # Eval model on train
    transformers = []
    evaluator = Evaluator(model, dataset, transformers, verbosity=verbosity)
    scores = evaluator.compute_model_performance([regression_metric])

    assert scores[regression_metric.name] > .7

  def test_sklearn_classification_overfit(self):
    """Test that sklearn models can overfit simple classification datasets."""
    n_samples = 10
    n_features = 3
    n_tasks = 1
    
    # Generate dummy dataset
    np.random.seed(123)
    ids = np.arange(n_samples)
    X = np.random.rand(n_samples, n_features)
    y = np.random.randint(2, size=(n_samples, n_tasks))
    w = np.ones((n_samples, n_tasks))
    dataset = Dataset.from_numpy(self.train_dir, X, y, w, ids)

    verbosity = "high"
    classification_metric = Metric(metrics.roc_auc_score, verbosity=verbosity)
    sklearn_model = RandomForestClassifier()
    model = SklearnModel(sklearn_model, self.model_dir)

    # Fit trained model
    model.fit(dataset)
    model.save()

    # Eval model on train
    transformers = []
    evaluator = Evaluator(model, dataset, transformers, verbosity=verbosity)
    scores = evaluator.compute_model_performance([classification_metric])

    assert scores[classification_metric.name] > .9

  def test_sklearn_skewed_classification_overfit(self):
    """Test sklearn models can overfit 0/1 datasets with few actives."""
    n_samples = 100
    n_features = 3
    n_tasks = 1
    
    # Generate dummy dataset
    np.random.seed(123)
    p = .05
    ids = np.arange(n_samples)
    X = np.random.rand(n_samples, n_features)
    y = np.random.binomial(1, p, size=(n_samples, n_tasks))
    w = np.ones((n_samples, n_tasks))
  
    dataset = Dataset.from_numpy(self.train_dir, X, y, w, ids)

    verbosity = "high"
    classification_metric = Metric(metrics.roc_auc_score, verbosity=verbosity)
    sklearn_model = RandomForestClassifier()
    model = SklearnModel(sklearn_model, self.model_dir)

    # Fit trained model
    model.fit(dataset)
    model.save()

    # Eval model on train
    transformers = []
    evaluator = Evaluator(model, dataset, transformers, verbosity=verbosity)
    scores = evaluator.compute_model_performance([classification_metric])

    assert scores[classification_metric.name] > .9

  def test_keras_regression_overfit(self):
    """Test that keras models can overfit simple regression datasets."""
    g = tf.Graph()
    sess = tf.Session(graph=g)
    K.set_session(sess)
    with g.as_default():
      n_samples = 10
      n_features = 3
      n_tasks = 1 
      
      # Generate dummy dataset
      np.random.seed(123)
      ids = np.arange(n_samples)
      X = np.random.rand(n_samples, n_features)
      y = np.random.rand(n_samples, n_tasks)
      w = np.ones((n_samples, n_tasks))

      dataset = Dataset.from_numpy(self.train_dir, X, y, w, ids)

      verbosity = "high"
      regression_metric = Metric(metrics.r2_score, verbosity=verbosity)
      keras_model = MultiTaskDNN(n_tasks, n_features, "regression",
                                 dropout=0., learning_rate=.15, decay=1e-4)
      model = KerasModel(keras_model, self.model_dir)

      # Fit trained model
      model.fit(dataset, nb_epoch=200)
      model.save()

      # Eval model on train
      transformers = []
      evaluator = Evaluator(model, dataset, transformers, verbosity=verbosity)
      scores = evaluator.compute_model_performance([regression_metric])

      assert scores[regression_metric.name] > .7

  def test_tf_regression_overfit(self):
    """Test that TensorFlow models can overfit simple regression datasets."""
    n_samples = 10
    n_features = 3
    n_tasks = 1
    
    # Generate dummy dataset
    np.random.seed(123)
    ids = np.arange(n_samples)
    X = np.random.rand(n_samples, n_features)
    y = np.zeros((n_samples, n_tasks))
    w = np.ones((n_samples, n_tasks))
    dataset = Dataset.from_numpy(self.train_dir, X, y, w, ids)

    verbosity = "high"
    regression_metric = Metric(metrics.mean_squared_error, verbosity=verbosity)
    # TODO(rbharath): This breaks with optimizer="momentum". Why?
    tensorflow_model = TensorflowMultiTaskRegressor(
        n_tasks, n_features, self.model_dir, dropouts=[0.],
        learning_rate=0.003, weight_init_stddevs=[np.sqrt(6)/np.sqrt(1000)],
        batch_size=n_samples, verbosity=verbosity)
    model = TensorflowModel(tensorflow_model, self.model_dir)

    # Fit trained model
    model.fit(dataset, nb_epoch=100)
    model.save()

    # Eval model on train
    transformers = []
    evaluator = Evaluator(model, dataset, transformers, verbosity=verbosity)
    scores = evaluator.compute_model_performance([regression_metric])

    assert scores[regression_metric.name] < .1

  def test_keras_classification_overfit(self):
    """Test that keras models can overfit simple classification datasets."""
    g = tf.Graph()
    sess = tf.Session(graph=g)
    K.set_session(sess)
    with g.as_default():
      n_samples = 10
      n_features = 3
      n_tasks = 1
      
      # Generate dummy dataset
      np.random.seed(123)
      ids = np.arange(n_samples)
      X = np.random.rand(n_samples, n_features)
      y = np.random.randint(2, size=(n_samples, n_tasks))
      w = np.ones((n_samples, n_tasks))
    
      dataset = Dataset.from_numpy(self.train_dir, X, y, w, ids)

      verbosity = "high"
      classification_metric = Metric(metrics.roc_auc_score, verbosity=verbosity)
      keras_model  = MultiTaskDNN(n_tasks, n_features, "classification",
                                  learning_rate=.15, decay=1e-4, dropout=0.)
      model = KerasModel(keras_model, self.model_dir)
      

      # Fit trained model
      model.fit(dataset, nb_epoch=200)
      model.save()

      # Eval model on train
      transformers = []
      evaluator = Evaluator(model, dataset, transformers, verbosity=verbosity)
      scores = evaluator.compute_model_performance([classification_metric])

      assert scores[classification_metric.name] > .9

  def test_keras_skewed_classification_overfit(self):
    """Test keras models can overfit 0/1 datasets with few actives."""
    g = tf.Graph()
    sess = tf.Session(graph=g)
    K.set_session(sess)
    with g.as_default():
      n_samples = 100
      n_features = 3
      n_tasks = 1
      
      # Generate dummy dataset
      np.random.seed(123)
      p = .05
      ids = np.arange(n_samples)
      X = np.random.rand(n_samples, n_features)
      y = np.random.binomial(1, p, size=(n_samples, n_tasks))
      w = np.ones((n_samples, n_tasks))
    
      dataset = Dataset.from_numpy(self.train_dir, X, y, w, ids)

      verbosity = "high"
      classification_metric = Metric(metrics.roc_auc_score, verbosity=verbosity)
      keras_model = MultiTaskDNN(n_tasks, n_features, "classification",
                                 dropout=0., learning_rate=.15, decay=1e-4)
      model = KerasModel(keras_model, self.model_dir)

      # Fit trained model
      model.fit(dataset, batch_size=n_samples, nb_epoch=200)
      model.save()

      # Eval model on train
      transformers = []
      evaluator = Evaluator(model, dataset, transformers, verbosity=verbosity)
      scores = evaluator.compute_model_performance([classification_metric])

      assert scores[classification_metric.name] > .9

  def test_tf_classification_overfit(self):
    """Test that tensorflow models can overfit simple classification datasets."""
    n_samples = 10
    n_features = 3
    n_tasks = 1
    n_classes = 2
    
    # Generate dummy dataset
    np.random.seed(123)
    ids = np.arange(n_samples)
    X = np.random.rand(n_samples, n_features)
    y = np.zeros((n_samples, n_tasks))
    w = np.ones((n_samples, n_tasks))
    dataset = Dataset.from_numpy(self.train_dir, X, y, w, ids)

    verbosity = "high"
    classification_metric = Metric(metrics.accuracy_score, verbosity=verbosity)
    tensorflow_model = TensorflowMultiTaskClassifier(
        n_tasks, n_features, self.model_dir, dropouts=[0.],
        learning_rate=0.0003, weight_init_stddevs=[.1],
        batch_size=n_samples, verbosity=verbosity)
    model = TensorflowModel(tensorflow_model, self.model_dir)

    # Fit trained model
    model.fit(dataset, nb_epoch=100)
    model.save()

    # Eval model on train
    transformers = []
    evaluator = Evaluator(model, dataset, transformers, verbosity=verbosity)
    scores = evaluator.compute_model_performance([classification_metric])

    assert scores[classification_metric.name] > .9

  def test_tf_skewed_classification_overfit(self):
    """Test tensorflow models can overfit 0/1 datasets with few actives."""
    #n_samples = 100
    n_samples = 100
    n_features = 3
    n_tasks = 1
    n_classes = 2
    
    # Generate dummy dataset
    np.random.seed(123)
    p = .05
    ids = np.arange(n_samples)
    X = np.random.rand(n_samples, n_features)
    y = np.random.binomial(1, p, size=(n_samples, n_tasks))
    w = np.ones((n_samples, n_tasks))
  
    dataset = Dataset.from_numpy(self.train_dir, X, y, w, ids)

    verbosity = "high"
    classification_metric = Metric(metrics.roc_auc_score, verbosity=verbosity)
    tensorflow_model = TensorflowMultiTaskClassifier(
        n_tasks, n_features, self.model_dir, dropouts=[0.],
        learning_rate=0.003, weight_init_stddevs=[.1],
        batch_size=n_samples, verbosity=verbosity)
    model = TensorflowModel(tensorflow_model, self.model_dir)

    # Fit trained model
    model.fit(dataset, nb_epoch=100)
    model.save()

    # Eval model on train
    transformers = []
    evaluator = Evaluator(model, dataset, transformers, verbosity=verbosity)
    scores = evaluator.compute_model_performance([classification_metric])

    assert scores[classification_metric.name] > .75

  def test_tf_skewed_missing_classification_overfit(self):
    """TF, skewed data, few actives

    Test tensorflow models overfit 0/1 datasets with missing data and few
    actives. This is intended to be as close to singletask MUV datasets as
    possible.
    """
    n_samples = 5120
    n_features = 6
    n_tasks = 1
    n_classes = 2
    
    # Generate dummy dataset
    np.random.seed(123)
    p = .002
    ids = np.arange(n_samples)
    X = np.random.rand(n_samples, n_features)
    y = np.random.binomial(1, p, size=(n_samples, n_tasks))
    w = np.ones((n_samples, n_tasks))
    y_flat, w_flat = np.squeeze(y), np.squeeze(w)
    y_nonzero = y_flat[w_flat != 0]
    num_nonzero = np.count_nonzero(y_nonzero)
    weight_nonzero = len(y_nonzero)/num_nonzero
    w_flat[y_flat != 0] = weight_nonzero
    w = np.reshape(w_flat, (n_samples, n_tasks))
  
    dataset = Dataset.from_numpy(self.train_dir, X, y, w, ids)

    verbosity = "high"
    classification_metric = Metric(metrics.roc_auc_score, verbosity=verbosity)
    tensorflow_model = TensorflowMultiTaskClassifier(
        n_tasks, n_features, self.model_dir, dropouts=[0.],
        learning_rate=0.003, weight_init_stddevs=[1.],
        batch_size=n_samples, verbosity=verbosity)
    model = TensorflowModel(tensorflow_model, self.model_dir)

    # Fit trained model
    model.fit(dataset, nb_epoch=50)
    model.save()

    # Eval model on train
    transformers = []
    evaluator = Evaluator(model, dataset, transformers, verbosity=verbosity)
    scores = evaluator.compute_model_performance([classification_metric])

    assert scores[classification_metric.name] > .8

  def test_sklearn_multitask_classification_overfit(self):
    """Test SKLearn singletask-to-multitask overfits tiny data."""
    n_tasks = 10
    tasks = ["task%d" % task for task in range(n_tasks)]
    n_samples = 10
    n_features = 3
    
    # Generate dummy dataset
    np.random.seed(123)
    ids = np.arange(n_samples)
    X = np.random.rand(n_samples, n_features)
    y = np.random.randint(2, size=(n_samples, n_tasks))
    w = np.ones((n_samples, n_tasks))
  
    dataset = Dataset.from_numpy(self.train_dir, X, y, w, ids)

    verbosity = "high"
    classification_metric = Metric(metrics.roc_auc_score, verbosity=verbosity, task_averager=np.mean)
    def model_builder(model_dir):
      sklearn_model = RandomForestClassifier()
      return SklearnModel(sklearn_model, model_dir)
    model = SingletaskToMultitask(tasks, model_builder, self.model_dir)

    # Fit trained model
    model.fit(dataset)
    model.save()

    # Eval model on train
    transformers = []
    evaluator = Evaluator(model, dataset, transformers, verbosity=verbosity)
    scores = evaluator.compute_model_performance([classification_metric])

    assert scores[classification_metric.name] > .9

  def test_keras_multitask_classification_overfit(self):
    """Test keras multitask overfits tiny data."""
    g = tf.Graph()
    sess = tf.Session(graph=g)
    K.set_session(sess)
    with g.as_default():
      n_tasks = 10
      n_samples = 10
      n_features = 3
      
      # Generate dummy dataset
      np.random.seed(123)
      ids = np.arange(n_samples)
      X = np.random.rand(n_samples, n_features)
      y = np.random.randint(2, size=(n_samples, n_tasks))
      w = np.ones((n_samples, n_tasks))
      dataset = Dataset.from_numpy(self.train_dir, X, y, w, ids)

      verbosity = "high"
      classification_metric = Metric(
          metrics.roc_auc_score, verbosity=verbosity,
         task_averager=np.mean, mode="classification")
      keras_model = MultiTaskDNN(n_tasks, n_features, "classification",
                                 dropout=0., learning_rate=.15, decay=1e-4)
      model = KerasModel(keras_model, self.model_dir, verbosity=verbosity)

      # Fit trained model
      model.fit(dataset, nb_epoch=50)
      model.save()

      # Eval model on train
      transformers = []
      evaluator = Evaluator(model, dataset, transformers, verbosity=verbosity)
      scores = evaluator.compute_model_performance([classification_metric])
      assert scores[classification_metric.name] > .9

  def test_tf_multitask_classification_overfit(self):
    """Test tf multitask overfits tiny data."""
    n_tasks = 10
    n_samples = 10
    n_features = 3
    n_classes = 2
    
    # Generate dummy dataset
    np.random.seed(123)
    ids = np.arange(n_samples)
    X = np.random.rand(n_samples, n_features)
    y = np.zeros((n_samples, n_tasks))
    w = np.ones((n_samples, n_tasks))
  
    dataset = Dataset.from_numpy(self.train_dir, X, y, w, ids)

    verbosity = "high"
    classification_metric = Metric(metrics.accuracy_score, verbosity=verbosity, task_averager=np.mean)
    tensorflow_model = TensorflowMultiTaskClassifier(
        n_tasks, n_features, self.model_dir, dropouts=[0.],
        learning_rate=0.0003, weight_init_stddevs=[.1],
        batch_size=n_samples, verbosity=verbosity)
    model = TensorflowModel(tensorflow_model, self.model_dir)

    # Fit trained model
    model.fit(dataset)
    model.save()

    # Eval model on train
    transformers = []
    evaluator = Evaluator(model, dataset, transformers, verbosity=verbosity)
    scores = evaluator.compute_model_performance([classification_metric])

    assert scores[classification_metric.name] > .9

  def test_sklearn_multitask_regression_overfit(self):
    """Test SKLearn singletask-to-multitask overfits tiny regression data."""
    n_tasks = 2
    tasks = ["task%d" % task for task in range(n_tasks)]
    n_samples = 10
    n_features = 3
    
    # Generate dummy dataset
    np.random.seed(123)
    ids = np.arange(n_samples)
    X = np.random.rand(n_samples, n_features)
    y = np.random.rand(n_samples, n_tasks)
    w = np.ones((n_samples, n_tasks))

    dataset = Dataset.from_numpy(self.train_dir, X, y, w, ids)

    verbosity = "high"
    regression_metric = Metric(metrics.r2_score, verbosity=verbosity, task_averager=np.mean)
    def model_builder(model_dir):
      sklearn_model = RandomForestRegressor()
      return SklearnModel(sklearn_model, model_dir)
    model = SingletaskToMultitask(tasks, model_builder, self.model_dir)

    # Fit trained model
    model.fit(dataset)
    model.save()

    # Eval model on train
    transformers = []
    evaluator = Evaluator(model, dataset, transformers, verbosity=verbosity)
    scores = evaluator.compute_model_performance([regression_metric])

    assert scores[regression_metric.name] > .7

  def test_keras_multitask_regression_overfit(self):
    """Test keras multitask overfits tiny data."""
    g = tf.Graph()
    sess = tf.Session(graph=g)
    K.set_session(sess)
    with g.as_default():
      n_tasks = 10
      n_samples = 10
      n_features = 3
      
      # Generate dummy dataset
      np.random.seed(123)
      ids = np.arange(n_samples)
      X = np.random.rand(n_samples, n_features)
      y = np.random.randint(2, size=(n_samples, n_tasks))
      w = np.ones((n_samples, n_tasks))
      dataset = Dataset.from_numpy(self.train_dir, X, y, w, ids)

      verbosity = "high"
      regression_metric = Metric(metrics.r2_score, verbosity=verbosity,
                                 task_averager=np.mean, mode="regression")
      keras_model = MultiTaskDNN(n_tasks, n_features, "regression",
                                 dropout=0., learning_rate=.1, decay=1e-4)
      model = KerasModel(keras_model, self.model_dir, verbosity=verbosity)

      # Fit trained model
      model.fit(dataset, nb_epoch=100)
      model.save()

      # Eval model on train
      transformers = []
      evaluator = Evaluator(model, dataset, transformers, verbosity=verbosity)
      scores = evaluator.compute_model_performance([regression_metric])

      assert scores[regression_metric.name] > .75

  def test_tf_multitask_regression_overfit(self):
    """Test tf multitask overfits tiny data."""
    n_tasks = 10
    n_samples = 10
    n_features = 3
    n_classes = 2
    
    # Generate dummy dataset
    np.random.seed(123)
    ids = np.arange(n_samples)
    X = np.random.rand(n_samples, n_features)
    y = np.zeros((n_samples, n_tasks))
    w = np.ones((n_samples, n_tasks))
  
    dataset = Dataset.from_numpy(self.train_dir, X, y, w, ids)

    model_params = {
      "layer_sizes": [1000],
      "dropouts": [.0],
      "learning_rate": 0.0003,
      "momentum": .9,
      "batch_size": n_samples,
      "num_regression_tasks": n_tasks,
      "num_classes": n_classes,
      "num_features": n_features,
      "weight_init_stddevs": [.1],
      "bias_init_consts": [1.],
      "nb_epoch": 100,
      "penalty": 0.0,
      "optimizer": "adam",
      "data_shape": dataset.get_data_shape()
    }

    verbosity = "high"
    regression_metric = Metric(metrics.mean_squared_error, verbosity=verbosity,
                               task_averager=np.mean, mode="regression")
    tensorflow_model = TensorflowMultiTaskRegressor(
        n_tasks, n_features, self.model_dir, dropouts=[0.],
        learning_rate=0.0003, weight_init_stddevs=[.1],
        batch_size=n_samples, verbosity=verbosity)
    model = TensorflowModel(tensorflow_model, self.model_dir)

    # Fit trained model
    model.fit(dataset, nb_epoch=50)
    model.save()

    # Eval model on train
    transformers = []
    evaluator = Evaluator(model, dataset, transformers, verbosity=verbosity)
    scores = evaluator.compute_model_performance([regression_metric])

    assert scores[regression_metric.name] < .1
