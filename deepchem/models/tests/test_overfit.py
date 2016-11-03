"""
Tests to make sure deepchem models can overfit on tiny datasets.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "GPL"

import os
import tempfile
import numpy as np
import unittest
import sklearn
import shutil
import tensorflow as tf
import deepchem as dc
from keras import backend as K
from tensorflow.python.framework import test_util
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

class TestOverfit(test_util.TensorFlowTestCase):
  """
  Test that models can overfit simple datasets.
  """
  def setUp(self):
    super(TestOverfit, self).setUp()
    self.current_dir = os.path.dirname(os.path.abspath(__file__))

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
    dataset = dc.data.NumpyDataset(X, y, w, ids)

    verbosity = "high"
    regression_metric = dc.metrics.Metric(
        dc.metrics.r2_score, verbosity=verbosity)
    sklearn_model = RandomForestRegressor()
    model = dc.models.SklearnModel(sklearn_model)

    # Fit trained model
    model.fit(dataset)
    model.save()

    # Eval model on train
    scores = model.evaluate(dataset, [regression_metric])
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
    dataset = dc.data.NumpyDataset(X, y, w, ids)

    verbosity = "high"
    classification_metric = dc.metrics.Metric(
        dc.metrics.roc_auc_score, verbosity=verbosity)
    sklearn_model = RandomForestClassifier()
    model = dc.models.SklearnModel(sklearn_model)

    # Fit trained model
    model.fit(dataset)
    model.save()

    # Eval model on train
    scores = model.evaluate(dataset, [classification_metric])
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
  
    dataset = dc.data.NumpyDataset(X, y, w, ids)

    verbosity = "high"
    classification_metric = dc.metrics.Metric(
        dc.metrics.roc_auc_score, verbosity=verbosity)
    sklearn_model = RandomForestClassifier()
    model = dc.models.SklearnModel(sklearn_model)

    # Fit trained model
    model.fit(dataset)
    model.save()

    # Eval model on train
    scores = model.evaluate(dataset, [classification_metric])
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

      dataset = dc.data.NumpyDataset(X, y, w, ids)

      verbosity = "high"
      regression_metric = dc.metrics.Metric(
          dc.metrics.r2_score, verbosity=verbosity)
      keras_model = dc.models.MultiTaskDNN(
          n_tasks, n_features, "regression",
          dropout=0., learning_rate=.15, decay=1e-4)
      model = dc.models.KerasModel(keras_model)

      # Fit trained model
      model.fit(dataset, nb_epoch=200)
      model.save()

      # Eval model on train
      scores = model.evaluate(dataset, [regression_metric])
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
    dataset = dc.data.NumpyDataset(X, y, w, ids)

    verbosity = "high"
    regression_metric = dc.metrics.Metric(
        dc.metrics.mean_squared_error, verbosity=verbosity)
    # TODO(rbharath): This breaks with optimizer="momentum". Why?
    tensorflow_model = dc.models.TensorflowMultiTaskRegressor(
        n_tasks, n_features, dropouts=[0.],
        learning_rate=0.003, weight_init_stddevs=[np.sqrt(6)/np.sqrt(1000)],
        batch_size=n_samples, verbosity=verbosity)
    model = dc.models.TensorflowModel(tensorflow_model)

    # Fit trained model
    model.fit(dataset, nb_epoch=100)
    model.save()

    # Eval model on train
    scores = model.evaluate(dataset, [regression_metric])
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
    
      dataset = dc.data.NumpyDataset(X, y, w, ids)

      verbosity = "high"
      classification_metric = dc.metrics.Metric(
          dc.metrics.roc_auc_score, verbosity=verbosity)
      keras_model = dc.models.MultiTaskDNN(
          n_tasks, n_features, "classification",
          learning_rate=.15, decay=1e-4, dropout=0.)
      model = dc.models.KerasModel(keras_model)

      # Fit trained model
      model.fit(dataset, nb_epoch=200)
      model.save()

      # Eval model on train
      scores = model.evaluate(dataset, [classification_metric])
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
    
      dataset = dc.data.NumpyDataset(X, y, w, ids)

      verbosity = "high"
      classification_metric = dc.metrics.Metric(
          dc.metrics.roc_auc_score, verbosity=verbosity)
      keras_model = dc.models.MultiTaskDNN(
          n_tasks, n_features, "classification",
          dropout=0., learning_rate=.15, decay=1e-4)
      model = dc.models.KerasModel(keras_model)

      # Fit trained model
      model.fit(dataset, batch_size=n_samples, nb_epoch=200)
      model.save()

      # Eval model on train
      scores = model.evaluate(dataset, [classification_metric])
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
    dataset = dc.data.NumpyDataset(X, y, w, ids)

    verbosity = "high"
    classification_metric = dc.metrics.Metric(
        dc.metrics.accuracy_score, verbosity=verbosity)
    tensorflow_model = dc.models.TensorflowMultiTaskClassifier(
        n_tasks, n_features, dropouts=[0.],
        learning_rate=0.0003, weight_init_stddevs=[.1],
        batch_size=n_samples, verbosity=verbosity)
    model = dc.models.TensorflowModel(tensorflow_model)

    # Fit trained model
    model.fit(dataset, nb_epoch=100)
    model.save()

    # Eval model on train
    scores = model.evaluate(dataset, [classification_metric])
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
  
    dataset = dc.data.NumpyDataset(X, y, w, ids)

    verbosity = "high"
    classification_metric = dc.metrics.Metric(
        dc.metrics.roc_auc_score, verbosity=verbosity)
    tensorflow_model = dc.models.TensorflowMultiTaskClassifier(
        n_tasks, n_features, dropouts=[0.],
        learning_rate=0.003, weight_init_stddevs=[.1],
        batch_size=n_samples, verbosity=verbosity)
    model = dc.models.TensorflowModel(tensorflow_model)

    # Fit trained model
    model.fit(dataset, nb_epoch=100)
    model.save()

    # Eval model on train
    scores = model.evaluate(dataset, [classification_metric])
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
  
    dataset = dc.data.DiskDataset.from_numpy(X, y, w, ids)

    verbosity = "high"
    classification_metric = dc.metrics.Metric(
        dc.metrics.roc_auc_score, verbosity=verbosity)
    tensorflow_model = dc.models.TensorflowMultiTaskClassifier(
        n_tasks, n_features, dropouts=[0.],
        learning_rate=0.003, weight_init_stddevs=[1.],
        batch_size=n_samples, verbosity=verbosity)
    model = dc.models.TensorflowModel(tensorflow_model)

    # Fit trained model
    model.fit(dataset, nb_epoch=50)
    model.save()

    # Eval model on train
    scores = model.evaluate(dataset, [classification_metric])
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
    dataset = dc.data.DiskDataset.from_numpy(X, y, w, ids)

    classification_metric = dc.metrics.Metric(
        dc.metrics.roc_auc_score, verbosity="high", task_averager=np.mean)
    def model_builder(model_dir):
      sklearn_model = RandomForestClassifier()
      return dc.models.SklearnModel(sklearn_model, model_dir)
    model = dc.models.SingletaskToMultitask(tasks, model_builder)

    # Fit trained model
    model.fit(dataset)
    model.save()

    # Eval model on train
    scores = model.evaluate(dataset, [classification_metric])
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
      dataset = dc.data.NumpyDataset(X, y, w, ids)

      verbosity = "high"
      classification_metric = dc.metrics.Metric(
          dc.metrics.roc_auc_score, verbosity=verbosity,
         task_averager=np.mean, mode="classification")
      keras_model = dc.models.MultiTaskDNN(
          n_tasks, n_features, "classification", dropout=0., learning_rate=.15,
          decay=1e-4)
      model = dc.models.KerasModel(keras_model, verbosity=verbosity)

      # Fit trained model
      model.fit(dataset, nb_epoch=50)
      model.save()

      # Eval model on train
      scores = model.evaluate(dataset, [classification_metric])
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
    dataset = dc.data.NumpyDataset(X, y, w, ids)

    verbosity = "high"
    classification_metric = dc.metrics.Metric(
      dc.metrics.accuracy_score, verbosity=verbosity, task_averager=np.mean)
    tensorflow_model = dc.models.TensorflowMultiTaskClassifier(
        n_tasks, n_features, dropouts=[0.],
        learning_rate=0.0003, weight_init_stddevs=[.1],
        batch_size=n_samples, verbosity=verbosity)
    model = dc.models.TensorflowModel(tensorflow_model)

    # Fit trained model
    model.fit(dataset)
    model.save()

    # Eval model on train
    scores = model.evaluate(dataset, [classification_metric])
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

    dataset = dc.data.DiskDataset.from_numpy(X, y, w, ids)

    regression_metric = dc.metrics.Metric(
        dc.metrics.r2_score, verbosity="high", task_averager=np.mean)
    def model_builder(model_dir):
      sklearn_model = RandomForestRegressor()
      return dc.models.SklearnModel(sklearn_model, model_dir)
    model = dc.models.SingletaskToMultitask(tasks, model_builder)

    # Fit trained model
    model.fit(dataset)
    model.save()

    # Eval model on train
    scores = model.evaluate(dataset, [regression_metric])
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
      dataset = dc.data.NumpyDataset(X, y, w, ids)

      verbosity = "high"
      regression_metric = dc.metrics.Metric(
          dc.metrics.r2_score, verbosity=verbosity, task_averager=np.mean,
          mode="regression")
      keras_model = dc.models.MultiTaskDNN(
          n_tasks, n_features, "regression", dropout=0., learning_rate=.1,
          decay=1e-4)
      model = dc.models.KerasModel(keras_model, verbosity=verbosity)

      # Fit trained model
      model.fit(dataset, nb_epoch=100)
      model.save()

      # Eval model on train
      scores = model.evaluate(dataset, [regression_metric])
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
  
    dataset = dc.data.NumpyDataset(X, y, w, ids)

    verbosity = "high"
    regression_metric = dc.metrics.Metric(
        dc.metrics.mean_squared_error, verbosity=verbosity,
        task_averager=np.mean, mode="regression")
    tensorflow_model = dc.models.TensorflowMultiTaskRegressor(
        n_tasks, n_features, dropouts=[0.],
        learning_rate=0.0003, weight_init_stddevs=[.1],
        batch_size=n_samples, verbosity=verbosity)
    model = dc.models.TensorflowModel(tensorflow_model)

    # Fit trained model
    model.fit(dataset, nb_epoch=50)
    model.save()

    # Eval model on train
    scores = model.evaluate(dataset, [regression_metric])
    assert scores[regression_metric.name] < .1

  def test_tf_robust_multitask_regression_overfit(self):
    """Test tf robust multitask overfits tiny data."""
    np.random.seed(123)
    tf.set_random_seed(123)
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
  
    dataset = dc.data.NumpyDataset(X, y, w, ids)

    verbosity = "high"
    regression_metric = dc.metrics.Metric(
        dc.metrics.mean_squared_error, verbosity=verbosity,
        task_averager=np.mean, mode="regression")
    tensorflow_model = dc.models.RobustMultitaskRegressor(
        n_tasks, n_features, layer_sizes=[50],
        bypass_layer_sizes=[10], dropouts=[0.],
        learning_rate=0.003, weight_init_stddevs=[.1],
        batch_size=n_samples, verbosity=verbosity)
    model = dc.models.TensorflowModel(tensorflow_model)

    # Fit trained model
    model.fit(dataset, nb_epoch=25)
    model.save()

    # Eval model on train
    scores = model.evaluate(dataset, [regression_metric])
    assert scores[regression_metric.name] < .2

  def test_graph_conv_singletask_classification_overfit(self):
    """Test graph-conv multitask overfits tiny data."""
    np.random.seed(123)
    tf.set_random_seed(123)
    g = tf.Graph()
    sess = tf.Session(graph=g)
    K.set_session(sess)
    verbosity = "high"
    with g.as_default():
      n_tasks = 1
      n_samples = 10
      n_features = 3
      n_classes = 2
      
      # Load mini log-solubility dataset.
      featurizer = dc.feat.ConvMolFeaturizer()
      tasks = ["outcome"]
      input_file = os.path.join(self.current_dir, "example_classification.csv")
      loader = dc.load.DataLoader(
          tasks=tasks, smiles_field="smiles", featurizer=featurizer,
          verbosity=verbosity)
      dataset = loader.featurize(input_file)

      classification_metric = dc.metrics.Metric(
          dc.metrics.accuracy_score, verbosity=verbosity)

      n_feat = 71
      batch_size = 10
      graph_model = dc.nn.SequentialGraph(n_feat)
      graph_model.add(dc.nn.GraphConv(64, activation='relu'))
      graph_model.add(dc.nn.BatchNormalization(epsilon=1e-5, mode=1))
      graph_model.add(dc.nn.GraphPool())
      # Gather Projection
      graph_model.add(dc.nn.Dense(128, activation='relu'))
      graph_model.add(dc.nn.BatchNormalization(epsilon=1e-5, mode=1))
      graph_model.add(dc.nn.GraphGather(batch_size, activation="tanh"))

      with self.test_session() as sess:
        model = dc.models.MultitaskGraphClassifier(
          sess, graph_model, n_tasks, batch_size=batch_size,
          learning_rate=1e-3, learning_rate_decay_time=1000,
          optimizer_type="adam", beta1=.9, beta2=.999, verbosity="high")

        # Fit trained model
        model.fit(dataset, nb_epoch=20)
        model.save()

        # Eval model on train
        scores = model.evaluate(dataset, [classification_metric])

      assert scores[classification_metric.name] > .75

  def test_siamese_singletask_classification_overfit(self):
    """Test siamese singletask model overfits tiny data."""
    np.random.seed(123)
    tf.set_random_seed(123)
    g = tf.Graph()
    sess = tf.Session(graph=g)
    K.set_session(sess)
    verbosity = "high"
    with g.as_default():
      n_tasks = 1
      n_feat = 71
      max_depth = 4
      n_pos = 6
      n_neg = 4
      test_batch_size = 10
      n_train_trials = 60
      support_batch_size = n_pos + n_neg
      replace = False
      
      # Load mini log-solubility dataset.
      featurizer = dc.feat.ConvMolFeaturizer()
      tasks = ["outcome"]
      input_file = os.path.join(self.current_dir, "example_classification.csv")
      loader = dc.load.DataLoader(
          tasks=tasks, smiles_field="smiles",
          featurizer=featurizer, verbosity=verbosity)
      dataset = loader.featurize(input_file)

      classification_metric = dc.metrics.Metric(
          dc.metrics.accuracy_score, verbosity=verbosity)

      support_model = dc.nn.SequentialSupportGraph(n_feat)
      
      # Add layers
      # output will be (n_atoms, 64)
      support_model.add(dc.nn.GraphConv(64, activation='relu'))
      # Need to add batch-norm separately to test/support due to differing
      # shapes.
      # output will be (n_atoms, 64)
      support_model.add_test(dc.nn.BatchNormalization(epsilon=1e-5, mode=1))
      # output will be (n_atoms, 64)
      support_model.add_support(dc.nn.BatchNormalization(epsilon=1e-5, mode=1))
      support_model.add(dc.nn.GraphPool())
      support_model.add_test(dc.nn.GraphGather(test_batch_size))
      support_model.add_support(dc.nn.GraphGather(support_batch_size))

      with self.test_session() as sess:
        model = dc.models.SupportGraphClassifier(
          sess, support_model, test_batch_size=test_batch_size,
          support_batch_size=support_batch_size, learning_rate=1e-3,
          verbosity="high")

        # Fit trained model. Dataset has 6 positives and 4 negatives, so set
        # n_pos/n_neg accordingly.  Set replace to false to ensure full dataset
        # is always passed in to support.
        model.fit(dataset, n_trials=n_train_trials, n_pos=n_pos,
                  n_neg=n_neg, replace=False)
        model.save()

        # Eval model on train. Dataset has 6 positives and 4 negatives, so set
        # n_pos/n_neg accordingly. Note that support is *not* excluded (so we
        # can measure model has memorized support).  Replacement is turned off to
        # ensure that support contains full training set. This checks that the
        # model has mastered memorization of provided support.
        scores = model.evaluate(dataset, classification_metric, n_trials=5,
                                n_pos=n_pos, n_neg=n_neg,
                                exclude_support=False)

      # Measure performance on 0-th task.
      assert scores[0] > .9

  def test_attn_lstm_singletask_classification_overfit(self):
    """Test attn lstm singletask overfits tiny data."""
    np.random.seed(123)
    tf.set_random_seed(123)
    g = tf.Graph()
    sess = tf.Session(graph=g)
    K.set_session(sess)
    with g.as_default():
      n_tasks = 1
      n_feat = 71
      max_depth = 4
      n_pos = 6
      n_neg = 4
      test_batch_size = 10
      support_batch_size = n_pos + n_neg
      n_train_trials = 60
      replace = False
      
      # Load mini log-solubility dataset.
      featurizer = dc.feat.ConvMolFeaturizer()
      tasks = ["outcome"]
      input_file = os.path.join(self.current_dir, "example_classification.csv")
      loader = dc.load.DataLoader(
          tasks=tasks, smiles_field="smiles", featurizer=featurizer,
          verbosity="low")
      dataset = loader.featurize(input_file)

      verbosity = "high"
      classification_metric = dc.metrics.Metric(
          dc.metrics.accuracy_score, verbosity=verbosity)

      support_model = dc.nn.SequentialSupportGraph(n_feat)
      
      # Add layers
      # output will be (n_atoms, 64)
      support_model.add(dc.nn.GraphConv(64, activation='relu'))
      # Need to add batch-norm separately to test/support due to differing
      # shapes.
      # output will be (n_atoms, 64)
      support_model.add_test(dc.nn.BatchNormalization(epsilon=1e-5, mode=1))
      # output will be (n_atoms, 64)
      support_model.add_support(dc.nn.BatchNormalization(epsilon=1e-5, mode=1))
      support_model.add(dc.nn.GraphPool())
      support_model.add_test(dc.nn.GraphGather(test_batch_size))
      support_model.add_support(dc.nn.GraphGather(support_batch_size))

      # Apply an attention lstm layer
      support_model.join(dc.nn.AttnLSTMEmbedding(
          test_batch_size, support_batch_size, max_depth))

      with self.test_session() as sess:
        model = dc.models.SupportGraphClassifier(
          sess, support_model, test_batch_size=test_batch_size,
          support_batch_size=support_batch_size, learning_rate=1e-3,
          verbosity="high")

        # Fit trained model. Dataset has 6 positives and 4 negatives, so set
        # n_pos/n_neg accordingly.  Set replace to false to ensure full dataset
        # is always passed in to support.
        model.fit(dataset, n_trials=n_train_trials, n_pos=n_pos, n_neg=n_neg,
                  replace=False)
        model.save()

        # Eval model on train. Dataset has 6 positives and 4 negatives, so set
        # n_pos/n_neg accordingly. Note that support is *not* excluded (so we
        # can measure model has memorized support).  Replacement is turned off to
        # ensure that support contains full training set. This checks that the
        # model has mastered memorization of provided support.
        scores = model.evaluate(dataset, classification_metric, n_trials=5,
                                n_pos=n_pos, n_neg=n_neg,
                                exclude_support=False)

      # Measure performance on 0-th task.
      assert scores[0] > .9

  def test_residual_lstm_singletask_classification_overfit(self):
    """Test resi-lstm multitask overfits tiny data."""
    g = tf.Graph()
    sess = tf.Session(graph=g)
    K.set_session(sess)
    with g.as_default():
      n_tasks = 1
      n_feat = 71
      max_depth = 4
      n_pos = 6
      n_neg = 4
      test_batch_size = 10
      support_batch_size = n_pos + n_neg
      n_train_trials = 60
      replace = False
      
      # Load mini log-solubility dataset.
      featurizer = dc.feat.ConvMolFeaturizer()
      tasks = ["outcome"]
      input_file = os.path.join(self.current_dir, "example_classification.csv")
      loader = dc.load.DataLoader(
          tasks=tasks, smiles_field="smiles",
          featurizer=featurizer, verbosity="low")
      dataset = loader.featurize(input_file)

      verbosity = "high"
      classification_metric = dc.metrics.Metric(
          dc.metrics.accuracy_score, verbosity=verbosity)

      support_model = dc.nn.SequentialSupportGraph(n_feat)
      
      # Add layers
      # output will be (n_atoms, 64)
      support_model.add(dc.nn.GraphConv(64, activation='relu'))
      # Need to add batch-norm separately to test/support due to differing
      # shapes.
      # output will be (n_atoms, 64)
      support_model.add_test(dc.nn.BatchNormalization(epsilon=1e-5, mode=1))
      # output will be (n_atoms, 64)
      support_model.add_support(dc.nn.BatchNormalization(epsilon=1e-5, mode=1))
      support_model.add(dc.nn.GraphPool())
      support_model.add_test(dc.nn.GraphGather(test_batch_size))
      support_model.add_support(dc.nn.GraphGather(support_batch_size))

      # Apply a residual lstm layer
      support_model.join(dc.nn.ResiLSTMEmbedding(
          test_batch_size, support_batch_size, max_depth))

      with self.test_session() as sess:
        model = dc.models.SupportGraphClassifier(
          sess, support_model, test_batch_size=test_batch_size,
          support_batch_size=support_batch_size, learning_rate=1e-3,
          verbosity="high")

        # Fit trained model. Dataset has 6 positives and 4 negatives, so set
        # n_pos/n_neg accordingly.  Set replace to false to ensure full dataset
        # is always passed in to support.

        model.fit(dataset, n_trials=n_train_trials, n_pos=n_pos, n_neg=n_neg,
                  replace=False)
        model.save()

        # Eval model on train. Dataset has 6 positives and 4 negatives, so set
        # n_pos/n_neg accordingly. Note that support is *not* excluded (so we
        # can measure model has memorized support).  Replacement is turned off to
        # ensure that support contains full training set. This checks that the
        # model has mastered memorization of provided support.
        scores = model.evaluate(dataset, classification_metric, n_trials=5,
                                n_pos=n_pos, n_neg=n_neg,
                                exclude_support=False)

      # Measure performance on 0-th task.
      assert scores[0] > .9
