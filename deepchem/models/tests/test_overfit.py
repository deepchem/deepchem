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
from keras import backend as K
from keras.layers import Dense, BatchNormalization
from tensorflow.python.framework import test_util
from deepchem.featurizers.featurize import DataLoader
from deepchem.featurizers.fingerprints import CircularFingerprint
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from deepchem import metrics
from deepchem.datasets import DiskDataset, NumpyDataset
from deepchem.metrics import Metric
from deepchem.models.tests import TestAPI
from deepchem.utils.evaluate import Evaluator
from deepchem.models.sklearn_models import SklearnModel
from deepchem.models.keras_models import KerasModel
from deepchem.models.keras_models.fcnet import MultiTaskDNN
from deepchem.models.tensorflow_models import TensorflowModel
from deepchem.models.tensorflow_models.fcnet import TensorflowMultiTaskRegressor
from deepchem.models.tensorflow_models.fcnet import TensorflowMultiTaskClassifier
from deepchem.models.tensorflow_models.robust_multitask import RobustMultitaskRegressor
from deepchem.models.multitask import SingletaskToMultitask
from deepchem.models.tf_keras_models.graph_models import SequentialGraphModel
from deepchem.models.tf_keras_models.keras_layers import GraphConv
from deepchem.models.tf_keras_models.keras_layers import GraphPool
from deepchem.models.tf_keras_models.keras_layers import GraphGather
from deepchem.featurizers.graph_features import ConvMolFeaturizer
from deepchem.models.tf_keras_models.graph_models import SequentialSupportGraphModel
from deepchem.models.tf_keras_models.multitask_classifier import MultitaskGraphClassifier
from deepchem.models.tf_keras_models.support_classifier import SupportGraphClassifier
from deepchem.models.tf_keras_models.keras_layers import AttnLSTMEmbedding
from deepchem.models.tf_keras_models.keras_layers import ResiLSTMEmbedding

class TestOverfitAPI(test_util.TensorFlowTestCase):
  """
  Test that models can overfit simple datasets.
  """
  def setUp(self):
    super(TestOverfitAPI, self).setUp()
    self.root = '/tmp'
    self.smiles_field = "smiles"
    self.current_dir = os.path.dirname(os.path.abspath(__file__))
    self.train_dir = tempfile.mkdtemp()
    self.data_dir = tempfile.mkdtemp()
    self.model_dir = tempfile.mkdtemp()

  def tearDown(self):
    shutil.rmtree(self.train_dir)
    shutil.rmtree(self.data_dir)

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
    dataset = NumpyDataset(X, y, w, ids)

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
    dataset = NumpyDataset(X, y, w, ids)

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
  
    dataset = NumpyDataset(X, y, w, ids)

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

      dataset = NumpyDataset(X, y, w, ids)

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
    dataset = NumpyDataset(X, y, w, ids)

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
    
      dataset = NumpyDataset(X, y, w, ids)

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
    
      dataset = NumpyDataset(X, y, w, ids)

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
    dataset = NumpyDataset(X, y, w, ids)

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
  
    dataset = NumpyDataset(X, y, w, ids)

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
  
    dataset = NumpyDataset(X, y, w, ids)

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
  
    dataset = DiskDataset.from_numpy(self.train_dir, X, y, w, ids)

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
      dataset = NumpyDataset(X, y, w, ids)

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
  
    dataset = NumpyDataset(X, y, w, ids)

    verbosity = "high"
    classification_metric = Metric(metrics.accuracy_score, verbosity=verbosity,
                                   task_averager=np.mean)
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

    dataset = DiskDataset.from_numpy(self.train_dir, X, y, w, ids)

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
      dataset = NumpyDataset(X, y, w, ids)

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
  
    dataset = NumpyDataset(X, y, w, ids)

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

  def test_tf_robust_multitask_regression_overfit(self):
    """Test tf robust multitask overfits tiny data."""
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
  
    dataset = NumpyDataset(X, y, w, ids)

    verbosity = "high"
    regression_metric = Metric(metrics.mean_squared_error, verbosity=verbosity,
                               task_averager=np.mean, mode="regression")
    tensorflow_model = RobustMultitaskRegressor(
        n_tasks, n_features, self.model_dir, layer_sizes=[50],
        bypass_layer_sizes=[10], dropouts=[0.],
        learning_rate=0.003, weight_init_stddevs=[.1],
        batch_size=n_samples, verbosity=verbosity)
    model = TensorflowModel(tensorflow_model, self.model_dir)

    # Fit trained model
    model.fit(dataset, nb_epoch=25)
    model.save()

    # Eval model on train
    transformers = []
    evaluator = Evaluator(model, dataset, transformers, verbosity=verbosity)
    scores = evaluator.compute_model_performance([regression_metric])

    assert scores[regression_metric.name] < .15

  def test_graph_conv_singletask_classification_overfit(self):
    """Test graph-conv multitask overfits tiny data."""
    np.random.seed(123)
    tf.set_random_seed(123)
    g = tf.Graph()
    sess = tf.Session(graph=g)
    K.set_session(sess)
    with g.as_default():
      n_tasks = 1
      n_samples = 10
      n_features = 3
      n_classes = 2
      
      # Load mini log-solubility dataset.
      splittype = "scaffold"
      featurizer = ConvMolFeaturizer()
      tasks = ["outcome"]
      task_type = "classification"
      task_types = {task: task_type for task in tasks}
      input_file = os.path.join(self.current_dir, "example_classification.csv")
      loader = DataLoader(tasks=tasks,
                          smiles_field=self.smiles_field,
                          featurizer=featurizer,
                          verbosity="low")
      dataset = loader.featurize(input_file, self.data_dir)

      verbosity = "high"
      classification_metric = Metric(metrics.accuracy_score, verbosity=verbosity)

      #n_atoms = 50
      n_feat = 71
      batch_size = 10
      graph_model = SequentialGraphModel(n_feat)
      graph_model.add(GraphConv(64, activation='relu'))
      graph_model.add(BatchNormalization(epsilon=1e-5, mode=1))
      graph_model.add(GraphPool())
      # Gather Projection
      graph_model.add(Dense(128, activation='relu'))
      graph_model.add(BatchNormalization(epsilon=1e-5, mode=1))
      graph_model.add(GraphGather(batch_size, activation="tanh"))

      with self.test_session() as sess:
        model = MultitaskGraphClassifier(
          sess, graph_model, n_tasks, self.model_dir, batch_size=batch_size,
          learning_rate=1e-3, learning_rate_decay_time=1000,
          optimizer_type="adam", beta1=.9, beta2=.999, verbosity="high")

        # Fit trained model
        model.fit(dataset, nb_epoch=20)
        model.save()

        # Eval model on train
        transformers = []
        evaluator = Evaluator(model, dataset, transformers, verbosity=verbosity)
        scores = evaluator.compute_model_performance([classification_metric])

      ######################################################### DEBUG
      print("scores")
      print(scores)
      ######################################################### DEBUG
      assert scores[classification_metric.name] > .85

  def test_attn_lstm_singletask_classification_overfit(self):
    """Test support graph-conv multitask overfits tiny data."""
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
      replace = False
      
      # Load mini log-solubility dataset.
      splittype = "scaffold"
      featurizer = ConvMolFeaturizer()
      tasks = ["outcome"]
      task_type = "classification"
      task_types = {task: task_type for task in tasks}
      input_file = os.path.join(self.current_dir, "example_classification.csv")
      loader = DataLoader(tasks=tasks,
                          smiles_field=self.smiles_field,
                          featurizer=featurizer,
                          verbosity="low")
      dataset = loader.featurize(input_file, self.data_dir)

      verbosity = "high"
      classification_metric = Metric(metrics.accuracy_score, verbosity=verbosity)

      support_model = SequentialSupportGraphModel(n_feat)
      
      # Add layers
      # output will be (n_atoms, 64)
      support_model.add(GraphConv(64, activation='relu'))
      # Need to add batch-norm separately to test/support due to differing
      # shapes.
      # output will be (n_atoms, 64)
      support_model.add_test(BatchNormalization(epsilon=1e-5, mode=1))
      # output will be (n_atoms, 64)
      support_model.add_support(BatchNormalization(epsilon=1e-5, mode=1))
      support_model.add(GraphPool())
      support_model.add_test(GraphGather(test_batch_size))
      support_model.add_support(GraphGather(support_batch_size))

      # Apply an attention lstm layer
      support_model.join(AttnLSTMEmbedding(test_batch_size, support_batch_size,
                                           max_depth))

      with self.test_session() as sess:
        model = SupportGraphClassifier(
          sess, support_model, n_tasks, self.model_dir, 
          test_batch_size=test_batch_size, support_batch_size=support_batch_size,
          learning_rate=1e-3, learning_rate_decay_time=1000,
          optimizer_type="adam", beta1=.9, beta2=.999, verbosity="high")

        # Fit trained model. Dataset has 6 positives and 4 negatives, so set
        # n_pos/n_neg accordingly.  Set replace to false to ensure full dataset
        # is always passed in to support.

        # TODO(rbharath): Why does this work with 0 epochs?!!!
        # I think it's because the distance calculation is still meaningful
        # even with random features. The cutoffs also mean that the outputted
        # scores vectors threshold at logit(epsilon), logit(1-epsilon) and stay
        # fixed.
        model.fit(dataset, nb_epoch=1, n_trials_per_epoch=10, n_pos=n_pos, n_neg=n_neg,
                  replace=False)
        model.save()

        # Eval model on train. Dataset has 6 positives and 4 negatives, so set
        # n_pos/n_neg accordingly. Note that support is *not* excluded (so we
        # can measure model has memorized support).  Replacement is turned off to
        # ensure that support contains full training set. This checks that the
        # model has mastered memorization of provided support.
        scores = model.evaluate(dataset, range(n_tasks),
                                classification_metric, n_trials=5,
                                n_pos=n_pos, n_neg=n_neg,
                                exclude_support=False, replace=False)

      # Measure performance on 0-th task.
      assert scores[0] > .9

  '''
  TODO(rbharath): This test doesn't pass although it should. Debug to understand root
  causes of these errors.
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
      replace = False
      
      # Load mini log-solubility dataset.
      splittype = "scaffold"
      featurizer = ConvMolFeaturizer()
      tasks = ["outcome"]
      task_type = "classification"
      task_types = {task: task_type for task in tasks}
      input_file = os.path.join(self.current_dir, "example_classification.csv")
      loader = DataLoader(tasks=tasks,
                          smiles_field=self.smiles_field,
                          featurizer=featurizer,
                          verbosity="low")
      dataset = loader.featurize(input_file, self.data_dir)

      verbosity = "high"
      classification_metric = Metric(metrics.accuracy_score, verbosity=verbosity)

      support_model = SequentialSupportGraphModel(n_feat)
      
      # Add layers
      # output will be (n_atoms, 64)
      support_model.add(GraphConv(64, activation='relu'))
      # Need to add batch-norm separately to test/support due to differing
      # shapes.
      # output will be (n_atoms, 64)
      support_model.add_test(BatchNormalization(epsilon=1e-5, mode=1))
      # output will be (n_atoms, 64)
      support_model.add_support(BatchNormalization(epsilon=1e-5, mode=1))
      support_model.add(GraphPool())
      support_model.add_test(GraphGather(test_batch_size))
      support_model.add_support(GraphGather(support_batch_size))

      # Apply an attention lstm layer
      support_model.join(ResiLSTMEmbedding(test_batch_size, support_batch_size,
                                           max_depth))

      with self.test_session() as sess:
        model = SupportGraphClassifier(
          sess, support_model, n_tasks, self.model_dir, 
          test_batch_size=test_batch_size, support_batch_size=support_batch_size,
          learning_rate=1e-3, learning_rate_decay_time=1000,
          optimizer_type="adam", beta1=.9, beta2=.999, verbosity="high")

        # Fit trained model. Dataset has 6 positives and 4 negatives, so set
        # n_pos/n_neg accordingly.  Set replace to false to ensure full dataset
        # is always passed in to support.

        model.fit(dataset, nb_epoch=10, n_trials_per_epoch=10, n_pos=n_pos, n_neg=n_neg,
                  replace=False)
        model.save()

        # Eval model on train. Dataset has 6 positives and 4 negatives, so set
        # n_pos/n_neg accordingly. Note that support is *not* excluded (so we
        # can measure model has memorized support).  Replacement is turned off to
        # ensure that support contains full training set. This checks that the
        # model has mastered memorization of provided support.
        scores = model.evaluate(dataset, range(n_tasks),
                                classification_metric, n_trials=5,
                                n_pos=n_pos, n_neg=n_neg,
                                exclude_support=False, replace=False)

      # Measure performance on 0-th task.
      assert scores[0] > .9
  '''
