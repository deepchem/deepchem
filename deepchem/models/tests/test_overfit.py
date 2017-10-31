"""
Tests to make sure deepchem models can overfit on tiny datasets.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

from nose.plugins.attrib import attr

__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "MIT"

import os
import tempfile
import numpy as np
import unittest
import sklearn
import shutil
import tensorflow as tf
import deepchem as dc
import scipy.io
from deepchem.models.tensorgraph.optimizers import Adam
from tensorflow.python.framework import test_util
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from flaky import flaky


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

    regression_metric = dc.metrics.Metric(dc.metrics.r2_score)
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

    classification_metric = dc.metrics.Metric(dc.metrics.roc_auc_score)
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

    classification_metric = dc.metrics.Metric(dc.metrics.roc_auc_score)
    sklearn_model = RandomForestClassifier()
    model = dc.models.SklearnModel(sklearn_model)

    # Fit trained model
    model.fit(dataset)
    model.save()

    # Eval model on train
    scores = model.evaluate(dataset, [classification_metric])
    assert scores[classification_metric.name] > .9

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

    regression_metric = dc.metrics.Metric(dc.metrics.mean_squared_error)
    # TODO(rbharath): This breaks with optimizer="momentum". Why?
    model = dc.models.TensorflowMultiTaskRegressor(
        n_tasks,
        n_features,
        dropouts=[0.],
        learning_rate=0.003,
        weight_init_stddevs=[np.sqrt(6) / np.sqrt(1000)],
        batch_size=n_samples)

    # Fit trained model
    model.fit(dataset, nb_epoch=100)
    model.save()

    # Eval model on train
    scores = model.evaluate(dataset, [regression_metric])
    assert scores[regression_metric.name] < .1

  def test_tg_regression_overfit(self):
    """Test that TensorGraph models can overfit simple regression datasets."""
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

    regression_metric = dc.metrics.Metric(dc.metrics.mean_squared_error)
    # TODO(rbharath): This breaks with optimizer="momentum". Why?
    model = dc.models.TensorGraphMultiTaskRegressor(
        n_tasks,
        n_features,
        dropouts=[0.],
        weight_init_stddevs=[np.sqrt(6) / np.sqrt(1000)],
        batch_size=n_samples)
    model.set_optimizer(Adam(learning_rate=0.003, beta1=0.9, beta2=0.999))

    # Fit trained model
    model.fit(dataset, nb_epoch=100)
    model.save()

    # Eval model on train
    scores = model.evaluate(dataset, [regression_metric])
    assert scores[regression_metric.name] < .1

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

    classification_metric = dc.metrics.Metric(dc.metrics.accuracy_score)
    model = dc.models.TensorflowMultiTaskClassifier(
        n_tasks,
        n_features,
        dropouts=[0.],
        learning_rate=0.0003,
        weight_init_stddevs=[.1],
        batch_size=n_samples)

    # Fit trained model
    model.fit(dataset, nb_epoch=100)
    model.save()

    # Eval model on train
    scores = model.evaluate(dataset, [classification_metric])
    assert scores[classification_metric.name] > .9

  def test_tg_classification_overfit(self):
    """Test that TensorGraph models can overfit simple classification datasets."""
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

    classification_metric = dc.metrics.Metric(dc.metrics.accuracy_score)
    model = dc.models.TensorGraphMultiTaskClassifier(
        n_tasks,
        n_features,
        dropouts=[0.],
        weight_init_stddevs=[.1],
        batch_size=n_samples)
    model.set_optimizer(Adam(learning_rate=0.0003, beta1=0.9, beta2=0.999))

    # Fit trained model
    model.fit(dataset, nb_epoch=100)
    model.save()

    # Eval model on train
    scores = model.evaluate(dataset, [classification_metric])
    assert scores[classification_metric.name] > .9

  def test_tf_fittransform_regression_overfit(self):
    """Test that TensorFlow FitTransform models can overfit simple regression datasets."""
    n_samples = 10
    n_features = 3
    n_tasks = 1

    # Generate dummy dataset
    np.random.seed(123)
    ids = np.arange(n_samples)
    X = np.random.rand(n_samples, n_features, n_features)
    y = np.zeros((n_samples, n_tasks))
    w = np.ones((n_samples, n_tasks))
    dataset = dc.data.NumpyDataset(X, y, w, ids)

    fit_transformers = [dc.trans.CoulombFitTransformer(dataset)]
    regression_metric = dc.metrics.Metric(dc.metrics.mean_squared_error)
    model = dc.models.TensorflowMultiTaskFitTransformRegressor(
        n_tasks, [n_features, n_features],
        dropouts=[0.],
        learning_rate=0.003,
        weight_init_stddevs=[np.sqrt(6) / np.sqrt(1000)],
        batch_size=n_samples,
        fit_transformers=fit_transformers,
        n_evals=1)

    # Fit trained model
    model.fit(dataset, nb_epoch=100)
    model.save()

    # Eval model on train
    scores = model.evaluate(dataset, [regression_metric])
    assert scores[regression_metric.name] < .1

  def test_tg_fittransform_regression_overfit(self):
    """Test that TensorGraph FitTransform models can overfit simple regression datasets."""
    n_samples = 10
    n_features = 3
    n_tasks = 1

    # Generate dummy dataset
    np.random.seed(123)
    ids = np.arange(n_samples)
    X = np.random.rand(n_samples, n_features, n_features)
    y = np.zeros((n_samples, n_tasks))
    w = np.ones((n_samples, n_tasks))
    dataset = dc.data.NumpyDataset(X, y, w, ids)

    fit_transformers = [dc.trans.CoulombFitTransformer(dataset)]
    regression_metric = dc.metrics.Metric(dc.metrics.mean_squared_error)
    model = dc.models.TensorGraphMultiTaskFitTransformRegressor(
        n_tasks, [n_features, n_features],
        dropouts=[0.],
        weight_init_stddevs=[np.sqrt(6) / np.sqrt(1000)],
        batch_size=n_samples,
        fit_transformers=fit_transformers,
        n_evals=1)
    model.set_optimizer(Adam(learning_rate=0.003, beta1=0.9, beta2=0.999))

    # Fit trained model
    model.fit(dataset, nb_epoch=100)
    model.save()

    # Eval model on train
    scores = model.evaluate(dataset, [regression_metric])
    assert scores[regression_metric.name] < .1

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

    classification_metric = dc.metrics.Metric(dc.metrics.roc_auc_score)
    model = dc.models.TensorflowMultiTaskClassifier(
        n_tasks,
        n_features,
        dropouts=[0.],
        learning_rate=0.003,
        weight_init_stddevs=[.1],
        batch_size=n_samples)

    # Fit trained model
    model.fit(dataset, nb_epoch=100)
    model.save()

    # Eval model on train
    scores = model.evaluate(dataset, [classification_metric])
    assert scores[classification_metric.name] > .75

  def test_tg_skewed_classification_overfit(self):
    """Test TensorGraph models can overfit 0/1 datasets with few actives."""
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

    classification_metric = dc.metrics.Metric(dc.metrics.roc_auc_score)
    model = dc.models.TensorGraphMultiTaskClassifier(
        n_tasks,
        n_features,
        dropouts=[0.],
        weight_init_stddevs=[.1],
        batch_size=n_samples)
    model.set_optimizer(Adam(learning_rate=0.003, beta1=0.9, beta2=0.999))

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
    weight_nonzero = len(y_nonzero) / num_nonzero
    w_flat[y_flat != 0] = weight_nonzero
    w = np.reshape(w_flat, (n_samples, n_tasks))

    dataset = dc.data.DiskDataset.from_numpy(X, y, w, ids)

    classification_metric = dc.metrics.Metric(dc.metrics.roc_auc_score)
    model = dc.models.TensorflowMultiTaskClassifier(
        n_tasks,
        n_features,
        dropouts=[0.],
        learning_rate=0.003,
        weight_init_stddevs=[1.],
        batch_size=n_samples)

    # Fit trained model
    model.fit(dataset, nb_epoch=50)
    model.save()

    # Eval model on train
    scores = model.evaluate(dataset, [classification_metric])
    assert scores[classification_metric.name] > .8

  def test_tg_skewed_missing_classification_overfit(self):
    """TG, skewed data, few actives

    Test TensorGraph models overfit 0/1 datasets with missing data and few
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
    weight_nonzero = len(y_nonzero) / num_nonzero
    w_flat[y_flat != 0] = weight_nonzero
    w = np.reshape(w_flat, (n_samples, n_tasks))

    dataset = dc.data.DiskDataset.from_numpy(X, y, w, ids)

    classification_metric = dc.metrics.Metric(dc.metrics.roc_auc_score)
    model = dc.models.TensorGraphMultiTaskClassifier(
        n_tasks,
        n_features,
        dropouts=[0.],
        weight_init_stddevs=[1.],
        batch_size=n_samples)
    model.set_optimizer(Adam(learning_rate=0.003, beta1=0.9, beta2=0.999))

    # Fit trained model
    model.fit(dataset, nb_epoch=100)
    model.save()

    # Eval model on train
    scores = model.evaluate(dataset, [classification_metric])
    assert scores[classification_metric.name] > .7

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
        dc.metrics.roc_auc_score, task_averager=np.mean)

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

    classification_metric = dc.metrics.Metric(
        dc.metrics.accuracy_score, task_averager=np.mean)
    model = dc.models.TensorflowMultiTaskClassifier(
        n_tasks,
        n_features,
        dropouts=[0.],
        learning_rate=0.0003,
        weight_init_stddevs=[.1],
        batch_size=n_samples)

    # Fit trained model
    model.fit(dataset)
    model.save()

    # Eval model on train
    scores = model.evaluate(dataset, [classification_metric])
    assert scores[classification_metric.name] > .9

  @flaky
  def test_tg_multitask_classification_overfit(self):
    """Test TensorGraph multitask overfits tiny data."""
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

    classification_metric = dc.metrics.Metric(
        dc.metrics.accuracy_score, task_averager=np.mean)
    model = dc.models.TensorGraphMultiTaskClassifier(
        n_tasks,
        n_features,
        dropouts=[0.],
        weight_init_stddevs=[.1],
        batch_size=n_samples)
    model.set_optimizer(Adam(learning_rate=0.0003, beta1=0.9, beta2=0.999))

    # Fit trained model
    model.fit(dataset)
    model.save()

    # Eval model on train
    scores = model.evaluate(dataset, [classification_metric])
    assert scores[classification_metric.name] > .9

  def test_tf_robust_multitask_classification_overfit(self):
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
    dataset = dc.data.NumpyDataset(X, y, w, ids)

    classification_metric = dc.metrics.Metric(
        dc.metrics.accuracy_score, task_averager=np.mean)
    model = dc.models.RobustMultitaskClassifier(
        n_tasks,
        n_features,
        layer_sizes=[50],
        bypass_layer_sizes=[10],
        dropouts=[0.],
        learning_rate=0.003,
        weight_init_stddevs=[.1],
        batch_size=n_samples)

    # Fit trained model
    model.fit(dataset, nb_epoch=25)
    model.save()

    # Eval model on train
    scores = model.evaluate(dataset, [classification_metric])
    assert scores[classification_metric.name] > .9

  def test_tf_logreg_multitask_classification_overfit(self):
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

    classification_metric = dc.metrics.Metric(
        dc.metrics.accuracy_score, task_averager=np.mean)
    model = dc.models.TensorflowLogisticRegression(
        n_tasks,
        n_features,
        learning_rate=0.5,
        weight_init_stddevs=[.01],
        batch_size=n_samples)

    # Fit trained model
    model.fit(dataset)
    model.save()

    # Eval model on train
    scores = model.evaluate(dataset, [classification_metric])
    assert scores[classification_metric.name] > .9

  def test_IRV_multitask_classification_overfit(self):
    """Test IRV classifier overfits tiny data."""
    n_tasks = 5
    n_samples = 10
    n_features = 128
    n_classes = 2

    # Generate dummy dataset
    np.random.seed(123)
    ids = np.arange(n_samples)
    X = np.random.randint(2, size=(n_samples, n_features))
    y = np.ones((n_samples, n_tasks))
    w = np.ones((n_samples, n_tasks))
    dataset = dc.data.NumpyDataset(X, y, w, ids)
    IRV_transformer = dc.trans.IRVTransformer(5, n_tasks, dataset)
    dataset_trans = IRV_transformer.transform(dataset)
    classification_metric = dc.metrics.Metric(
        dc.metrics.accuracy_score, task_averager=np.mean)
    model = dc.models.TensorflowMultiTaskIRVClassifier(
        n_tasks, K=5, learning_rate=0.01, batch_size=n_samples)

    # Fit trained model
    model.fit(dataset_trans)
    model.save()

    # Eval model on train
    scores = model.evaluate(dataset_trans, [classification_metric])
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
        dc.metrics.r2_score, task_averager=np.mean)

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

  @flaky
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

    regression_metric = dc.metrics.Metric(
        dc.metrics.mean_squared_error, task_averager=np.mean, mode="regression")
    model = dc.models.TensorflowMultiTaskRegressor(
        n_tasks,
        n_features,
        dropouts=[0.],
        learning_rate=0.0003,
        weight_init_stddevs=[.1],
        batch_size=n_samples)

    # Fit trained model
    model.fit(dataset, nb_epoch=50)
    model.save()

    # Eval model on train
    scores = model.evaluate(dataset, [regression_metric])
    assert scores[regression_metric.name] < .1

  def test_tg_multitask_regression_overfit(self):
    """Test TensorGraph multitask overfits tiny data."""
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

    regression_metric = dc.metrics.Metric(
        dc.metrics.mean_squared_error, task_averager=np.mean, mode="regression")
    model = dc.models.TensorGraphMultiTaskRegressor(
        n_tasks,
        n_features,
        dropouts=[0.],
        weight_init_stddevs=[.1],
        batch_size=n_samples)
    model.set_optimizer(Adam(learning_rate=0.0003, beta1=0.9, beta2=0.999))

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

    regression_metric = dc.metrics.Metric(
        dc.metrics.mean_squared_error, task_averager=np.mean, mode="regression")
    model = dc.models.RobustMultitaskRegressor(
        n_tasks,
        n_features,
        layer_sizes=[50],
        bypass_layer_sizes=[10],
        dropouts=[0.],
        learning_rate=0.003,
        weight_init_stddevs=[.1],
        batch_size=n_samples)

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
    n_tasks = 1
    n_samples = 10
    n_features = 3
    n_classes = 2

    # Load mini log-solubility dataset.
    featurizer = dc.feat.ConvMolFeaturizer()
    tasks = ["outcome"]
    input_file = os.path.join(self.current_dir, "example_classification.csv")
    loader = dc.data.CSVLoader(
        tasks=tasks, smiles_field="smiles", featurizer=featurizer)
    dataset = loader.featurize(input_file)

    classification_metric = dc.metrics.Metric(dc.metrics.accuracy_score)

    n_feat = 75
    batch_size = 10

    graph_model = dc.nn.SequentialGraph(n_feat)
    graph_model.add(dc.nn.GraphConv(64, n_feat, activation='relu'))
    graph_model.add(dc.nn.BatchNormalization(epsilon=1e-5, mode=1))
    graph_model.add(dc.nn.GraphPool())
    # Gather Projection
    graph_model.add(dc.nn.Dense(128, 64, activation='relu'))
    graph_model.add(dc.nn.BatchNormalization(epsilon=1e-5, mode=1))
    graph_model.add(dc.nn.GraphGather(batch_size, activation="tanh"))

    model = dc.models.MultitaskGraphClassifier(
        graph_model,
        n_tasks,
        n_feat,
        batch_size=batch_size,
        learning_rate=1e-3,
        learning_rate_decay_time=1000,
        optimizer_type="adam",
        beta1=.9,
        beta2=.999)

    # Fit trained model
    model.fit(dataset, nb_epoch=20)
    model.save()

    # Eval model on train
    scores = model.evaluate(dataset, [classification_metric])

    assert scores[classification_metric.name] > .65

  def test_graph_conv_singletask_regression_overfit(self):
    """Test graph-conv multitask overfits tiny data."""
    np.random.seed(123)
    tf.set_random_seed(123)
    g = tf.Graph()
    sess = tf.Session(graph=g)
    n_tasks = 1
    n_samples = 10
    n_features = 3
    n_classes = 2

    # Load mini log-solubility dataset.
    featurizer = dc.feat.ConvMolFeaturizer()
    tasks = ["outcome"]
    input_file = os.path.join(self.current_dir, "example_regression.csv")
    loader = dc.data.CSVLoader(
        tasks=tasks, smiles_field="smiles", featurizer=featurizer)
    dataset = loader.featurize(input_file)

    classification_metric = dc.metrics.Metric(
        dc.metrics.mean_squared_error, task_averager=np.mean)

    n_feat = 75
    batch_size = 10

    graph_model = dc.nn.SequentialGraph(n_feat)
    graph_model.add(dc.nn.GraphConv(64, n_feat, activation='relu'))
    graph_model.add(dc.nn.BatchNormalization(epsilon=1e-5, mode=1))
    graph_model.add(dc.nn.GraphPool())
    # Gather Projection
    graph_model.add(dc.nn.Dense(128, 64))
    graph_model.add(dc.nn.BatchNormalization(epsilon=1e-5, mode=1))
    graph_model.add(dc.nn.GraphGather(batch_size, activation="tanh"))

    model = dc.models.MultitaskGraphRegressor(
        graph_model,
        n_tasks,
        n_feat,
        batch_size=batch_size,
        learning_rate=1e-2,
        learning_rate_decay_time=1000,
        optimizer_type="adam",
        beta1=.9,
        beta2=.999)

    # Fit trained model
    model.fit(dataset, nb_epoch=40)
    model.save()

    # Eval model on train
    scores = model.evaluate(dataset, [classification_metric])

    assert scores[classification_metric.name] < .2

  def test_DTNN_multitask_regression_overfit(self):
    """Test deep tensor neural net overfits tiny data."""
    np.random.seed(123)
    tf.set_random_seed(123)

    input_file = os.path.join(self.current_dir, "example_DTNN.mat")
    dataset = scipy.io.loadmat(input_file)
    X = dataset['X']
    y = dataset['T']
    w = np.ones_like(y)
    dataset = dc.data.DiskDataset.from_numpy(X, y, w, ids=None)
    regression_metric = dc.metrics.Metric(
        dc.metrics.pearson_r2_score, task_averager=np.mean)
    n_tasks = y.shape[1]
    batch_size = 10

    graph_model = dc.nn.SequentialDTNNGraph()
    graph_model.add(dc.nn.DTNNEmbedding(n_embedding=20))
    graph_model.add(dc.nn.DTNNStep(n_embedding=20))
    graph_model.add(dc.nn.DTNNStep(n_embedding=20))
    graph_model.add(dc.nn.DTNNGather(n_embedding=20))
    n_feat = 20
    model = dc.models.MultitaskGraphRegressor(
        graph_model,
        n_tasks,
        n_feat,
        batch_size=batch_size,
        learning_rate=1e-3,
        learning_rate_decay_time=1000,
        optimizer_type="adam",
        beta1=.9,
        beta2=.999)

    # Fit trained model
    model.fit(dataset, nb_epoch=20)
    model.save()

    # Eval model on train
    scores = model.evaluate(dataset, [regression_metric])

    assert scores[regression_metric.name] > .9

  def test_tensorgraph_DTNN_multitask_regression_overfit(self):
    """Test deep tensor neural net overfits tiny data."""
    np.random.seed(123)
    tf.set_random_seed(123)

    input_file = os.path.join(self.current_dir, "example_DTNN.mat")
    dataset = scipy.io.loadmat(input_file)
    X = dataset['X']
    y = dataset['T']
    w = np.ones_like(y)
    dataset = dc.data.DiskDataset.from_numpy(X, y, w, ids=None)
    regression_metric = dc.metrics.Metric(
        dc.metrics.pearson_r2_score, task_averager=np.mean)
    n_tasks = y.shape[1]
    batch_size = 10

    model = dc.models.DTNNTensorGraph(
        n_tasks,
        n_embedding=20,
        n_distance=100,
        batch_size=batch_size,
        learning_rate=0.001,
        use_queue=False,
        mode="regression")

    # Fit trained model
    model.fit(dataset, nb_epoch=20)

    # Eval model on train
    scores = model.evaluate(dataset, [regression_metric])

    assert scores[regression_metric.name] > .9

  @attr('slow')
  def test_ANI_multitask_regression_overfit(self):
    """Test ANI-1 regression overfits tiny data."""
    input_file = os.path.join(self.current_dir, "example_DTNN.mat")
    np.random.seed(123)
    tf.set_random_seed(123)
    dataset = scipy.io.loadmat(input_file)
    X = np.concatenate([np.expand_dims(dataset['Z'], 2), dataset['R']], axis=2)
    X = X[:, :13, :]
    y = dataset['T']
    w = np.ones_like(y)
    dataset = dc.data.DiskDataset.from_numpy(X, y, w, ids=None)
    regression_metric = dc.metrics.Metric(
        dc.metrics.pearson_r2_score, mode="regression")
    n_tasks = y.shape[1]
    batch_size = 10

    transformers = [
        dc.trans.NormalizationTransformer(transform_y=True, dataset=dataset),
    ]

    for transformer in transformers:
      dataset = transformer.transform(dataset)

    model = dc.models.ANIRegression(
        n_tasks,
        13,
        atom_number_cases=[1, 6, 7, 8],
        batch_size=batch_size,
        learning_rate=0.001,
        use_queue=False,
        mode="regression")

    # Fit trained model
    model.fit(dataset, nb_epoch=50)

    # Eval model on train
    scores = model.evaluate(dataset, [regression_metric], transformers[0:1])

    assert scores[regression_metric.name] > .8

  @attr('slow')
  def test_BP_symmetry_function_overfit(self):
    """Test ANI-1 regression overfits tiny data."""
    input_file = os.path.join(self.current_dir, "example_DTNN.mat")
    np.random.seed(123)
    tf.set_random_seed(123)
    dataset = scipy.io.loadmat(input_file)
    X = np.concatenate([np.expand_dims(dataset['Z'], 2), dataset['R']], axis=2)
    X = X[:, :13, :]
    y = dataset['T']
    w = np.ones_like(y)
    dataset = dc.data.DiskDataset.from_numpy(X, y, w, ids=None)
    regression_metric = dc.metrics.Metric(
        dc.metrics.pearson_r2_score, mode="regression")
    n_tasks = y.shape[1]
    batch_size = 10

    transformers = [
        dc.trans.NormalizationTransformer(transform_y=True, dataset=dataset),
    ]

    for transformer in transformers:
      dataset = transformer.transform(dataset)

    model = dc.models.ANIRegression(
        n_tasks,
        13,
        atom_number_cases=[1, 6, 7, 8],
        batch_size=batch_size,
        learning_rate=0.001,
        use_queue=False,
        mode="regression")

    # Fit trained model
    model.fit(dataset, nb_epoch=50)

    # Eval model on train
    scores = model.evaluate(dataset, [regression_metric], transformers[0:1])

    assert scores[regression_metric.name] > .8

  def test_DAG_singletask_regression_overfit(self):
    """Test DAG regressor multitask overfits tiny data."""
    np.random.seed(123)
    tf.set_random_seed(123)
    n_tasks = 1

    # Load mini log-solubility dataset.
    featurizer = dc.feat.ConvMolFeaturizer()
    tasks = ["outcome"]
    input_file = os.path.join(self.current_dir, "example_regression.csv")
    loader = dc.data.CSVLoader(
        tasks=tasks, smiles_field="smiles", featurizer=featurizer)
    dataset = loader.featurize(input_file)

    regression_metric = dc.metrics.Metric(
        dc.metrics.pearson_r2_score, task_averager=np.mean)

    n_feat = 75
    batch_size = 10
    transformer = dc.trans.DAGTransformer(max_atoms=50)
    dataset = transformer.transform(dataset)

    graph = dc.nn.SequentialDAGGraph(n_atom_feat=n_feat, max_atoms=50)
    graph.add(dc.nn.DAGLayer(30, n_feat, max_atoms=50, batch_size=batch_size))
    graph.add(dc.nn.DAGGather(30, max_atoms=50))

    model = dc.models.MultitaskGraphRegressor(
        graph,
        n_tasks,
        n_feat,
        batch_size=batch_size,
        learning_rate=0.001,
        learning_rate_decay_time=1000,
        optimizer_type="adam",
        beta1=.9,
        beta2=.999)

    # Fit trained model
    model.fit(dataset, nb_epoch=50)
    model.save()
    # Eval model on train
    scores = model.evaluate(dataset, [regression_metric])

    assert scores[regression_metric.name] > .8

  def test_tensorgraph_DAG_singletask_regression_overfit(self):
    """Test DAG regressor multitask overfits tiny data."""
    np.random.seed(123)
    tf.set_random_seed(123)
    n_tasks = 1

    # Load mini log-solubility dataset.
    featurizer = dc.feat.ConvMolFeaturizer()
    tasks = ["outcome"]
    input_file = os.path.join(self.current_dir, "example_regression.csv")
    loader = dc.data.CSVLoader(
        tasks=tasks, smiles_field="smiles", featurizer=featurizer)
    dataset = loader.featurize(input_file)

    regression_metric = dc.metrics.Metric(
        dc.metrics.pearson_r2_score, task_averager=np.mean)

    n_feat = 75
    batch_size = 10
    transformer = dc.trans.DAGTransformer(max_atoms=50)
    dataset = transformer.transform(dataset)

    model = dc.models.DAGTensorGraph(
        n_tasks,
        max_atoms=50,
        n_atom_feat=n_feat,
        batch_size=batch_size,
        learning_rate=0.001,
        use_queue=False,
        mode="regression")

    # Fit trained model
    model.fit(dataset, nb_epoch=50)
    # Eval model on train
    scores = model.evaluate(dataset, [regression_metric])

    assert scores[regression_metric.name] > .8

  def test_weave_singletask_classification_overfit(self):
    """Test weave model overfits tiny data."""
    np.random.seed(123)
    tf.set_random_seed(123)
    n_tasks = 1

    # Load mini log-solubility dataset.
    featurizer = dc.feat.WeaveFeaturizer()
    tasks = ["outcome"]
    input_file = os.path.join(self.current_dir, "example_classification.csv")
    loader = dc.data.CSVLoader(
        tasks=tasks, smiles_field="smiles", featurizer=featurizer)
    dataset = loader.featurize(input_file)

    classification_metric = dc.metrics.Metric(dc.metrics.accuracy_score)

    n_atom_feat = 75
    n_pair_feat = 14
    n_feat = 128
    batch_size = 10
    max_atoms = 50

    graph = dc.nn.AlternateSequentialWeaveGraph(
        batch_size,
        max_atoms=max_atoms,
        n_atom_feat=n_atom_feat,
        n_pair_feat=n_pair_feat)
    graph.add(dc.nn.AlternateWeaveLayer(max_atoms, 75, 14))
    graph.add(dc.nn.AlternateWeaveLayer(max_atoms, 50, 50, update_pair=False))
    graph.add(dc.nn.Dense(n_feat, 50, activation='tanh'))
    graph.add(dc.nn.BatchNormalization(epsilon=1e-5, mode=1))
    graph.add(
        dc.nn.AlternateWeaveGather(
            batch_size, n_input=n_feat, gaussian_expand=True))

    model = dc.models.MultitaskGraphClassifier(
        graph,
        n_tasks,
        n_feat,
        batch_size=batch_size,
        learning_rate=1e-3,
        learning_rate_decay_time=1000,
        optimizer_type="adam",
        beta1=.9,
        beta2=.999)

    # Fit trained model
    model.fit(dataset, nb_epoch=20)
    model.save()

    # Eval model on train
    scores = model.evaluate(dataset, [classification_metric])

    assert scores[classification_metric.name] > .65

  def test_tensorgraph_weave_singletask_classification_overfit(self):
    """Test weave model overfits tiny data."""
    np.random.seed(123)
    tf.set_random_seed(123)
    n_tasks = 1

    # Load mini log-solubility dataset.
    featurizer = dc.feat.WeaveFeaturizer()
    tasks = ["outcome"]
    input_file = os.path.join(self.current_dir, "example_classification.csv")
    loader = dc.data.CSVLoader(
        tasks=tasks, smiles_field="smiles", featurizer=featurizer)
    dataset = loader.featurize(input_file)

    classification_metric = dc.metrics.Metric(dc.metrics.accuracy_score)

    n_atom_feat = 75
    n_pair_feat = 14
    n_feat = 128
    batch_size = 10

    model = dc.models.WeaveTensorGraph(
        n_tasks,
        n_atom_feat=n_atom_feat,
        n_pair_feat=n_pair_feat,
        n_graph_feat=n_feat,
        batch_size=batch_size,
        learning_rate=0.001,
        use_queue=False,
        mode="classification")

    # Fit trained model
    model.fit(dataset, nb_epoch=20)

    # Eval model on train
    scores = model.evaluate(dataset, [classification_metric])

    assert scores[classification_metric.name] > .65

  def test_weave_singletask_regression_overfit(self):
    """Test weave model overfits tiny data."""
    np.random.seed(123)
    tf.set_random_seed(123)
    n_tasks = 1

    # Load mini log-solubility dataset.
    featurizer = dc.feat.WeaveFeaturizer()
    tasks = ["outcome"]
    input_file = os.path.join(self.current_dir, "example_regression.csv")
    loader = dc.data.CSVLoader(
        tasks=tasks, smiles_field="smiles", featurizer=featurizer)
    dataset = loader.featurize(input_file)

    regression_metric = dc.metrics.Metric(
        dc.metrics.pearson_r2_score, task_averager=np.mean)

    n_atom_feat = 75
    n_pair_feat = 14
    n_feat = 128
    batch_size = 10
    max_atoms = 50

    graph = dc.nn.AlternateSequentialWeaveGraph(
        batch_size,
        max_atoms=max_atoms,
        n_atom_feat=n_atom_feat,
        n_pair_feat=n_pair_feat)
    graph.add(dc.nn.AlternateWeaveLayer(max_atoms, 75, 14))
    graph.add(dc.nn.AlternateWeaveLayer(max_atoms, 50, 50, update_pair=False))
    graph.add(dc.nn.Dense(n_feat, 50, activation='tanh'))
    graph.add(dc.nn.BatchNormalization(epsilon=1e-5, mode=1))
    graph.add(
        dc.nn.AlternateWeaveGather(
            batch_size, n_input=n_feat, gaussian_expand=True))

    model = dc.models.MultitaskGraphRegressor(
        graph,
        n_tasks,
        n_feat,
        batch_size=batch_size,
        learning_rate=1e-3,
        learning_rate_decay_time=1000,
        optimizer_type="adam",
        beta1=.9,
        beta2=.999)

    # Fit trained model
    model.fit(dataset, nb_epoch=40)
    model.save()

    # Eval model on train
    scores = model.evaluate(dataset, [regression_metric])

    assert scores[regression_metric.name] > .9

  def test_tensorgraph_weave_singletask_regression_overfit(self):
    """Test weave model overfits tiny data."""
    np.random.seed(123)
    tf.set_random_seed(123)
    n_tasks = 1

    # Load mini log-solubility dataset.
    featurizer = dc.feat.WeaveFeaturizer()
    tasks = ["outcome"]
    input_file = os.path.join(self.current_dir, "example_regression.csv")
    loader = dc.data.CSVLoader(
        tasks=tasks, smiles_field="smiles", featurizer=featurizer)
    dataset = loader.featurize(input_file)

    regression_metric = dc.metrics.Metric(
        dc.metrics.pearson_r2_score, task_averager=np.mean)

    n_atom_feat = 75
    n_pair_feat = 14
    n_feat = 128
    batch_size = 10

    model = dc.models.WeaveTensorGraph(
        n_tasks,
        n_atom_feat=n_atom_feat,
        n_pair_feat=n_pair_feat,
        n_graph_feat=n_feat,
        batch_size=batch_size,
        learning_rate=0.001,
        use_queue=False,
        mode="regression")

    # Fit trained model
    model.fit(dataset, nb_epoch=120)

    # Eval model on train
    scores = model.evaluate(dataset, [regression_metric])

    assert scores[regression_metric.name] > .8

  @attr("slow")
  def test_MPNN_singletask_regression_overfit(self):
    """Test MPNN overfits tiny data."""
    np.random.seed(123)
    tf.set_random_seed(123)
    n_tasks = 1

    # Load mini log-solubility dataset.
    featurizer = dc.feat.WeaveFeaturizer()
    tasks = ["outcome"]
    input_file = os.path.join(self.current_dir, "example_regression.csv")
    loader = dc.data.CSVLoader(
        tasks=tasks, smiles_field="smiles", featurizer=featurizer)
    dataset = loader.featurize(input_file)

    regression_metric = dc.metrics.Metric(
        dc.metrics.pearson_r2_score, task_averager=np.mean)

    n_atom_feat = 75
    n_pair_feat = 14
    batch_size = 10
    model = dc.models.MPNNTensorGraph(
        n_tasks,
        n_atom_feat=n_atom_feat,
        n_pair_feat=n_pair_feat,
        T=2,
        M=3,
        batch_size=batch_size,
        learning_rate=0.001,
        use_queue=False,
        mode="regression")

    # Fit trained model
    model.fit(dataset, nb_epoch=50)

    # Eval model on train
    scores = model.evaluate(dataset, [regression_metric])

    assert scores[regression_metric.name] > .8

  def test_textCNN_singletask_classification_overfit(self):
    """Test textCNN model overfits tiny data."""
    np.random.seed(123)
    tf.set_random_seed(123)
    n_tasks = 1

    featurizer = dc.feat.RawFeaturizer()
    tasks = ["outcome"]
    input_file = os.path.join(self.current_dir, "example_classification.csv")
    loader = dc.data.CSVLoader(
        tasks=tasks, smiles_field="smiles", featurizer=featurizer)
    dataset = loader.featurize(input_file)

    classification_metric = dc.metrics.Metric(dc.metrics.accuracy_score)

    char_dict, length = dc.models.TextCNNTensorGraph.build_char_dict(dataset)
    batch_size = 10

    model = dc.models.TextCNNTensorGraph(
        n_tasks,
        char_dict,
        seq_length=length,
        batch_size=batch_size,
        learning_rate=0.001,
        use_queue=False,
        mode="classification")

    # Fit trained model
    model.fit(dataset, nb_epoch=200)

    # Eval model on train
    scores = model.evaluate(dataset, [classification_metric])

    assert scores[classification_metric.name] > .8

  def test_textCNN_singletask_regression_overfit(self):
    """Test textCNN model overfits tiny data."""
    np.random.seed(123)
    tf.set_random_seed(123)
    n_tasks = 1

    # Load mini log-solubility dataset.
    featurizer = dc.feat.RawFeaturizer()
    tasks = ["outcome"]
    input_file = os.path.join(self.current_dir, "example_regression.csv")
    loader = dc.data.CSVLoader(
        tasks=tasks, smiles_field="smiles", featurizer=featurizer)
    dataset = loader.featurize(input_file)

    regression_metric = dc.metrics.Metric(
        dc.metrics.pearson_r2_score, task_averager=np.mean)

    char_dict, length = dc.models.TextCNNTensorGraph.build_char_dict(dataset)
    batch_size = 10

    model = dc.models.TextCNNTensorGraph(
        n_tasks,
        char_dict,
        seq_length=length,
        batch_size=batch_size,
        learning_rate=0.001,
        use_queue=False,
        mode="regression")

    # Fit trained model
    model.fit(dataset, nb_epoch=200)

    # Eval model on train
    scores = model.evaluate(dataset, [regression_metric])

    assert scores[regression_metric.name] > .9

  def test_siamese_singletask_classification_overfit(self):
    """Test siamese singletask model overfits tiny data."""
    np.random.seed(123)
    tf.set_random_seed(123)
    n_tasks = 1
    n_feat = 75
    max_depth = 4
    n_pos = 6
    n_neg = 4
    test_batch_size = 10
    n_train_trials = 80
    support_batch_size = n_pos + n_neg

    # Load mini log-solubility dataset.
    featurizer = dc.feat.ConvMolFeaturizer()
    tasks = ["outcome"]
    input_file = os.path.join(self.current_dir, "example_classification.csv")
    loader = dc.data.CSVLoader(
        tasks=tasks, smiles_field="smiles", featurizer=featurizer)
    dataset = loader.featurize(input_file)

    classification_metric = dc.metrics.Metric(dc.metrics.accuracy_score)

    support_model = dc.nn.SequentialSupportGraph(n_feat)

    # Add layers
    # output will be (n_atoms, 64)
    support_model.add(dc.nn.GraphConv(64, n_feat, activation='relu'))
    # Need to add batch-norm separately to test/support due to differing
    # shapes.
    # output will be (n_atoms, 64)
    support_model.add_test(dc.nn.BatchNormalization(epsilon=1e-5, mode=1))
    # output will be (n_atoms, 64)
    support_model.add_support(dc.nn.BatchNormalization(epsilon=1e-5, mode=1))
    support_model.add(dc.nn.GraphPool())
    support_model.add_test(dc.nn.GraphGather(test_batch_size))
    support_model.add_support(dc.nn.GraphGather(support_batch_size))

    model = dc.models.SupportGraphClassifier(
        support_model,
        test_batch_size=test_batch_size,
        support_batch_size=support_batch_size,
        learning_rate=1e-3)

    # Fit trained model. Dataset has 6 positives and 4 negatives, so set
    # n_pos/n_neg accordingly.
    model.fit(
        dataset, n_episodes_per_epoch=n_train_trials, n_pos=n_pos, n_neg=n_neg)
    model.save()

    # Eval model on train. Dataset has 6 positives and 4 negatives, so set
    # n_pos/n_neg accordingly. Note that support is *not* excluded (so we
    # can measure model has memorized support).  Replacement is turned off to
    # ensure that support contains full training set. This checks that the
    # model has mastered memorization of provided support.
    scores, _ = model.evaluate(
        dataset,
        classification_metric,
        n_trials=5,
        n_pos=n_pos,
        n_neg=n_neg,
        exclude_support=False)

    ##################################################### DEBUG
    # TODO(rbharath): Check if something went wrong here...
    # Measure performance on 0-th task.
    #assert scores[0] > .9
    assert scores[0] > .75
    ##################################################### DEBUG

  def test_attn_lstm_singletask_classification_overfit(self):
    """Test attn lstm singletask overfits tiny data."""
    np.random.seed(123)
    tf.set_random_seed(123)
    n_tasks = 1
    n_feat = 75
    max_depth = 4
    n_pos = 6
    n_neg = 4
    test_batch_size = 10
    support_batch_size = n_pos + n_neg
    n_train_trials = 80

    # Load mini log-solubility dataset.
    featurizer = dc.feat.ConvMolFeaturizer()
    tasks = ["outcome"]
    input_file = os.path.join(self.current_dir, "example_classification.csv")
    loader = dc.data.CSVLoader(
        tasks=tasks, smiles_field="smiles", featurizer=featurizer)
    dataset = loader.featurize(input_file)
    classification_metric = dc.metrics.Metric(dc.metrics.accuracy_score)

    support_model = dc.nn.SequentialSupportGraph(n_feat)

    # Add layers
    # output will be (n_atoms, 64)
    support_model.add(dc.nn.GraphConv(64, n_feat, activation='relu'))
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
    support_model.join(
        dc.nn.AttnLSTMEmbedding(test_batch_size, support_batch_size, 64,
                                max_depth))

    model = dc.models.SupportGraphClassifier(
        support_model,
        test_batch_size=test_batch_size,
        support_batch_size=support_batch_size,
        learning_rate=1e-3)

    # Fit trained model. Dataset has 6 positives and 4 negatives, so set
    # n_pos/n_neg accordingly.
    model.fit(
        dataset, n_episodes_per_epoch=n_train_trials, n_pos=n_pos, n_neg=n_neg)
    model.save()

    # Eval model on train. Dataset has 6 positives and 4 negatives, so set
    # n_pos/n_neg accordingly. Note that support is *not* excluded (so we
    # can measure model has memorized support).  Replacement is turned off to
    # ensure that support contains full training set. This checks that the
    # model has mastered memorization of provided support.
    scores, _ = model.evaluate(
        dataset,
        classification_metric,
        n_trials=5,
        n_pos=n_pos,
        n_neg=n_neg,
        exclude_support=False)

    # Measure performance on 0-th task.
    ##################################################### DEBUG
    # TODO(rbharath): Check if something went wrong here...
    # Measure performance on 0-th task.
    #assert scores[0] > .85
    assert scores[0] > .79
    ##################################################### DEBUG

  def test_residual_lstm_singletask_classification_overfit(self):
    """Test resi-lstm multitask overfits tiny data."""
    n_tasks = 1
    n_feat = 75
    max_depth = 4
    n_pos = 6
    n_neg = 4
    test_batch_size = 10
    support_batch_size = n_pos + n_neg
    n_train_trials = 80

    # Load mini log-solubility dataset.
    featurizer = dc.feat.ConvMolFeaturizer()
    tasks = ["outcome"]
    input_file = os.path.join(self.current_dir, "example_classification.csv")
    loader = dc.data.CSVLoader(
        tasks=tasks, smiles_field="smiles", featurizer=featurizer)
    dataset = loader.featurize(input_file)

    classification_metric = dc.metrics.Metric(dc.metrics.accuracy_score)

    support_model = dc.nn.SequentialSupportGraph(n_feat)

    # Add layers
    # output will be (n_atoms, 64)
    support_model.add(dc.nn.GraphConv(64, n_feat, activation='relu'))
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
    support_model.join(
        dc.nn.ResiLSTMEmbedding(test_batch_size, support_batch_size, 64,
                                max_depth))

    model = dc.models.SupportGraphClassifier(
        support_model,
        test_batch_size=test_batch_size,
        support_batch_size=support_batch_size,
        learning_rate=1e-3)

    # Fit trained model. Dataset has 6 positives and 4 negatives, so set
    # n_pos/n_neg accordingly.

    model.fit(
        dataset, n_episodes_per_epoch=n_train_trials, n_pos=n_pos, n_neg=n_neg)
    model.save()

    # Eval model on train. Dataset has 6 positives and 4 negatives, so set
    # n_pos/n_neg accordingly. Note that support is *not* excluded (so we
    # can measure model has memorized support).  Replacement is turned off to
    # ensure that support contains full training set. This checks that the
    # model has mastered memorization of provided support.
    scores, _ = model.evaluate(
        dataset,
        classification_metric,
        n_trials=5,
        n_pos=n_pos,
        n_neg=n_neg,
        exclude_support=False)

    # Measure performance on 0-th task.
    ##################################################### DEBUG
    # TODO(rbharath): Check if something went wrong here...
    # Measure performance on 0-th task.
    #assert scores[0] > .9
    assert scores[0] > .65
    ##################################################### DEBUG

  def test_tf_progressive_regression_overfit(self):
    """Test tf progressive multitask overfits tiny data."""
    np.random.seed(123)
    n_tasks = 9
    n_samples = 10
    n_features = 3
    n_classes = 2

    # Generate dummy dataset
    np.random.seed(123)
    ids = np.arange(n_samples)
    X = np.random.rand(n_samples, n_features)
    y = np.ones((n_samples, n_tasks))
    w = np.ones((n_samples, n_tasks))

    dataset = dc.data.NumpyDataset(X, y, w, ids)

    metric = dc.metrics.Metric(dc.metrics.rms_score, task_averager=np.mean)
    model = dc.models.ProgressiveMultitaskRegressor(
        n_tasks,
        n_features,
        layer_sizes=[50],
        bypass_layer_sizes=[10],
        dropouts=[0.],
        learning_rate=0.003,
        weight_init_stddevs=[.1],
        seed=123,
        alpha_init_stddevs=[.02],
        batch_size=n_samples)

    # Fit trained model
    model.fit(dataset, nb_epoch=20)
    model.save()

    # Eval model on train
    scores = model.evaluate(dataset, [metric])
    y_pred = model.predict(dataset)
    assert scores[metric.name] < .2
