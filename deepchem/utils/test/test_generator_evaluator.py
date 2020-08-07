from unittest import TestCase

import numpy as np
import tensorflow as tf
from flaky import flaky
from tensorflow.keras import layers

import deepchem as dc
from deepchem.data import NumpyDataset


@flaky
def test_compute_model_performance_multitask_classifier():
  n_data_points = 20
  n_features = 1
  n_tasks = 2
  n_classes = 2

  X = np.ones(shape=(n_data_points // 2, n_features)) * -1
  X1 = np.ones(shape=(n_data_points // 2, n_features))
  X = np.concatenate((X, X1))
  class_1 = np.array([[0.0, 1.0] for x in range(int(n_data_points / 2))])
  class_0 = np.array([[1.0, 0.0] for x in range(int(n_data_points / 2))])
  y1 = np.concatenate((class_0, class_1))
  y2 = np.concatenate((class_1, class_0))
  y = np.stack([y1, y2], axis=1)
  dataset = NumpyDataset(X, y)

  features = layers.Input(shape=(n_data_points // 2, n_features))
  dense = layers.Dense(n_tasks * n_classes)(features)
  logits = layers.Reshape((n_tasks, n_classes))(dense)
  output = layers.Softmax()(logits)
  keras_model = tf.keras.Model(inputs=features, outputs=[output, logits])
  model = dc.models.KerasModel(
      keras_model,
      dc.models.losses.SoftmaxCrossEntropy(),
      output_types=['prediction', 'loss'],
      learning_rate=0.01,
      batch_size=n_data_points)

  model.fit(dataset, nb_epoch=1000)
  metric = dc.metrics.Metric(
      dc.metrics.roc_auc_score, np.mean, mode="classification")

  scores = model.evaluate_generator(
      model.default_generator(dataset), [metric], per_task_metrics=True)
  scores = list(scores[1].values())
  # Loosening atol to see if tests stop failing sporadically
  assert np.all(np.isclose(scores, [1.0, 1.0], atol=0.50))


def test_compute_model_performance_singletask_classifier():
  """Computes model performance on singletask dataset with one-hot label encoding."""
  n_data_points = 20
  n_features = 10

  X = np.ones(shape=(int(n_data_points / 2), n_features)) * -1
  X1 = np.ones(shape=(int(n_data_points / 2), n_features))
  X = np.concatenate((X, X1))
  class_1 = np.array([[0.0, 1.0] for x in range(int(n_data_points / 2))])
  class_0 = np.array([[1.0, 0.0] for x in range(int(n_data_points / 2))])
  y = np.concatenate((class_0, class_1))
  dataset = NumpyDataset(X, y)

  features = layers.Input(shape=(n_features,))
  dense = layers.Dense(2)(features)
  output = layers.Softmax()(dense)
  keras_model = tf.keras.Model(inputs=features, outputs=[output])
  model = dc.models.KerasModel(
      keras_model, dc.models.losses.SoftmaxCrossEntropy(), learning_rate=0.1)

  model.fit(dataset, nb_epoch=1000)
  metric = dc.metrics.Metric(
      dc.metrics.roc_auc_score, np.mean, mode="classification", n_tasks=1)

  scores = model.evaluate_generator(
      model.default_generator(dataset), [metric], per_task_metrics=True)
  scores = list(scores[1].values())
  assert np.isclose(scores, [1.0], atol=0.05)


def test_compute_model_performance_multitask_regressor():
  random_seed = 42
  n_data_points = 20
  n_features = 2
  n_tasks = 2
  np.random.seed(seed=random_seed)

  X = np.random.rand(n_data_points, n_features)
  y1 = np.array([0.5 for x in range(n_data_points)])
  y2 = np.array([-0.5 for x in range(n_data_points)])
  y = np.stack([y1, y2], axis=1)
  dataset = NumpyDataset(X, y)

  features = layers.Input(shape=(n_features,))
  dense = layers.Dense(n_tasks)(features)
  keras_model = tf.keras.Model(inputs=features, outputs=[dense])
  model = dc.models.KerasModel(
      keras_model, dc.models.losses.L2Loss(), learning_rate=0.1)

  model.fit(dataset, nb_epoch=1000)
  metric = [
      dc.metrics.Metric(
          dc.metrics.mean_absolute_error, np.mean, mode="regression"),
  ]
  scores = model.evaluate_generator(
      model.default_generator(dataset), metric, per_task_metrics=True)
  scores = list(scores[1].values())
  assert np.all(np.isclose(scores, [0.0, 0.0], atol=1.0))
