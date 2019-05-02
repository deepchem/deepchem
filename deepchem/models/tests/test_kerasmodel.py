import os
import unittest
import deepchem as dc
import numpy as np
import tensorflow as tf
from tensorflow.python.eager import context


class TestKerasModel(unittest.TestCase):

  def test_overfit_graph_model(self):
    """Test fitting a KerasModel defined as a graph."""
    n_data_points = 10
    n_features = 2
    X = np.random.rand(n_data_points, n_features)
    y = (X[:, 0] > X[:, 1]).astype(np.float32)
    dataset = dc.data.NumpyDataset(X, y)
    inputs = tf.keras.Input(shape=(n_features,))
    hidden = tf.keras.layers.Dense(10, activation='relu')(inputs)
    logits = tf.keras.layers.Dense(1)(hidden)
    outputs = tf.keras.layers.Activation('sigmoid')(logits)
    keras_model = tf.keras.Model(inputs=inputs, outputs=[outputs, logits])
    model = dc.models.KerasModel(
        keras_model,
        dc.models.losses.SigmoidCrossEntropy(),
        output_types=['prediction', 'loss'],
        learning_rate=0.005)
    model.fit(dataset, nb_epoch=1000)
    prediction = np.squeeze(model.predict_on_batch(X))
    assert np.array_equal(y, np.round(prediction))
    metric = dc.metrics.Metric(dc.metrics.roc_auc_score)
    scores = model.evaluate(dataset, [metric])
    assert scores[metric.name] > 0.9

  def test_overfit_graph_model_eager(self):
    """Test fitting a KerasModel defined as a graph, in eager mode."""
    with context.eager_mode():
      self.test_overfit_graph_model()

  def test_overfit_sequential_model(self):
    """Test fitting a KerasModel defined as a sequential model."""
    n_data_points = 10
    n_features = 2
    X = np.random.rand(n_data_points, n_features)
    y = (X[:, 0] > X[:, 1]).astype(np.float32)
    dataset = dc.data.NumpyDataset(X, y)
    keras_model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model = dc.models.KerasModel(
        keras_model, dc.models.losses.BinaryCrossEntropy(), learning_rate=0.005)
    model.fit(dataset, nb_epoch=1000)
    prediction = np.squeeze(model.predict_on_batch(X))
    assert np.array_equal(y, np.round(prediction))
    metric = dc.metrics.Metric(dc.metrics.roc_auc_score)
    generator = model.default_generator(dataset)
    scores = model.evaluate_generator(generator, [metric])
    assert scores[metric.name] > 0.9

  def test_overfit_sequential_model_eager(self):
    """Test fitting a KerasModel defined as a sequential model, in eager mode."""
    with context.eager_mode():
      self.test_overfit_sequential_model()

  def test_checkpointing(self):
    """Test loading and saving checkpoints with KerasModel."""
    # Create two models using the same model directory.

    keras_model1 = tf.keras.Sequential([tf.keras.layers.Dense(10)])
    keras_model2 = tf.keras.Sequential([tf.keras.layers.Dense(10)])
    model1 = dc.models.KerasModel(keras_model1, dc.models.losses.L2Loss())
    model2 = dc.models.KerasModel(
        keras_model2, dc.models.losses.L2Loss(), model_dir=model1.model_dir)

    # Check that they produce different results.

    X = np.random.rand(5, 5)
    y1 = model1.predict_on_batch(X)
    y2 = model2.predict_on_batch(X)
    assert not np.array_equal(y1, y2)

    # Save a checkpoint from the first model and load it into the second one,
    # and make sure they now match.

    model1.save_checkpoint()
    model2.restore()
    y3 = model1.predict_on_batch(X)
    y4 = model2.predict_on_batch(X)
    assert np.array_equal(y1, y3)
    assert np.array_equal(y1, y4)

  def test_checkpointing_eager(self):
    """Test loading and saving checkpoints with KerasModel, in eager mode."""
    with context.eager_mode():
      self.test_checkpointing()

  def test_uncertainty(self):
    """Test estimating uncertainty a KerasModel."""
    n_samples = 30
    n_features = 1
    noise = 0.1
    X = np.random.rand(n_samples, n_features)
    y = (10 * X + np.random.normal(scale=noise, size=(n_samples, n_features)))
    dataset = dc.data.NumpyDataset(X, y)

    # Build a model that predicts uncertainty.

    inputs = tf.keras.Input(shape=(n_features,))
    hidden = tf.keras.layers.Dense(200, activation='relu')(inputs)
    dropout = tf.keras.layers.Dropout(rate=0.1)(hidden)
    output = tf.keras.layers.Dense(n_features)(dropout)
    log_var = tf.keras.layers.Dense(n_features)(dropout)
    var = tf.keras.layers.Activation(tf.exp)(log_var)
    keras_model = tf.keras.Model(
        inputs=inputs, outputs=[output, var, output, log_var])

    def loss(outputs, labels, weights):
      diff = labels[0] - outputs[0]
      log_var = outputs[1]
      var = tf.exp(log_var)
      return tf.reduce_mean(diff * diff / var + log_var)

    model = dc.models.KerasModel(
        keras_model,
        loss,
        output_types=['prediction', 'variance', 'loss', 'loss'],
        learning_rate=0.003)

    # Fit the model and see if its predictions are correct.

    model.fit(dataset, nb_epoch=2500)
    pred, std = model.predict_uncertainty(dataset)
    assert np.mean(np.abs(y - pred)) < 1.0
    assert noise < np.mean(std) < 1.0

  def test_uncertainty_eager(self):
    """Test estimating uncertainty a KerasModel, in eager mode."""
    with context.eager_mode():
      self.test_uncertainty()

  def test_saliency_mapping(self):
    """Test computing a saliency map."""
    n_tasks = 3
    n_features = 5
    keras_model = tf.keras.Sequential([
        tf.keras.layers.Dense(20, activation='tanh'),
        tf.keras.layers.Dense(n_tasks)
    ])
    model = dc.models.KerasModel(keras_model, dc.models.losses.L2Loss())
    x = np.random.random(n_features)
    s = model.compute_saliency(x)
    assert s.shape[0] == n_tasks
    assert s.shape[1] == n_features

    # Take a tiny step in the direction of s and see if the output changes by
    # the expected amount.

    delta = 0.01
    for task in range(n_tasks):
      norm = np.sqrt(np.sum(s[task]**2))
      step = 0.5 * delta / norm
      pred1 = model.predict_on_batch((x + s[task] * step).reshape(
          (1, n_features))).flatten()
      pred2 = model.predict_on_batch((x - s[task] * step).reshape(
          (1, n_features))).flatten()
      self.assertAlmostEqual(
          pred1[task], (pred2 + norm * delta)[task], places=4)

  def test_saliency_mapping_eager(self):
    """Test computing a saliency map, in eager mode."""
    with context.eager_mode():
      self.test_saliency_mapping()

  def test_saliency_shapes(self):
    """Test computing saliency maps for multiple outputs with multiple dimensions."""
    inputs = tf.keras.Input(shape=(2, 3))
    flatten = tf.keras.layers.Flatten()(inputs)
    output1 = tf.keras.layers.Reshape((4, 1))(tf.keras.layers.Dense(4)(flatten))
    output2 = tf.keras.layers.Reshape((1, 5))(tf.keras.layers.Dense(5)(flatten))
    keras_model = tf.keras.Model(inputs=inputs, outputs=[output1, output2])
    model = dc.models.KerasModel(keras_model, dc.models.losses.L2Loss())
    x = np.random.random((2, 3))
    s = model.compute_saliency(x)
    assert len(s) == 2
    assert s[0].shape == (4, 1, 2, 3)
    assert s[1].shape == (1, 5, 2, 3)

  def test_saliency_shapes_eager(self):
    """Test computing saliency maps for multiple outputs with multiple dimensions, in eager mode."""
    with context.eager_mode():
      self.test_saliency_shapes()

  def test_tensorboard(self):
    n_data_points = 20
    n_features = 2
    X = np.random.rand(n_data_points, n_features)
    y = [[0.0, 1.0] for x in range(n_data_points)]
    dataset = dc.data.NumpyDataset(X, y)
    keras_model = tf.keras.Sequential([
        tf.keras.layers.Dense(2, activation='softmax'),
    ])
    model = dc.models.KerasModel(
        keras_model,
        dc.models.losses.CategoricalCrossEntropy(),
        tensorboard=True,
        tensorboard_log_frequency=1)
    model.fit(dataset, nb_epoch=10)
    files_in_dir = os.listdir(model.model_dir)
    event_file = list(filter(lambda x: x.startswith("events"), files_in_dir))
    assert len(event_file) > 0
    event_file = os.path.join(model.model_dir, event_file[0])
    file_size = os.stat(event_file).st_size
    assert file_size > 0
