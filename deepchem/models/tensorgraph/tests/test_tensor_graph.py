import unittest

import numpy as np
import os
from nose.tools import assert_true, nottest
from flaky import flaky
import tensorflow as tf

import deepchem as dc
from deepchem.data import NumpyDataset
from deepchem.data.datasets import Databag
from deepchem.models.tensorgraph.layers import Dense, SoftMaxCrossEntropy, ReduceMean, SoftMax, Constant, Variable
from deepchem.models.tensorgraph.layers import Feature, Label
from deepchem.models.tensorgraph.layers import ReduceSquareDifference, Add
from deepchem.models.tensorgraph.tensor_graph import TensorGraph
from deepchem.models.tensorgraph.optimizers import GradientDescent, ExponentialDecay


class TestTensorGraph(unittest.TestCase):
  """
  Test that graph topologies work correctly.
  """

  def test_single_task_classifier(self):
    n_data_points = 20
    n_features = 2
    X = np.random.rand(n_data_points, n_features)
    y = [[0, 1] for x in range(n_data_points)]
    dataset = NumpyDataset(X, y)
    features = Feature(shape=(None, n_features))
    dense = Dense(out_channels=2, in_layers=[features])
    output = SoftMax(in_layers=[dense])
    label = Label(shape=(None, 2))
    smce = SoftMaxCrossEntropy(in_layers=[label, dense])
    loss = ReduceMean(in_layers=[smce])
    tg = dc.models.TensorGraph(learning_rate=0.01)
    tg.add_output(output)
    tg.set_loss(loss)
    tg.fit(dataset, nb_epoch=1000)
    prediction = np.squeeze(tg.predict_on_batch(X))
    assert_true(np.all(np.isclose(prediction, y, atol=0.4)))

  @flaky
  def test_multi_task_classifier(self):
    n_data_points = 20
    n_features = 2

    X = np.random.rand(n_data_points, n_features)
    y1 = np.array([[0, 1] for x in range(n_data_points)])
    y2 = np.array([[1, 0] for x in range(n_data_points)])
    X = NumpyDataset(X)
    ys = [NumpyDataset(y1), NumpyDataset(y2)]

    databag = Databag()

    features = Feature(shape=(None, n_features))
    databag.add_dataset(features, X)

    outputs = []
    entropies = []
    for i in range(2):
      label = Label(shape=(None, 2))
      dense = Dense(out_channels=2, in_layers=[features])
      output = SoftMax(in_layers=[dense])
      smce = SoftMaxCrossEntropy(in_layers=[label, dense])

      entropies.append(smce)
      outputs.append(output)
      databag.add_dataset(label, ys[i])

    total_loss = ReduceMean(in_layers=entropies)

    tg = dc.models.TensorGraph(learning_rate=0.01)
    for output in outputs:
      tg.add_output(output)
    tg.set_loss(total_loss)

    tg.fit_generator(
        databag.iterbatches(
            epochs=1000, batch_size=tg.batch_size, pad_batches=True))
    predictions = tg.predict_on_generator(databag.iterbatches())
    for i in range(2):
      y_real = ys[i].X
      y_pred = predictions[i]
      assert_true(np.all(np.isclose(y_pred, y_real, atol=0.6)))

  def test_single_task_regressor(self):
    n_data_points = 20
    n_features = 2
    X = np.random.rand(n_data_points, n_features)
    y = [0.5 for x in range(n_data_points)]
    dataset = NumpyDataset(X, y)
    features = Feature(shape=(None, n_features))
    dense = Dense(out_channels=1, in_layers=[features])
    label = Label(shape=(None, 1))
    loss = ReduceSquareDifference(in_layers=[dense, label])
    tg = dc.models.TensorGraph(learning_rate=0.01)
    tg.add_output(dense)
    tg.set_loss(loss)
    tg.fit(dataset, nb_epoch=1000)
    prediction = np.squeeze(tg.predict_on_batch(X))
    assert_true(np.all(np.isclose(prediction, y, atol=3.0)))

  def test_multi_task_regressor(self):
    n_data_points = 20
    n_features = 2

    X = np.random.rand(n_data_points, n_features)
    y1 = np.expand_dims(np.array([0.5 for x in range(n_data_points)]), axis=-1)
    y2 = np.expand_dims(np.array([-0.5 for x in range(n_data_points)]), axis=-1)
    X = NumpyDataset(X)
    ys = [NumpyDataset(y1), NumpyDataset(y2)]

    databag = Databag()

    features = Feature(shape=(None, n_features))
    databag.add_dataset(features, X)

    outputs = []
    losses = []
    for i in range(2):
      label = Label(shape=(None, 1))
      dense = Dense(out_channels=1, in_layers=[features])
      loss = ReduceSquareDifference(in_layers=[dense, label])

      outputs.append(dense)
      losses.append(loss)
      databag.add_dataset(label, ys[i])

    total_loss = ReduceMean(in_layers=losses)

    tg = dc.models.TensorGraph(learning_rate=0.01)
    for output in outputs:
      tg.add_output(output)
    tg.set_loss(total_loss)

    tg.fit_generator(
        databag.iterbatches(
            epochs=1000, batch_size=tg.batch_size, pad_batches=True))
    predictions = tg.predict_on_generator(databag.iterbatches())
    for i in range(2):
      y_real = ys[i].X
      y_pred = predictions[i]
      assert_true(np.all(np.isclose(y_pred, y_real, atol=1.5)))

  @flaky
  def test_no_queue(self):
    n_data_points = 20
    n_features = 2
    X = np.random.rand(n_data_points, n_features)
    y = [[0, 1] for x in range(n_data_points)]
    dataset = NumpyDataset(X, y)
    features = Feature(shape=(None, n_features))
    dense = Dense(out_channels=2, in_layers=[features])
    output = SoftMax(in_layers=[dense])
    label = Label(shape=(None, 2))
    smce = SoftMaxCrossEntropy(in_layers=[label, dense])
    loss = ReduceMean(in_layers=[smce])
    tg = dc.models.TensorGraph(learning_rate=0.01, use_queue=False)
    tg.add_output(output)
    tg.set_loss(loss)
    tg.fit(dataset, nb_epoch=1000)
    prediction = np.squeeze(tg.predict_on_batch(X))
    assert_true(np.all(np.isclose(prediction, y, atol=0.4)))

  @flaky
  def test_set_optimizer(self):
    n_data_points = 20
    n_features = 2
    X = np.random.rand(n_data_points, n_features)
    y = [[0, 1] for x in range(n_data_points)]
    dataset = NumpyDataset(X, y)
    features = Feature(shape=(None, n_features))
    dense = Dense(out_channels=2, in_layers=[features])
    output = SoftMax(in_layers=[dense])
    label = Label(shape=(None, 2))
    smce = SoftMaxCrossEntropy(in_layers=[label, dense])
    loss = ReduceMean(in_layers=[smce])
    tg = dc.models.TensorGraph(learning_rate=0.01, use_queue=False)
    tg.add_output(output)
    tg.set_loss(loss)
    global_step = tg.get_global_step()
    learning_rate = ExponentialDecay(
        initial_rate=0.1, decay_rate=0.96, decay_steps=100000)
    tg.set_optimizer(GradientDescent(learning_rate=learning_rate))
    tg.fit(dataset, nb_epoch=1000)
    prediction = np.squeeze(tg.predict_on_batch(X))
    tg.save()

    tg1 = TensorGraph.load_from_dir(tg.model_dir)
    prediction2 = np.squeeze(tg1.predict_on_batch(X))
    assert_true(np.all(np.isclose(prediction, prediction2, atol=0.01)))

  @nottest
  def test_tensorboard(self):
    n_data_points = 20
    n_features = 2
    X = np.random.rand(n_data_points, n_features)
    y = [[0, 1] for x in range(n_data_points)]
    dataset = NumpyDataset(X, y)
    features = Feature(shape=(None, n_features))
    dense = Dense(out_channels=2, in_layers=[features])
    output = SoftMax(in_layers=[dense])
    label = Label(shape=(None, 2))
    smce = SoftMaxCrossEntropy(in_layers=[label, dense])
    loss = ReduceMean(in_layers=[smce])
    tg = dc.models.TensorGraph(
        tensorboard=True,
        tensorboard_log_frequency=1,
        learning_rate=0.01,
        model_dir='/tmp/tensorgraph')
    tg.add_output(output)
    tg.set_loss(loss)
    tg.fit(dataset, nb_epoch=1000)
    files_in_dir = os.listdir(tg.model_dir)
    event_file = list(filter(lambda x: x.startswith("events"), files_in_dir))
    assert_true(len(event_file) > 0)
    event_file = os.path.join(tg.model_dir, event_file[0])
    file_size = os.stat(event_file).st_size
    assert_true(file_size > 0)

  def test_save_load(self):
    n_data_points = 20
    n_features = 2
    X = np.random.rand(n_data_points, n_features)
    y = [[0, 1] for x in range(n_data_points)]
    dataset = NumpyDataset(X, y)
    features = Feature(shape=(None, n_features))
    dense = Dense(out_channels=2, in_layers=[features])
    output = SoftMax(in_layers=[dense])
    label = Label(shape=(None, 2))
    smce = SoftMaxCrossEntropy(in_layers=[label, dense])
    loss = ReduceMean(in_layers=[smce])
    tg = dc.models.TensorGraph(learning_rate=0.01)
    tg.add_output(output)
    tg.set_loss(loss)
    tg.fit(dataset, nb_epoch=1)
    prediction = np.squeeze(tg.predict_on_batch(X))
    tg.save()

    tg1 = TensorGraph.load_from_dir(tg.model_dir)
    prediction2 = np.squeeze(tg1.predict_on_batch(X))
    assert_true(np.all(np.isclose(prediction, prediction2, atol=0.01)))

  def test_shared_layer(self):
    n_data_points = 20
    n_features = 2

    X = np.random.rand(n_data_points, n_features)
    y1 = np.array([[0, 1] for x in range(n_data_points)])
    X = NumpyDataset(X)
    ys = [NumpyDataset(y1)]

    databag = Databag()

    features = Feature(shape=(None, n_features))
    databag.add_dataset(features, X)

    outputs = []

    label = Label(shape=(None, 2))
    dense1 = Dense(out_channels=2, in_layers=[features])
    dense2 = dense1.shared(in_layers=[features])
    output1 = SoftMax(in_layers=[dense1])
    output2 = SoftMax(in_layers=[dense2])
    smce = SoftMaxCrossEntropy(in_layers=[label, dense1])

    outputs.append(output1)
    outputs.append(output2)
    databag.add_dataset(label, ys[0])

    total_loss = ReduceMean(in_layers=[smce])

    tg = dc.models.TensorGraph(learning_rate=0.01)
    for output in outputs:
      tg.add_output(output)
    tg.set_loss(total_loss)

    tg.fit_generator(
        databag.iterbatches(
            epochs=1, batch_size=tg.batch_size, pad_batches=True))
    prediction = tg.predict_on_generator(databag.iterbatches())
    assert_true(np.all(np.isclose(prediction[0], prediction[1], atol=0.01)))

  def test_operators(self):
    """Test math operators on Layers."""
    v1 = np.random.uniform(size=(2, 3)).astype(np.float32)
    v2 = np.random.uniform(size=(2, 3)).astype(np.float32)
    c1 = Constant(v1)
    c2 = Constant(v2)
    tg = dc.models.TensorGraph()
    tg.set_loss(c1)
    expected = []
    tg.add_output(c1 + c2)
    expected.append(v1 + v2)
    tg.add_output(c1 + v2)
    expected.append(v1 + v2)
    tg.add_output(1 + c2)
    expected.append(1 + v2)
    tg.add_output(c1 - c2)
    expected.append(v1 - v2)
    tg.add_output(c1 - v2)
    expected.append(v1 - v2)
    tg.add_output(1 - c2)
    expected.append(1 - v2)
    tg.add_output(c1 * c2)
    expected.append(v1 * v2)
    tg.add_output(c1 * v2)
    expected.append(v1 * v2)
    tg.add_output(2 * c2)
    expected.append(2 * v2)
    tg.add_output(-c1)
    expected.append(-v1)
    tg.add_output(c1 / c2)
    expected.append(v1 / v2)
    tg.add_output(c1 / 2)
    expected.append(v1 / 2)
    for o, e in zip(tg.outputs, expected):
      value = tg.predict_on_batch(np.array([0]), outputs=o)
      assert np.array_equal(e, value)

  def test_initialize_variable(self):
    """Test methods for initializing a variable."""
    # Set by variable constructor.

    tg = dc.models.TensorGraph(use_queue=False)
    features = Feature(shape=(None, 1))
    tg.set_loss(Dense(1, in_layers=features))
    var = Variable([10.0])
    tg.add_output(var)
    assert tg.predict_on_batch(np.zeros((1, 1))) == [10.0]

    # Set by set_variable_initial_values().

    tg = dc.models.TensorGraph(use_queue=False)
    tg.set_loss(Dense(1, in_layers=features))
    var.set_variable_initial_values([[15.0]])
    tg.add_output(var)
    assert tg.predict_on_batch(np.zeros((1, 1))) == [15.0]

  def test_copy_layers(self):
    """Test copying layers."""
    tg = dc.models.TensorGraph()
    features = Feature(shape=(None, 10))
    dense = Dense(
        10, in_layers=features, biases_initializer=tf.random_normal_initializer)
    constant = Constant(10.0)
    output = dense + constant
    tg.add_output(output)
    tg.set_loss(output)
    tg.fit_generator([])
    replacements = {constant: Constant(20.0)}
    copy = output.copy(replacements, tg)
    assert isinstance(copy, Add)
    assert isinstance(copy.in_layers[0], Dense)
    assert isinstance(copy.in_layers[0].in_layers[0], Feature)
    assert copy.in_layers[1] == replacements[constant]
    variables = tg.get_layer_variables(dense)
    with tg._get_tf("Graph").as_default():
      values = tg.session.run(variables)
    for v1, v2 in zip(values, copy.in_layers[0].variable_values):
      assert np.array_equal(v1, v2)

  def test_copy_layers_shared(self):
    """Test copying layers with shared variables."""
    tg = dc.models.TensorGraph()
    features = Feature(shape=(None, 10))
    dense = Dense(
        10, in_layers=features, biases_initializer=tf.random_normal_initializer)
    constant = Constant(10.0)
    output = dense + constant
    tg.add_output(output)
    tg.set_loss(output)
    replacements = {features: features, constant: Constant(20.0)}
    copy = output.copy(replacements, shared=True)
    tg.add_output(copy)
    assert isinstance(copy, Add)
    assert isinstance(copy.in_layers[0], Dense)
    assert isinstance(copy.in_layers[0].in_layers[0], Feature)
    assert copy.in_layers[1] == replacements[constant]
    variables1 = tg.get_layer_variables(dense)
    variables2 = tg.get_layer_variables(copy.in_layers[0])
    for v1, v2, in zip(variables1, variables2):
      assert v1 == v2
    feed_dict = {features: np.random.random((5, 10))}
    v1, v2 = tg.predict_on_generator([feed_dict], outputs=[output, copy])
    assert_true(np.all(np.isclose(v1 + 10, v2)))

  def test_submodels(self):
    """Test optimizing submodels."""
    tg = dc.models.TensorGraph(learning_rate=0.1, batch_size=1)
    features = Feature(shape=(None, 1))
    var1 = Variable([2.0])
    var2 = Variable([2.0])
    tg.add_output(var1)
    tg.add_output(var2)
    loss = (var1 - 1) * (var1 - 1) + (var2 - 1) * (var2 - 1) + features
    tg.set_loss(loss)
    subloss1 = var1 * var1 + features
    subloss2 = var1 * var1 + var2 * var2 + features
    submodel1 = tg.create_submodel(loss=subloss1)
    submodel2 = tg.create_submodel(layers=[var2], loss=subloss2)
    data = np.zeros((1, 1))
    generator = [{features: data}] * 500

    # Optimize submodel 1.  This should send var1 to 0 while leaving var2 unchanged.

    tg.fit_generator(generator, submodel=submodel1)
    self.assertAlmostEqual(
        0.0, tg.predict_on_batch(data, outputs=var1)[0], places=4)
    self.assertAlmostEqual(
        2.0, tg.predict_on_batch(data, outputs=var2)[0], places=4)

    # Optimize the main loss.  This should send both variables toward 1.

    tg.fit_generator(generator)
    self.assertAlmostEqual(
        1.0, tg.predict_on_batch(data, outputs=var1)[0], places=4)
    self.assertAlmostEqual(
        1.0, tg.predict_on_batch(data, outputs=var2)[0], places=4)

    # Optimize submodel 2.  This should send var2 to 0 while leaving var1 unchanged.

    tg.fit_generator(generator, submodel=submodel2)
    self.assertAlmostEqual(
        1.0, tg.predict_on_batch(data, outputs=var1)[0], places=4)
    self.assertAlmostEqual(
        0.0, tg.predict_on_batch(data, outputs=var2)[0], places=4)
