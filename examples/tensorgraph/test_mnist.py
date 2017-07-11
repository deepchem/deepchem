import unittest
import numpy as np
import tensorflow as tf
from nose.tools import assert_true
from sklearn.metrics import roc_curve, auc

import deepchem as dc
from deepchem.data import NumpyDataset

from deepchem.data.datasets import Databag
from deepchem.models.tensorgraph.layers import Dense, SoftMaxCrossEntropy, ReduceMean, SoftMax, ReduceSquareDifference
from deepchem.models.tensorgraph.layers import Reshape, Flatten, Feature, Conv2D, MaxPool, Label


class TestTensorGraphMNIST(unittest.TestCase):

  def test_mnist(self):
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    train = dc.data.NumpyDataset(mnist.train.images, mnist.train.labels)
    valid = dc.data.NumpyDataset(mnist.validation.images,
                                 mnist.validation.labels)

    # Images are square 28x28 (batch, height, width, channel)
    feature = Feature(shape=(None, 784), name="Feature")
    make_image = Reshape(shape=(-1, 28, 28, 1), in_layers=[feature])

    conv2d_1 = Conv2D(
        num_outputs=32,
        normalizer_fn=tf.contrib.layers.batch_norm,
        in_layers=[make_image])
    maxpool_1 = MaxPool(in_layers=[conv2d_1])

    conv2d_2 = Conv2D(
        num_outputs=64,
        normalizer_fn=tf.contrib.layers.batch_norm,
        in_layers=[maxpool_1])
    maxpool_2 = MaxPool(in_layers=[conv2d_2])
    flatten = Flatten(in_layers=[maxpool_2])

    dense1 = Dense(
        out_channels=1024, activation_fn=tf.nn.relu, in_layers=[flatten])
    dense2 = Dense(out_channels=10, in_layers=[dense1])
    label = Label(shape=(None, 10), name="Label")
    smce = SoftMaxCrossEntropy(in_layers=[label, dense2])
    loss = ReduceMean(in_layers=[smce])
    output = SoftMax(in_layers=[dense2])

    tg = dc.models.TensorGraph(
        model_dir='/tmp/mnist', batch_size=1000, use_queue=True)
    tg.add_output(output)
    tg.set_loss(loss)
    tg.fit(train, nb_epoch=2)

    prediction = np.squeeze(tg.predict_proba_on_batch(valid.X))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(10):
      fpr[i], tpr[i], thresh = roc_curve(valid.y[:, i], prediction[:, i])
      roc_auc[i] = auc(fpr[i], tpr[i])
      assert_true(roc_auc[i] > 0.99)

  def test_compute_model_performance_singletask_regressor_ordering(self):
    n_data_points = 1000
    n_features = 1

    X = np.array(range(n_data_points))
    X = np.expand_dims(X, axis=-1)
    y1 = X + 1
    X = NumpyDataset(X)
    ys = [NumpyDataset(y1)]

    databag = Databag()

    features = Feature(shape=(None, n_features))
    databag.add_dataset(features, X)

    outputs = []
    losses = []
    labels = []
    for i in range(1):
      label = Label(shape=(None, 1))
      dense = Dense(out_channels=1, in_layers=[features])
      loss = ReduceSquareDifference(in_layers=[dense, label])

      outputs.append(dense)
      losses.append(loss)
      labels.append(label)
      databag.add_dataset(label, ys[i])

    total_loss = ReduceMean(in_layers=losses)

    tg = dc.models.TensorGraph(mode="regression", learning_rate=0.1)
    for output in outputs:
      tg.add_output(output)
    tg.set_loss(total_loss)

    tg.fit_generator(
        databag.iterbatches(
            epochs=1000, batch_size=tg.batch_size, pad_batches=True))
    metric = [
        dc.metrics.Metric(
            dc.metrics.mean_absolute_error, np.mean, mode="regression"),
        dc.metrics.Metric(dc.metrics.pearson_r2_score, mode="regression")
    ]
    scores = tg.evaluate_generator(
        databag.iterbatches(batch_size=1),
        metric,
        labels=labels,
        per_task_metrics=True)
    print(scores)
    scores = list(scores[1].values())
    assert_true(np.all(np.isclose(scores, [0.0], atol=0.5)))
