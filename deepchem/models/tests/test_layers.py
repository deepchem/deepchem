import os
import unittest
import deepchem as dc
import numpy as np
import tensorflow as tf
from tensorflow.python.eager import context


class TestLayer(unittest.TestCase):

  def test_graph_conv(self):
    out_channel = 10
    min_deg = 0,
    max_deg = 10,
    activation_fn = tf.nn.relu
    graph_conv = dc.models.layers.GraphConv(
        out_channel=out_channel,
        min_deg=min_deg,
        max_deg=max_deg,
        activation_fn=activation_fn)

    config = graph_conv.get_config()
    graph_conv_new = dc.models.layers.GraphConv.from_config(config)

    assert graph_conv_new.out_channel == graph_conv.out_channel
    assert graph_conv_new.activation_fn == graph_conv.activation_fn
    assert graph_conv_new.max_degree == graph_conv.max_degree
    assert graph_conv_new.min_degree == graph_conv.min_degree

  def test_graph_gather(self):
    batch_size = 10
    activation_fn = tf.nn.relu
    graph_gather = dc.models.layers.GraphGather(
        batch_size=batch_size, activation_fn=activation_fn)

    config = graph_gather.get_config()
    graph_gather_new = dc.models.layers.GraphGather.from_config(config)

    assert graph_gather_new.batch_size == graph_gather.batch_size
    assert graph_gather_new.activation_fn == graph_gather.activation_fn

  def test_graph_pool(self):
    min_degree = 0
    max_degree = 10
    graph_pool = dc.models.layers.GraphPool(
        min_degree=min_degree, max_degree=max_degree)

    config = graph_pool.get_config()
    graph_pool_new = dc.models.layers.GraphPool.from_config(config)

    assert graph_pool_new.max_degree == graph_pool.max_degree
    assert graph_pool_new.min_degree == graph_pool.min_degree
