"""
Contains implementations of layers used in ChemCeption and Smiles2Vec models.
"""

from __future__ import division
from __future__ import unicode_literals

__author__ = "Vignesh Ram Somnath"
__license__ = "MIT"

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Concatenate, ReLU, Add, MaxPool2D


class Stem(tf.keras.layers.Layer):

  def __init__(self, num_filters, **kwargs):
    """
    Parameters
    ----------
    num_filters: int,
        Number of convolutional filters
    """
    self.num_filters = num_filters
    self._build_layer_components()
    super(Stem, self).__init__(**kwargs)

  def _build_layer_components(self):
    """Builds the layers components and set _layers attribute."""
    self.conv_layer = Conv2D(
        filters=self.num_filters,
        kernel_size=4,
        strides=2,
        activation=tf.nn.relu)

    self.activation_layer = ReLU()
    self._layers = [self.conv_layer, self.activation_layer]

  def call(self, inputs):
    conv1 = self.conv_layer(inputs)
    return self.activation_layer(conv1)


class InceptionResnetA(tf.keras.layers.Layer):

  def __init__(self, num_filters, input_dim, **kwargs):
    """
    Parameters
    ----------
    num_filters: int,
        Number of convolutional filters
    input_dim: int,
        Number of channels in the input.
    """
    self.num_filters = num_filters
    self.input_dim = input_dim
    self._build_layer_components()
    super(InceptionResnetA, self).__init__(**kwargs)

  def _build_layer_components(self):
    """Builds the layers components and set _layers attribute."""
    self.conv_block1 = [
        Conv2D(
            self.num_filters,
            kernel_size=(1, 1),
            strides=1,
            padding="same",
            activation=tf.nn.relu)
    ]

    self.conv_block2 = [
        Conv2D(
            filters=self.num_filters,
            kernel_size=(1, 1),
            strides=1,
            activation=tf.nn.relu,
            padding="same")
    ]

    self.conv_block2.append(
        Conv2D(
            filters=self.num_filters,
            kernel_size=(3, 3),
            strides=1,
            activation=tf.nn.relu,
            padding="same"))

    self.conv_block3 = [
        Conv2D(
            filters=self.num_filters,
            kernel_size=1,
            strides=1,
            activation=tf.nn.relu,
            padding="same")
    ]

    self.conv_block3.append(
        Conv2D(
            filters=int(self.num_filters * 1.5),
            kernel_size=(3, 3),
            strides=1,
            activation=tf.nn.relu,
            padding="same"))
    self.conv_block3.append(
        Conv2D(
            filters=self.num_filters * 2,
            kernel_size=(3, 3),
            strides=1,
            activation=tf.nn.relu,
            padding="same"))

    self.conv_block4 = [
        Conv2D(
            filters=self.input_dim,
            kernel_size=(1, 1),
            strides=1,
            padding="same")
    ]

    self.concat_layer = Concatenate()
    self.add_layer = Add()
    self.activation_layer = ReLU()

    self._layers = self.conv_block1 + self.conv_block2 + self.conv_block3 + self.conv_block4
    self._layers.extend(
        [self.concat_layer, self.add_layer, self.activation_layer])

  def call(self, inputs):
    conv1 = inputs
    for layer in self.conv_block1:
      conv1 = layer(conv1)

    conv2 = inputs
    for layer in self.conv_block2:
      conv2 = layer(conv2)

    conv3 = inputs
    for layer in self.conv_block3:
      conv3 = layer(conv3)

    concat_conv = self.concat_layer([conv1, conv2, conv3])

    conv4 = concat_conv
    for layer in self.conv_block4:
      conv4 = layer(conv4)
    output = self.add_layer([conv4, inputs])
    return self.activation_layer(output)


class InceptionResnetB(tf.keras.layers.Layer):

  def __init__(self, num_filters, input_dim, **kwargs):
    """
    Parameters
    ----------
    num_filters: int,
        Number of convolutional filters
    input_dim: int,
        Number of channels in the input.
    """
    self.num_filters = num_filters
    self.input_dim = input_dim
    self._build_layer_components()
    super(InceptionResnetB, self).__init__(**kwargs)

  def _build_layer_components(self):
    """Builds the layers components and set _layers attribute."""
    self.conv_block1 = [
        Conv2D(
            self.num_filters,
            kernel_size=1,
            strides=1,
            padding="same",
            activation=tf.nn.relu)
    ]

    self.conv_block2 = [
        Conv2D(
            filters=self.num_filters,
            kernel_size=(1, 1),
            strides=1,
            activation=tf.nn.relu,
            padding="same")
    ]
    self.conv_block2.append(
        Conv2D(
            filters=int(self.num_filters * 1.25),
            kernel_size=(1, 7),
            strides=1,
            activation=tf.nn.relu,
            padding="same"))
    self.conv_block2.append(
        Conv2D(
            filters=int(self.num_filters * 1.5),
            kernel_size=(7, 1),
            strides=1,
            activation=tf.nn.relu,
            padding="same"))

    self.conv_block3 = [
        Conv2D(
            filters=self.input_dim, kernel_size=1, strides=1, padding="same")
    ]

    self.concat_layer = Concatenate()
    self.add_layer = Add()
    self.activation_layer = ReLU()

    self._layers = self.conv_block1 + self.conv_block2 + self.conv_block3
    self._layers.extend(
        [self.concat_layer, self.add_layer, self.activation_layer])

  def call(self, inputs):
    conv1 = inputs
    for layer in self.conv_block1:
      conv1 = layer(conv1)

    conv2 = inputs
    for layer in self.conv_block2:
      conv2 = layer(conv2)

    concat_conv = self.concat_layer([conv1, conv2])

    conv3 = concat_conv
    for layer in self.conv_block3:
      conv3 = layer(conv3)

    output = self.add_layer([conv3, inputs])
    output = self.activation_layer(output)
    return output


class InceptionResnetC(tf.keras.layers.Layer):

  def __init__(self, num_filters, input_dim, **kwargs):
    """
    Parameters
    ----------
    num_filters: int,
        Number of convolutional filters
    input_dim: int,
        Number of channels in the input.
    """
    self.num_filters = num_filters
    self.input_dim = input_dim
    self._build_layer_components()
    super(InceptionResnetC, self).__init__(**kwargs)

  def _build_layer_components(self):
    """Builds the layers components and set _layers attribute."""
    self.conv_block1 = [
        Conv2D(
            self.num_filters,
            kernel_size=(1, 1),
            strides=1,
            padding="same",
            activation=tf.nn.relu)
    ]

    self.conv_block2 = [
        Conv2D(
            filters=self.num_filters,
            kernel_size=1,
            strides=1,
            activation=tf.nn.relu,
            padding="same")
    ]
    self.conv_block2.append(
        Conv2D(
            filters=int(self.num_filters * 1.16),
            kernel_size=(1, 3),
            strides=1,
            activation=tf.nn.relu,
            padding="same"))
    self.conv_block2.append(
        Conv2D(
            filters=int(self.num_filters * 1.33),
            kernel_size=(3, 1),
            strides=1,
            activation=tf.nn.relu,
            padding="same"))

    self.conv_block3 = [
        Conv2D(
            filters=self.input_dim,
            kernel_size=(1, 1),
            strides=1,
            padding="same")
    ]

    self.concat_layer = Concatenate()
    self.add_layer = Add()
    self.activation_layer = ReLU()

    self._layers = self.conv_block1 + self.conv_block2 + self.conv_block3
    self._layers.extend(
        [self.concat_layer, self.add_layer, self.activation_layer])

  def call(self, inputs):
    conv1 = inputs
    for layer in self.conv_block1:
      conv1 = layer(conv1)

    conv2 = inputs
    for layer in self.conv_block2:
      conv2 = layer(conv2)

    concat_conv = self.concat_layer([conv1, conv2])

    conv3 = concat_conv
    for layer in self.conv_block3:
      conv3 = layer(conv3)

    output = self.add_layer([conv3, inputs])
    output = self.activation_layer(output)
    return output


class ReductionA(tf.keras.layers.Layer):

  def __init__(self, num_filters, **kwargs):
    """
    Parameters
    ----------
    num_filters: int,
        Number of convolutional filters
    """
    self.num_filters = num_filters
    self._build_layer_components()
    super(ReductionA, self).__init__(**kwargs)

  def _build_layer_components(self):
    """Builds the layers components and set _layers attribute."""
    self.max_pool1 = MaxPool2D(pool_size=(3, 3), strides=2, padding="valid")

    self.conv_block1 = [
        Conv2D(
            int(self.num_filters * 1.5),
            kernel_size=(3, 3),
            strides=2,
            padding="valid",
            activation=tf.nn.relu)
    ]

    self.conv_block2 = [
        Conv2D(
            filters=self.num_filters,
            kernel_size=1,
            strides=1,
            activation=tf.nn.relu,
            padding="same")
    ]
    self.conv_block2.append(
        Conv2D(
            filters=self.num_filters,
            kernel_size=3,
            strides=1,
            activation=tf.nn.relu,
            padding="same"))
    self.conv_block2.append(
        Conv2D(
            filters=int(self.num_filters * 1.5),
            kernel_size=3,
            strides=2,
            activation=tf.nn.relu,
            padding="valid"))

    self.concat_layer = Concatenate()
    self.activation_layer = ReLU()

    self._layers = self.conv_block1 + self.conv_block2
    self._layers.extend(
        [self.max_pool1, self.concat_layer, self.activation_layer])

  def call(self, inputs):
    maxpool1 = self.max_pool1(inputs)
    conv1 = inputs
    for layer in self.conv_block1:
      conv1 = layer(conv1)

    conv2 = inputs
    for layer in self.conv_block2:
      conv2 = layer(conv2)

    output = self.concat_layer([maxpool1, conv1, conv2])
    output = self.activation_layer(output)
    return output


class ReductionB(tf.keras.layers.Layer):

  def __init__(self, num_filters, **kwargs):
    """
    Parameters
    ----------
    num_filters: int,
        Number of convolutional filters
    """
    self.num_filters = num_filters
    self._build_layer_components()
    super(ReductionB, self).__init__(**kwargs)

  def _build_layer_components(self):
    """Builds the layers components and set _layers attribute."""
    self.max_pool1 = MaxPool2D(pool_size=(3, 3), strides=2, padding="valid")

    self.conv_block1 = [
        Conv2D(
            self.num_filters,
            kernel_size=1,
            strides=1,
            padding="same",
            activation=tf.nn.relu)
    ]
    self.conv_block1.append(
        Conv2D(
            int(self.num_filters * 1.5),
            kernel_size=3,
            strides=2,
            padding="valid",
            activation=tf.nn.relu))

    self.conv_block2 = [
        Conv2D(
            filters=self.num_filters,
            kernel_size=1,
            strides=1,
            activation=tf.nn.relu,
            padding="same")
    ]
    self.conv_block2.append(
        Conv2D(
            filters=int(self.num_filters * 1.125),
            kernel_size=3,
            strides=2,
            activation=tf.nn.relu,
            padding="valid"))

    self.conv_block3 = [
        Conv2D(
            filters=self.num_filters,
            kernel_size=1,
            strides=1,
            activation=tf.nn.relu,
            padding="same")
    ]
    self.conv_block3.append(
        Conv2D(
            filters=int(self.num_filters * 1.125),
            kernel_size=(3, 1),
            strides=1,
            activation=tf.nn.relu,
            padding="same"))

    self.conv_block3.append(
        Conv2D(
            filters=int(self.num_filters * 1.25),
            kernel_size=(3, 3),
            strides=2,
            activation=tf.nn.relu,
            padding="valid"))

    self.concat_layer = Concatenate()
    self.activation_layer = ReLU()

    self._layers = self.conv_block1 + self.conv_block2 + self.conv_block3
    self._layers.extend(
        [self.max_pool1, self.concat_layer, self.activation_layer])

  def call(self, inputs):
    maxpool1 = self.max_pool1(inputs)
    conv1 = inputs
    for layer in self.conv_block1:
      conv1 = layer(conv1)

    conv2 = inputs
    for layer in self.conv_block2:
      conv2 = layer(conv2)

    conv3 = inputs
    for layer in self.conv_block3:
      conv3 = layer(conv3)

    concat = self.concat_layer([maxpool1, conv1, conv2, conv3])
    output = self.activation_layer(concat)
    return output
