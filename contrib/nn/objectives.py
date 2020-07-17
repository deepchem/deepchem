"""Ops for objectives 

Code borrowed from Keras.
"""
import warnings
import tensorflow as tf
from deepchem.nn import model_ops


def mean_squared_error(y_true, y_pred):
  return model_ops.mean(tf.square(y_pred - y_true), axis=-1)


def mean_absolute_error(y_true, y_pred):
  return model_ops.mean(tf.abs(y_pred - y_true), axis=-1)


def mean_absolute_percentage_error(y_true, y_pred):
  diff = tf.abs((y_true - y_pred) / model_ops.clip(
      tf.abs(y_true), model_ops.epsilon(), None))
  return 100. * model_ops.mean(diff, axis=-1)


def mean_squared_logarithmic_error(y_true, y_pred):
  first_log = tf.log(model_ops.clip(y_pred, model_ops.epsilon(), None) + 1.)
  second_log = tf.log(model_ops.clip(y_true, model_ops.epsilon(), None) + 1.)
  return model_ops.mean(tf.square(first_log - second_log), axis=-1)


def squared_hinge(y_true, y_pred):
  return model_ops.mean(
      tf.square(tf.maximum(1. - y_true * y_pred, 0.)), axis=-1)


def hinge(y_true, y_pred):
  return model_ops.mean(tf.maximum(1. - y_true * y_pred, 0.), axis=-1)


def categorical_crossentropy(y_true, y_pred):
  return model_ops.categorical_crossentropy(y_pred, y_true)


def sparse_categorical_crossentropy(y_true, y_pred):
  return model_ops.sparse_categorical_crossentropy(y_pred, y_true)


def binary_crossentropy(y_true, y_pred):
  return model_ops.mean(model_ops.binary_crossentropy(y_pred, y_true), axis=-1)


def kullback_leibler_divergence(y_true, y_pred):
  y_true = model_ops.clip(y_true, model_ops.epsilon(), 1)
  y_pred = model_ops.clip(y_pred, model_ops.epsilon(), 1)
  return model_ops.sum(y_true * tf.log(y_true / y_pred), axis=-1)


def poisson(y_true, y_pred):
  return model_ops.mean(
      y_pred - y_true * tf.log(y_pred + model_ops.epsilon()), axis=-1)


def cosine_proximity(y_true, y_pred):
  y_true = model_ops.l2_normalize(y_true, axis=-1)
  y_pred = model_ops.l2_normalize(y_pred, axis=-1)
  return -model_ops.mean(y_true * y_pred, axis=-1)


# Aliases.

mse = MSE = mean_squared_error
mae = MAE = mean_absolute_error
mape = MAPE = mean_absolute_percentage_error
msle = MSLE = mean_squared_logarithmic_error
kld = KLD = kullback_leibler_divergence
cosine = cosine_proximity
