"""
Activations for models.
"""
from __future__ import division
from __future__ import unicode_literals

import tensorflow as tf
from deepchem.models.tensorgraph import model_ops
from deepchem.models.tensorgraph.model_ops import get_ndim


def get_from_module(identifier,
                    module_params,
                    module_name,
                    instantiate=False,
                    kwargs=None):
  """Retrieves a class of function member of a module.

  Parameters
  ----------
  identifier: the object to retrieve. It could be specified
    by name (as a string), or by dict. In any other case,
    identifier itself will be returned without any changes.
  module_params: the members of a module
    (e.g. the output of globals()).
  module_name: string; the name of the target module. Only used
    to format error messages.
  instantiate: whether to instantiate the returned object
    (if it's a class).
  kwargs: a dictionary of keyword arguments to pass to the
    class constructor if `instantiate` is `True`.

  Returns
  -------
  The target object.

  Raises
  ------
  ValueError: if the identifier cannot be found.
  """
  import six
  if isinstance(identifier, six.string_types):
    res = module_params.get(identifier)
    if not res:
      raise ValueError('Invalid ' + str(module_name) + ': ' + str(identifier))
    if instantiate and not kwargs:
      return res()
    elif instantiate and kwargs:
      return res(**kwargs)
    else:
      return res

  return identifier


def softmax(x):
  ndim = get_ndim(x)
  if ndim == 2:
    return tf.nn.softmax(x)
  elif ndim == 3:
    e = tf.exp(x - model_ops.max(x, axis=-1, keepdims=True))
    s = model_ops.sum(e, axis=-1, keepdims=True)
    return e / s
  else:
    raise ValueError('Cannot apply softmax to a tensor '
                     'that is not 2D or 3D. '
                     'Here, ndim=' + str(ndim))


def elu(x, alpha=1.0):
  return model_ops.elu(x, alpha)


def selu(x):
  return model_ops.selu(x)


def softplus(x):
  return tf.nn.softplus(x)


def softsign(x):
  return tf.nn.softsign(x)


def relu(x, alpha=0., max_value=None):
  """The rectified linear activation function

  Wrapper around model_ops.relu.

  Parameters
  ----------
  x: tf.Tensor
    Input tensor
  """
  return model_ops.relu(x, alpha=alpha, max_value=max_value)


def tanh(x):
  """The hyperbolic tanget activation function

  Wrapper around tf.nn.tanh.

  Parameters
  ----------
  x: tf.Tensor
    Input tensor
  """
  return tf.nn.tanh(x)


def sigmoid(x):
  """The sigmoidal activation function

  Wrapper around tf.nn.sigmoid.

  Parameters
  ----------
  x: tf.Tensor
    Input tensor
  """
  return tf.nn.sigmoid(x)


def hard_sigmoid(x):
  """The hard sigmoidal activation function

  Piecewise-linear approximation to sigmoid. 

  Parameters
  ----------
  x: tf.Tensor
    Input tensor
  """
  return model_ops.hard_sigmoid(x)


def linear(x):
  """A linear activation function.

  Note that a linear activation function is simply the identity.

  Parameters
  ----------
  x: tf.Tensor
    Input tensor
  """
  return x


def get(identifier):
  if identifier is None:
    return linear
  return get_from_module(identifier, globals(), 'activation function')
