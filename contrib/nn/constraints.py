"""Place constraints on models."""
from deepchem.nn import model_ops
from deepchem.nn.activations import get_from_module

class Constraint(object):

  def __call__(self, p):
    return p

class MaxNorm(Constraint):
  """MaxNorm weight constraint.

  Constrains the weights incident to each hidden unit
  to have a norm less than or equal to a desired value.

  Parameters
  ----------
  m: the maximum norm for the incoming weights.
  axis: integer, axis along which to calculate weight norms.
    For instance, in a `Dense` layer the weight matrix
    has shape (input_dim, output_dim),
    set axis to 0 to constrain each weight vector
    of length `(input_dim,)`.

  # References
    - [Dropout: A Simple Way to Prevent Neural Networks from Overfitting Srivastava, Hinton, et al. 2014](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)
  """

  def __init__(self, m=2, axis=0):
    self.m = m
    self.axis = axis

  def __call__(self, p):
    norms = model_ops.sqrt(model_ops.sum(
        tf.square(p), axis=self.axis, keepdims=True))
    desired = model_ops.clip(norms, 0, self.m)
    p *= (desired / (model_ops.epsilon() + norms))
    return p

class NonNeg(Constraint):
  """Constrains the weights to be non-negative.
  """
  def __call__(self, p):
    p *= tf.cast(p >= 0., tf.float32)
    return p


class UnitNorm(Constraint):
  """Constrains the weights incident to each hidden unit to have unit norm.

  # Arguments
      axis: integer, axis along which to calculate weight norms.
          For instance, in a `Dense` layer the weight matrix
          has shape `(input_dim, output_dim)`,
          set `axis` to `0` to constrain each weight vector
          of length `(input_dim,)`.
          In a `Convolution2D` layer with `dim_ordering="tf"`,
          the weight tensor has shape
          `(rows, cols, input_depth, output_depth)`,
          set `axis` to `[0, 1, 2]`
          to constrain the weights of each filter tensor of size
          `(rows, cols, input_depth)`.
  """

  def __init__(self, axis=0):
    self.axis = axis

  def __call__(self, p):
    return p / (model_ops.epsilon() + model_ops.sqrt(
        model_ops.sum(tf.square(p), axis=self.axis, keepdims=True)))

# Aliases.
maxnorm = MaxNorm
nonneg = NonNeg
unitnorm = UnitNorm


def get(identifier, kwargs=None):
  return get_from_module(identifier, globals(), 'constraint',
                         instantiate=True, kwargs=kwargs)
