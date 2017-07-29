"""Optimizers and related classes for use with TensorGraph."""

import tensorflow as tf


class Optimizer(object):
  """An algorithm for optimizing a TensorGraph based model.

  This is an abstract class.  Subclasses represent specific optimization algorithms.
  """

  def _create_optimizer(self, global_step):
    """Construct the TensorFlow optimizer.

    Parameters
    ----------
    global_step: tensor
      a tensor containing the global step index during optimization, used for learning rate decay

    Returns
    -------
    a new TensorFlow optimizer implementing the algorithm
    """
    raise NotImplemented("Subclasses must implement this")


class LearningRateSchedule(object):
  """A schedule for changing the learning rate over the course of optimization.

  This is an abstract class.  Subclasses represent specific schedules.
  """

  def _create_tensor(self, global_step):
    """Construct a tensor that equals the learning rate.

    Parameters
    ----------
    global_step: tensor
      a tensor containing the global step index during optimization

    Returns
    -------
    a tensor that equals the learning rate
    """
    raise NotImplemented("Subclasses must implement this")


class Adam(Optimizer):
  """The Adam optimization algorithm."""

  def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999,
               epsilon=1e-08):
    """Construct an Adam optimizer.

    Parameters
    ----------
    learning_rate: float or LearningRateSchedule
      the learning rate to use for optimization
    beta1: float
      a parameter of the Adam algorithm
    beta2: float
      a parameter of the Adam algorithm
    epsilon: float
      a parameter of the Adam algorithm
    """
    self.learning_rate = learning_rate
    self.beta1 = beta1
    self.beta2 = beta2
    self.epsilon = epsilon

  def _create_optimizer(self, global_step):
    if isinstance(self.learning_rate, LearningRateSchedule):
      learning_rate = self.learning_rate._create_tensor(global_step)
    else:
      learning_rate = self.learning_rate
    return tf.train.AdamOptimizer(
        learning_rate=learning_rate,
        beta1=self.beta1,
        beta2=self.beta2,
        epsilon=self.epsilon)


class GradientDescent(Optimizer):
  """The gradient descent optimization algorithm."""

  def __init__(self, learning_rate=0.001):
    """Construct a gradient descent optimizer.

    Parameters
    ----------
    learning_rate: float or LearningRateSchedule
      the learning rate to use for optimization
    """
    self.learning_rate = learning_rate

  def _create_optimizer(self, global_step):
    if isinstance(self.learning_rate, LearningRateSchedule):
      learning_rate = self.learning_rate._create_tensor(global_step)
    else:
      learning_rate = self.learning_rate
    return tf.train.GradientDescentOptimizer(learning_rate=learning_rate)


class ExponentialDecay(LearningRateSchedule):
  """A learning rate that decreases exponentially with the number of training steps."""

  def __init__(self, initial_rate, decay_rate, decay_steps, staircase=True):
    """Create an exponentially decaying learning rate.

    The learning rate starts as initial_rate.  Every decay_steps training steps, it is multiplied by decay_rate.

    Parameters
    ----------
    initial_rate: float
      the initial learning rate
    decay_rate: float
      the base of the exponential
    decay_steps: int
      the number of training steps over which the rate decreases by decay_rate
    staircase: bool
      if True, the learning rate decreases by discrete jumps every decay_steps.
      if False, the learning rate decreases smoothly every step
    """
    self.initial_rate = initial_rate
    self.decay_rate = decay_rate
    self.decay_steps = decay_steps
    self.staircase = staircase

  def _create_tensor(self, global_step):
    return tf.train.exponential_decay(
        learning_rate=self.initial_rate,
        global_step=global_step,
        decay_rate=self.decay_rate,
        decay_steps=self.decay_steps,
        staircase=self.staircase)


class PolynomialDecay(LearningRateSchedule):
  """A learning rate that decreases from an initial value to a final value over a fixed number of training steps."""

  def __init__(self, initial_rate, final_rate, decay_steps, power=1.0):
    """Create a smoothly decaying learning rate.

    The learning rate starts as initial_rate.  It smoothly decreases to final_rate over decay_steps training steps.
    It decays as a function of (1-step/decay_steps)**power.  Once the final rate is reached, it remains there for
    the rest of optimization.

    Parameters
    ----------
    initial_rate: float
      the initial learning rate
    final_rate: float
      the final learning rate
    decay_steps: int
      the number of training steps over which the rate decreases from initial_rate to final_rate
    power: float
      the exponent controlling the shape of the decay
    """
    self.initial_rate = initial_rate
    self.final_rate = final_rate
    self.decay_steps = decay_steps
    self.power = power

  def _create_tensor(self, global_step):
    return tf.train.polynomial_decay(
        learning_rate=self.initial_rate,
        end_learning_rate=self.final_rate,
        global_step=global_step,
        decay_steps=self.decay_steps,
        power=self.power)
