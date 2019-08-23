"""Model-Agnostic Meta-Learning (MAML) algorithm for low data learning."""

from deepchem.models.optimizers import Adam, GradientDescent
import numpy as np
import os
import shutil
import tempfile
import tensorflow as tf
import time


class MetaLearner(object):
  """Model and data to which the MAML algorithm can be applied.

  To use MAML, create a subclass of this defining the learning problem to solve.
  It consists of a model that can be trained to perform many different tasks, and
  data for training it on a large (possibly infinite) set of different tasks.
  """

  def compute_model(self, inputs, variables, training):
    """Compute the model for a set of inputs and variables.

    Parameters
    ----------
    inputs: list of tensors
      the inputs to the model
    variables: list of tensors
      the values to use for the model's variables.  This might be the actual
      variables (as returned by the MetaLearner's variables property), or
      alternatively it might be the values of those variables after one or more
      steps of gradient descent for the current task.
    training: bool
      indicates whether the model is being invoked for training or prediction

    Returns
    -------
    (loss, outputs) where loss is the value of the model's loss function, and
    outputs is a list of the model's outputs
    """
    raise NotImplemented("Subclasses must implement this")

  @property
  def variables(self):
    """Get the list of Tensorflow variables to train."""
    raise NotImplemented("Subclasses must implement this")

  def select_task(self):
    """Select a new task to train on.

    If there is a fixed set of training tasks, this will typically cycle through them.
    If there are infinitely many training tasks, this can simply select a new one each
    time it is called.
    """
    raise NotImplemented("Subclasses must implement this")

  def get_batch(self):
    """Get a batch of data for training.

    This should return the data as a list of arrays, one for each of the model's
    inputs.  This will usually be called twice for each task, and should
    return a different batch on each call.
    """
    raise NotImplemented("Subclasses must implement this")


class MAML(object):
  """Implements the Model-Agnostic Meta-Learning algorithm for low data learning.

  The algorithm is described in Finn et al., "Model-Agnostic Meta-Learning for Fast
  Adaptation of Deep Networks" (https://arxiv.org/abs/1703.03400).  It is used for
  training models that can perform a variety of tasks, depending on what data they
  are trained on.  It assumes you have training data for many tasks, but only a small
  amount for each one.  It performs "meta-learning" by looping over tasks and trying
  to minimize the loss on each one *after* one or a few steps of gradient descent.
  That is, it does not try to create a model that can directly solve the tasks, but
  rather tries to create a model that is very easy to train.

  To use this class, create a subclass of MetaLearner that encapsulates the model
  and data for your learning problem.  Pass it to a MAML object and call fit().
  You can then use train_on_current_task() to fine tune the model for a particular
  task.
  """

  def __init__(self,
               learner,
               learning_rate=0.001,
               optimization_steps=1,
               meta_batch_size=10,
               optimizer=Adam(),
               model_dir=None):
    """Create an object for performing meta-optimization.

    Parameters
    ----------
    learner: MetaLearner
      defines the meta-learning problem
    learning_rate: float or Tensor
      the learning rate to use for optimizing each task (not to be confused with the one used
      for meta-learning).  This can optionally be made a variable (represented as a
      Tensor), in which case the learning rate will itself be learnable.
    optimization_steps: int
      the number of steps of gradient descent to perform for each task
    meta_batch_size: int
      the number of tasks to use for each step of meta-learning
    optimizer: Optimizer
      the optimizer to use for meta-learning (not to be confused with the gradient descent
      optimization performed for each task)
    model_dir: str
      the directory in which the model will be saved.  If None, a temporary directory will be created.
    """
    # Record inputs.

    self.learner = learner
    self._learning_rate = learning_rate
    self.meta_batch_size = meta_batch_size
    self.optimizer = optimizer

    # Create the output directory if necessary.

    self._model_dir_is_temp = False
    if model_dir is not None:
      if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    else:
      model_dir = tempfile.mkdtemp()
      self._model_dir_is_temp = True
    self.model_dir = model_dir
    self.save_file = "%s/%s" % (self.model_dir, "model")

    learner.select_task()
    example_inputs = learner.get_batch()
    self._input_shapes = [(None,) + i.shape[1:] for i in example_inputs]
    self._input_dtypes = [x.dtype for x in example_inputs]
    self._input_placeholders = [
        tf.placeholder(dtype=tf.as_dtype(t), shape=s)
        for s, t in zip(self._input_shapes, self._input_dtypes)
    ]
    self._meta_placeholders = [
        tf.placeholder(dtype=tf.as_dtype(t), shape=s)
        for s, t in zip(self._input_shapes, self._input_dtypes)
    ]
    variables = learner.variables
    self._loss, self._outputs = learner.compute_model(self._input_placeholders,
                                                      variables, False)
    loss, _ = learner.compute_model(self._input_placeholders, variables, True)

    # Build the meta-learning model.

    updated_variables = variables
    for i in range(optimization_steps):
      gradients = tf.gradients(loss, updated_variables)
      updated_variables = [
          v if g is None else v - self._learning_rate * g
          for v, g in zip(updated_variables, gradients)
      ]
      if i == optimization_steps - 1:
        # In the final loss, use different placeholders for all inputs so the loss will be
        # computed from a different batch.

        inputs = self._meta_placeholders
      else:
        inputs = self._input_placeholders
      loss, outputs = learner.compute_model(inputs, updated_variables, True)
    self._meta_loss = loss

    # Create variables for accumulating the gradients.

    variables = list(learner.variables)
    gradients = tf.gradients(self._meta_loss, variables)
    for i in reversed(range(len(variables))):
      if gradients[i] is None:
        del variables[i]
        del gradients[i]
    zero_gradients = [tf.zeros(g.shape, g.dtype) for g in gradients]
    summed_gradients = [tf.Variable(z, trainable=False) for z in zero_gradients]
    self._clear_gradients = tf.group(
        *[s.assign(z) for s, z in zip(summed_gradients, zero_gradients)])
    self._add_gradients = tf.group(
        *[s.assign_add(g) for s, g in zip(summed_gradients, gradients)])

    # Create the optimizers for meta-optimization and task optimization.

    self._global_step = tf.placeholder(tf.int32, [])
    grads_and_vars = list(zip(summed_gradients, variables))
    self._meta_train_op = optimizer._create_optimizer(
        self._global_step).apply_gradients(grads_and_vars)
    task_optimizer = GradientDescent(learning_rate=self._learning_rate)
    self._task_train_op = task_optimizer._create_optimizer(
        self._global_step).minimize(self._loss)
    self._session = tf.Session()
    self._session.run(tf.global_variables_initializer())

    # Create a Checkpoint for saving.

    self._checkpoint = tf.train.Checkpoint()
    self._checkpoint.listed = learner.variables

  def __del__(self):
    if '_model_dir_is_temp' in dir(self) and self._model_dir_is_temp:
      shutil.rmtree(self.model_dir)

  def fit(self,
          steps,
          max_checkpoints_to_keep=5,
          checkpoint_interval=600,
          restore=False):
    """Perform meta-learning to train the model.

    Parameters
    ----------
    steps: int
      the number of steps of meta-learning to perform
    max_checkpoints_to_keep: int
      the maximum number of checkpoint files to keep.  When this number is reached, older
      files are deleted.
    checkpoint_interval: float
      the time interval at which to save checkpoints, measured in seconds
    restore: bool
      if True, restore the model from the most recent checkpoint before training
      it further
    """
    if restore:
      self.restore()
    manager = tf.train.CheckpointManager(self._checkpoint, self.model_dir,
                                         max_checkpoints_to_keep)
    checkpoint_time = time.time()

    # Main optimization loop.

    for i in range(steps):
      self._session.run(self._clear_gradients)
      for j in range(self.meta_batch_size):
        self.learner.select_task()
        inputs = self.learner.get_batch()
        feed_dict = {}
        feed_dict[self._global_step] = i
        for k in range(len(inputs)):
          feed_dict[self._input_placeholders[k]] = inputs[k]
          feed_dict[self._meta_placeholders[k]] = inputs[k]
        self._session.run(self._add_gradients, feed_dict=feed_dict)
      self._session.run(self._meta_train_op)

      # Do checkpointing.

      if i == steps - 1 or time.time() >= checkpoint_time + checkpoint_interval:
        with self._session.as_default():
          manager.save()
        checkpoint_time = time.time()

  def restore(self):
    """Reload the model parameters from the most recent checkpoint file."""
    last_checkpoint = tf.train.latest_checkpoint(self.model_dir)
    if last_checkpoint is None:
      raise ValueError('No checkpoint found')
    self._checkpoint.restore(last_checkpoint).run_restore_ops(self._session)

  def train_on_current_task(self, optimization_steps=1, restore=True):
    """Perform a few steps of gradient descent to fine tune the model on the current task.

    Parameters
    ----------
    optimization_steps: int
      the number of steps of gradient descent to perform
    restore: bool
      if True, restore the model from the most recent checkpoint before optimizing
    """
    if restore:
      self.restore()
    inputs = self.learner.get_batch()
    feed_dict = {}
    for p, v in zip(self._input_placeholders, inputs):
      feed_dict[p] = v
    for i in range(optimization_steps):
      self._session.run(self._task_train_op, feed_dict=feed_dict)

  def predict_on_batch(self, inputs):
    """Compute the model's outputs for a batch of inputs.

    Parameters
    ----------
    inputs: list of arrays
      the inputs to the model

    Returns
    -------
    (loss, outputs) where loss is the value of the model's loss function, and
    outputs is a list of the model's outputs
    """
    feed_dict = {}
    for p, v in zip(self._input_placeholders, inputs):
      feed_dict[p] = v
    return self._session.run([self._loss, self._outputs], feed_dict=feed_dict)
