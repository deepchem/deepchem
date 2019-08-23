import numpy as np
import tensorflow as tf
import time
import logging
import os
try:
  from collections.abc import Sequence
except:
  from collections import Sequence

logger = logging.getLogger(__name__)

from deepchem.data import NumpyDataset
from deepchem.models.losses import Loss
from deepchem.models.models import Model
from deepchem.models.optimizers import Adam
from deepchem.trans import undo_transforms
from deepchem.utils.evaluate import GeneratorEvaluator


class KerasModel(Model):
  """This is a DeepChem model implemented by a Keras model.

  This class provides several advantages over using the Keras model's fitting
  and prediction methods directly.

  1. It provides better integration with the rest of DeepChem, such as direct
     support for Datasets and Transformers.

  2. It defines the loss in a more flexible way.  In particular, Keras does not
     support multidimensional weight matrices, which makes it impossible to
     implement most multitask models with Keras.

  3. It provides various additional features not found in the Keras Model class,
     such as uncertainty prediction and saliency mapping.

  The loss function for a model can be defined in two different ways.  For
  models that have only a single output and use a standard loss function, you
  can simply provide a dc.models.losses.Loss object.  This defines the loss for
  each sample or sample/task pair.  The result is automatically multiplied by
  the weights and averaged over the batch.  Any additional losses computed by
  model layers, such as weight decay penalties, are also added.

  For more complicated cases, you can instead provide a function that directly
  computes the total loss.  It must be of the form f(outputs, labels, weights),
  taking the list of outputs from the model, the expected values, and any weight
  matrices.  It should return a scalar equal to the value of the loss function
  for the batch.  No additional processing is done to the result; it is up to
  you to do any weighting, averaging, adding of penalty terms, etc.

  You can optionally provide an output_types argument, which describes how to
  interpret the model's outputs.  This should be a list of strings, one for each
  output.  Each entry must have one of the following values:

  - 'prediction': This is a normal output, and will be returned by predict().
    If output types are not specified, all outputs are assumed to be of this
    type.

  - 'loss': This output will be used in place of the normal outputs for
    computing the loss function.  For example, models that output probability
    distributions usually do it by computing unbounded numbers (the logits),
    then passing them through a softmax function to turn them into
    probabilities.  When computing the cross entropy, it is more numerically
    stable to use the logits directly rather than the probabilities.  You can
    do this by having the model produce both probabilities and logits as
    outputs, then specifying output_types=['prediction', 'loss'].  When
    predict() is called, only the first output (the probabilities) will be
    returned.  But during training, it is the second output (the logits) that
    will be passed to the loss function.

  - 'variance': This output is used for estimating the uncertainty in another
    output.  To create a model that can estimate uncertainty, there must be the
    same number of 'prediction' and 'variance' outputs.  Each variance output
    must have the same shape as the corresponding prediction output, and each
    element is an estimate of the variance in the corresponding prediction.
    Also be aware that if a model supports uncertainty, it MUST use dropout on
    every layer, and dropout most be enabled during uncertainty prediction.
    Otherwise, the uncertainties it computes will be inaccurate.
  """

  def __init__(self,
               model,
               loss,
               output_types=None,
               batch_size=100,
               model_dir=None,
               learning_rate=0.001,
               optimizer=None,
               tensorboard=False,
               tensorboard_log_frequency=100,
               **kwargs):
    """Create a new KerasModel.

    Parameters
    ----------
    model: tf.keras.Model
      the Keras model implementing the calculation
    loss: dc.models.losses.Loss or function
      a Loss or function defining how to compute the training loss for each
      batch, as described above
    output_types: list of strings
      the type of each output from the model, as described above
    batch_size: int
      default batch size for training and evaluating
    model_dir: str
      the directory on disk where the model will be stored.  If this is None,
      a temporary directory is created.
    learning_rate: float or LearningRateSchedule
      the learning rate to use for fitting.  If optimizer is specified, this is
      ignored.
    optimizer: Optimizer
      the optimizer to use for fitting.  If this is specified, learning_rate is
      ignored.
    tensorboard: bool
      whether to log progress to TensorBoard during training
    tensorboard_log_frequency: int
      the frequency at which to log data to TensorBoard, measured in batches
    """
    super(KerasModel, self).__init__(
        model_instance=model, model_dir=model_dir, **kwargs)
    self.model = model
    if isinstance(loss, Loss):
      self._loss_fn = _StandardLoss(model, loss)
    else:
      self._loss_fn = loss
    self.batch_size = batch_size
    if optimizer is None:
      self.optimizer = Adam(learning_rate=learning_rate)
    else:
      self.optimizer = optimizer
    self.tensorboard = tensorboard
    self.tensorboard_log_frequency = tensorboard_log_frequency
    if self.tensorboard:
      if tf.executing_eagerly():
        raise ValueError(
            "Logging to TensorBoard is not currently supported in eager mode")
      self._summary_writer = tf.summary.FileWriter(self.model_dir)
    if output_types is None:
      self._prediction_outputs = None
      self._loss_outputs = None
      self._variance_outputs = None
    else:
      self._prediction_outputs = []
      self._loss_outputs = []
      self._variance_outputs = []
      for i, type in enumerate(output_types):
        if type == 'prediction':
          self._prediction_outputs.append(i)
        elif type == 'loss':
          self._loss_outputs.append(i)
        elif type == 'variance':
          self._variance_outputs.append(i)
        else:
          raise ValueError('Unknown output type "%s"' % type)
      if len(self._loss_outputs) == 0:
        self._loss_outputs = self._prediction_outputs
    self._built = False
    self._inputs_built = False
    self._training_ops_built = False
    self._initialized_vars = set()
    self._output_functions = {}

  def _ensure_built(self):
    """The first time this is called, create internal data structures."""
    if self._built:
      return
    self._built = True
    if not tf.executing_eagerly():
      self.session = tf.Session()
    self._global_step = tf.Variable(0, trainable=False)
    self._tf_optimizer = self.optimizer._create_optimizer(self._global_step)
    self._checkpoint = tf.train.Checkpoint(
        optimizer=self._tf_optimizer, model=self.model)
    self._init_new_vars()

  def _create_inputs(self, example_inputs):
    """The first time this is called, create tensors representing the inputs and outputs."""
    if self._inputs_built:
      return
    self._ensure_built()
    self._inputs_built = True
    if len(self.model.inputs) > 0:
      self._input_shapes = [t.shape for t in self.model.inputs]
      self._input_dtypes = [t.dtype.as_numpy_dtype for t in self.model.inputs]
    else:
      self._input_shapes = [(None,) + i.shape[1:] for i in example_inputs]
      self._input_dtypes = [
          np.float32 if x.dtype == np.float64 else x.dtype
          for x in example_inputs
      ]
    if tf.executing_eagerly():
      return
    if len(self.model.inputs) > 0:
      self._input_placeholders = self.model.inputs
    else:
      # The model doesn't specify inputs, so guess the input shapes based on the
      # example batch.
      self._input_placeholders = [
          tf.placeholder(dtype=tf.as_dtype(t), shape=s)
          for s, t in zip(self._input_shapes, self._input_dtypes)
      ]
      if len(self._input_shapes) == 1:
        self.model.build(self._input_shapes[0])
      else:
        self.model.build(self._input_shapes)
    if len(self._input_placeholders) == 1:
      self._output_tensors = self.model(
          self._input_placeholders[0], training=False)
    else:
      self._output_tensors = self.model(
          self._input_placeholders, training=False)
    if isinstance(self._output_tensors, tf.Tensor):
      self._output_tensors = [self._output_tensors]
    if self._prediction_outputs is None:
      self._prediction_outputs = list(range(len(self._output_tensors)))
      self._loss_outputs = list(range(len(self._output_tensors)))
    self._init_new_vars()

  def _create_training_ops(self, example_batch):
    """The first time this is called, create tensors used in optimization."""
    if self._training_ops_built:
      return
    self._create_inputs(example_batch[0])
    self._training_ops_built = True
    self._label_dtypes = [
        np.float32 if x.dtype == np.float64 else x.dtype
        for x in example_batch[1]
    ]
    self._weights_dtypes = [
        np.float32 if x.dtype == np.float64 else x.dtype
        for x in example_batch[2]
    ]
    if tf.executing_eagerly():
      return
    self._label_placeholders = [
        tf.placeholder(dtype=tf.as_dtype(t), shape=(None,) + x.shape[1:])
        for x, t in zip(example_batch[1], self._label_dtypes)
    ]
    self._weights_placeholders = [
        tf.placeholder(dtype=tf.as_dtype(t), shape=(None,) + x.shape[1:])
        for x, t in zip(example_batch[2], self._weights_dtypes)
    ]
    self._loss_tensor = self._loss_fn(
        [self._output_tensors[i] for i in self._loss_outputs],
        self._label_placeholders, self._weights_placeholders)
    try:
      self._train_op = self._tf_optimizer.minimize(
          self._loss_tensor, global_step=self._global_step)
    except ValueError:
      # The loss doesn't depend on any variables.
      self._train_op = 0
    self._custom_train_op = {}
    self._init_new_vars()

  def _init_new_vars(self):
    """Initialize any new variables created since the last call to this method."""
    if not tf.executing_eagerly():
      vars = set(tf.global_variables())
      new_vars = vars.difference(self._initialized_vars)
      self.session.run(tf.variables_initializer(new_vars))
      self._initialized_vars = vars

  def fit(self,
          dataset,
          nb_epoch=10,
          max_checkpoints_to_keep=5,
          checkpoint_interval=1000,
          deterministic=False,
          restore=False,
          variables=None,
          loss=None,
          callbacks=[]):
    """Train this model on a dataset.

    Parameters
    ----------
    dataset: Dataset
      the Dataset to train on
    nb_epoch: int
      the number of epochs to train for
    max_checkpoints_to_keep: int
      the maximum number of checkpoints to keep.  Older checkpoints are discarded.
    checkpoint_interval: int
      the frequency at which to write checkpoints, measured in training steps.
      Set this to 0 to disable automatic checkpointing.
    deterministic: bool
      if True, the samples are processed in order.  If False, a different random
      order is used for each epoch.
    restore: bool
      if True, restore the model from the most recent checkpoint and continue training
      from there.  If False, retrain the model from scratch.
    variables: list of tf.Variable
      the variables to train.  If None (the default), all trainable variables in
      the model are used.
    loss: function
      a function of the form f(outputs, labels, weights) that computes the loss
      for each batch.  If None (the default), the model's standard loss function
      is used.
    callbacks: function or list of functions
      one or more functions of the form f(model, step) that will be invoked after
      every step.  This can be used to perform validation, logging, etc.
   """
    return self.fit_generator(
        self.default_generator(
            dataset, epochs=nb_epoch,
            deterministic=deterministic), max_checkpoints_to_keep,
        checkpoint_interval, restore, variables, loss, callbacks)

  def fit_generator(self,
                    generator,
                    max_checkpoints_to_keep=5,
                    checkpoint_interval=1000,
                    restore=False,
                    variables=None,
                    loss=None,
                    callbacks=[]):
    """Train this model on data from a generator.

    Parameters
    ----------
    generator: generator
      this should generate batches, each represented as a tuple of the form
      (inputs, labels, weights).
    max_checkpoints_to_keep: int
      the maximum number of checkpoints to keep.  Older checkpoints are discarded.
    checkpoint_interval: int
      the frequency at which to write checkpoints, measured in training steps.
      Set this to 0 to disable automatic checkpointing.
    restore: bool
      if True, restore the model from the most recent checkpoint and continue training
      from there.  If False, retrain the model from scratch.
    variables: list of tf.Variable
      the variables to train.  If None (the default), all trainable variables in
      the model are used.
    loss: function
      a function of the form f(outputs, labels, weights) that computes the loss
      for each batch.  If None (the default), the model's standard loss function
      is used.
    callbacks: function or list of functions
      one or more functions of the form f(model, step) that will be invoked after
      every step.  This can be used to perform validation, logging, etc.

    Returns
    -------
    the average loss over the most recent checkpoint interval
    """
    if not isinstance(callbacks, Sequence):
      callbacks = [callbacks]
    self._ensure_built()
    if checkpoint_interval > 0:
      manager = tf.train.CheckpointManager(self._checkpoint, self.model_dir,
                                           max_checkpoints_to_keep)
    avg_loss = 0.0
    averaged_batches = 0
    train_op = None
    time1 = time.time()

    # Main training loop.

    for batch in generator:
      self._create_training_ops(batch)
      if restore:
        self.restore()
        restore = False
      inputs, labels, weights = self._prepare_batch(batch)
      self._current_summary = None
      if tf.executing_eagerly():

        # In eager mode we execute the loss function, accumulating the gradients.

        if loss is None:
          loss = self._loss_fn
        with tf.GradientTape() as tape:
          if len(inputs) == 1:
            inputs = inputs[0]
          outputs = self.model(inputs)
          if isinstance(outputs, tf.Tensor):
            outputs = [outputs]
          if self._loss_outputs is not None:
            outputs = [outputs[i] for i in self._loss_outputs]
          batch_loss = loss(outputs, labels, weights)
        if variables is None:
          vars = self.model.trainable_variables
        else:
          vars = variables
        grads = tape.gradient(batch_loss, vars)
        self._tf_optimizer.apply_gradients(zip(grads, vars))
        tf.assign_add(self._global_step, 1)
        current_step = self._global_step.numpy()
      else:

        # In graph mode we execute the training op.

        if train_op is None:
          if loss is None and variables is None:
            train_op = self._train_op
          else:
            if variables is None:
              op_key = (None, loss)
            else:
              op_key = tuple(variables) + (loss,)
            if op_key not in self._custom_train_op:
              if loss is None:
                loss_tensor = self._loss_tensor
              else:
                loss_tensor = loss(
                    [self._output_tensors[i] for i in self._loss_outputs],
                    self._label_placeholders, self._weights_placeholders)
              if variables is None:
                vars = self.model.trainable_variables
              else:
                vars = variables
              self._custom_train_op[op_key] = self._tf_optimizer.minimize(
                  loss_tensor, global_step=self._global_step, var_list=vars)
            train_op = self._custom_train_op[op_key]
        fetches = [train_op, self._loss_tensor, self._global_step]
        feed_dict = dict(zip(self._input_placeholders, inputs))
        feed_dict.update(dict(zip(self._label_placeholders, labels)))
        feed_dict.update(dict(zip(self._weights_placeholders, weights)))
        fetched_values = self.session.run(fetches, feed_dict=feed_dict)
        batch_loss = fetched_values[1]
        current_step = fetched_values[2]

      avg_loss += batch_loss

      # Report progress and write checkpoints.
      averaged_batches += 1
      should_log = (current_step % self.tensorboard_log_frequency == 0)
      if should_log:
        avg_loss = float(avg_loss) / averaged_batches
        logger.info(
            'Ending global_step %d: Average loss %g' % (current_step, avg_loss))
        avg_loss = 0.0
        averaged_batches = 0

      if checkpoint_interval > 0 and current_step % checkpoint_interval == checkpoint_interval - 1:
        self._exec_with_session(lambda: manager.save())
      for c in callbacks:
        c(self, current_step)
      if self.tensorboard and should_log:
        self._log_value_to_tensorboard(tag='loss', simple_value=batch_loss)
        self._summary_writer.reopen()
        self._summary_writer.add_summary(self._current_summary, current_step)
        self._summary_writer.close()

    # Report final results.
    if averaged_batches > 0:
      avg_loss = float(avg_loss) / averaged_batches
      logger.info(
          'Ending global_step %d: Average loss %g' % (current_step, avg_loss))

    if checkpoint_interval > 0:
      self._exec_with_session(lambda: manager.save())

    time2 = time.time()
    logger.info("TIMING: model fitting took %0.3f s" % (time2 - time1))
    return avg_loss

  def _log_value_to_tensorboard(self, **kwargs):
    """This can be called during fitting to log a value to Tensorboard.

    Any keyword arguments passed to this method are passed on to summary.value.add().
    """
    if self._current_summary is None:
      self._current_summary = tf.Summary()
    self._current_summary.value.add(**kwargs)

  def fit_on_batch(self, X, y, w, variables=None, loss=None, callbacks=[]):
    """Perform a single step of training.

    Parameters
    ----------
    X: ndarray
      the inputs for the batch
    y: ndarray
      the labels for the batch
    w: ndarray
      the weights for the batch
    variables: list of tf.Variable
      the variables to train.  If None (the default), all trainable variables in
      the model are used.
    loss: function
      a function of the form f(outputs, labels, weights) that computes the loss
      for each batch.  If None (the default), the model's standard loss function
      is used.
    callbacks: function or list of functions
      one or more functions of the form f(model, step) that will be invoked after
      every step.  This can be used to perform validation, logging, etc.
   """
    if not self.built:
      self.build()
    dataset = NumpyDataset(X, y, w)
    return self.fit(
        dataset,
        nb_epoch=1,
        variables=variables,
        loss=loss,
        callbacks=callbacks)

  def _predict(self, generator, transformers, outputs, uncertainty):
    """
    Predict outputs for data provided by a generator.

    This is the private implementation of prediction.  Do not call it directly.
    Instead call one of the public prediction methods.

    Parameters
    ----------
    generator: generator
      this should generate batches, each represented as a tuple of the form
      (inputs, labels, weights).
    transformers: list of dc.trans.Transformers
      Transformers that the input data has been transformed by.  The output
      is passed through these transformers to undo the transformations.
    outputs: Tensor or list of Tensors
      The outputs to return.  If this is None, the model's standard prediction
      outputs will be returned.  Alternatively one or more Tensors within the
      model may be specified, in which case the output of those Tensors will be
      returned.
    uncertainty: bool
      specifies whether this is being called as part of estimating uncertainty.
      If True, it sets the training flag so that dropout will be enabled, and
      returns the values of the uncertainty outputs.
    Returns:
      a NumPy array of the model produces a single output, or a list of arrays
      if it produces multiple outputs
    """
    results = None
    variances = None
    if uncertainty:
      assert outputs is None
      if self._variance_outputs is None or len(self._variance_outputs) == 0:
        raise ValueError('This model cannot compute uncertainties')
      if len(self._variance_outputs) != len(self._prediction_outputs):
        raise ValueError(
            'The number of variances must exactly match the number of outputs')
    if tf.executing_eagerly() and outputs is not None and len(
        self.model.inputs) == 0:
      raise ValueError(
          "Cannot use 'outputs' argument in eager mode with a model that does not specify its inputs"
      )
    if isinstance(outputs, tf.Tensor):
      outputs = [outputs]
    for batch in generator:
      inputs, labels, weights = batch
      self._create_inputs(inputs)
      inputs, _, _ = self._prepare_batch((inputs, None, None))
      if tf.executing_eagerly():

        # In eager mode we invoke the model directly.

        if len(inputs) == 1:
          inputs = inputs[0]
        if outputs is not None:
          outputs = tuple(outputs)
          if outputs not in self._output_functions:
            self._output_functions[outputs] = tf.keras.backend.function(
                self.model.inputs, outputs)
          output_values = self._output_functions[outputs](inputs)
        else:
          output_values = self.model(inputs, training=False)
          if isinstance(output_values, tf.Tensor):
            output_values = [output_values]
          output_values = [t.numpy() for t in output_values]
      else:

        # In graph mode we execute the output tensors.

        if outputs is not None:
          fetches = outputs
        else:
          fetches = self._output_tensors
        feed_dict = dict(zip(self._input_placeholders, inputs))
        output_values = self.session.run(fetches, feed_dict=feed_dict)

      # Apply tranformers and record results.

      if uncertainty:
        var = [output_values[i] for i in self._variance_outputs]
        if variances is None:
          variances = [var]
        else:
          for i, t in enumerate(var):
            variances[i].append(t)
      if self._prediction_outputs is not None:
        output_values = [output_values[i] for i in self._prediction_outputs]
      if len(transformers) > 0:
        if len(output_values) > 1:
          raise ValueError(
              "predict() does not support Transformers for models with multiple outputs."
          )
        elif len(output_values) == 1:
          output_values = [undo_transforms(output_values[0], transformers)]
      if results is None:
        results = [[] for i in range(len(output_values))]
      for i, t in enumerate(output_values):
        results[i].append(t)

    # Concatenate arrays to create the final results.

    final_results = []
    final_variances = []
    for r in results:
      final_results.append(np.concatenate(r, axis=0))
    if uncertainty:
      for v in variances:
        final_variances.append(np.concatenate(v, axis=0))
      return zip(final_results, final_variances)
    # If only one output, just return array
    if len(final_results) == 1:
      return final_results[0]
    else:
      return final_results

  def predict_on_generator(self, generator, transformers=[], outputs=None):
    """
    Parameters
    ----------
    generator: generator
      this should generate batches, each represented as a tuple of the form
      (inputs, labels, weights).
    transformers: list of dc.trans.Transformers
      Transformers that the input data has been transformed by.  The output
      is passed through these transformers to undo the transformations.
    outputs: Tensor or list of Tensors
      The outputs to return.  If this is None, the model's standard prediction
      outputs will be returned.  Alternatively one or more Tensors within the
      model may be specified, in which case the output of those Tensors will be
      returned.
    Returns:
      a NumPy array of the model produces a single output, or a list of arrays
      if it produces multiple outputs
    """
    return self._predict(generator, transformers, outputs, False)

  def predict_on_batch(self, X, transformers=[], outputs=None):
    """Generates predictions for input samples, processing samples in a batch.

    Parameters
    ----------
    X: ndarray
      the input data, as a Numpy array.
    transformers: list of dc.trans.Transformers
      Transformers that the input data has been transformed by.  The output
      is passed through these transformers to undo the transformations.
    outputs: Tensor or list of Tensors
      The outputs to return.  If this is None, the model's standard prediction
      outputs will be returned.  Alternatively one or more Tensors within the
      model may be specified, in which case the output of those Tensors will be
      returned.

    Returns
    -------
    a NumPy array of the model produces a single output, or a list of arrays
    if it produces multiple outputs
    """
    dataset = NumpyDataset(X=X, y=None)
    return self.predict(dataset, transformers, outputs)

  def predict_uncertainty_on_batch(self, X, masks=50):
    """
    Predict the model's outputs, along with the uncertainty in each one.

    The uncertainty is computed as described in https://arxiv.org/abs/1703.04977.
    It involves repeating the prediction many times with different dropout masks.
    The prediction is computed as the average over all the predictions.  The
    uncertainty includes both the variation among the predicted values (epistemic
    uncertainty) and the model's own estimates for how well it fits the data
    (aleatoric uncertainty).  Not all models support uncertainty prediction.

    Parameters
    ----------
    X: ndarray
      the input data, as a Numpy array.
    masks: int
      the number of dropout masks to average over

    Returns
    -------
    for each output, a tuple (y_pred, y_std) where y_pred is the predicted
    value of the output, and each element of y_std estimates the standard
    deviation of the corresponding element of y_pred
    """
    dataset = NumpyDataset(X=X, y=None)
    return self.predict_uncertainty(dataset, masks)

  def predict(self, dataset, transformers=[], outputs=None):
    """
    Uses self to make predictions on provided Dataset object.

    Parameters
    ----------
    dataset: dc.data.Dataset
      Dataset to make prediction on
    transformers: list of dc.trans.Transformers
      Transformers that the input data has been transformed by.  The output
      is passed through these transformers to undo the transformations.
    outputs: Tensor or list of Tensors
      The outputs to return.  If this is None, the model's standard prediction
      outputs will be returned.  Alternatively one or more Tensors within the
      model may be specified, in which case the output of those Tensors will be
      returned.

    Returns
    -------
    a NumPy array of the model produces a single output, or a list of arrays
    if it produces multiple outputs
    """
    generator = self.default_generator(
        dataset, mode='predict', pad_batches=False)
    return self.predict_on_generator(generator, transformers, outputs)

  def predict_uncertainty(self, dataset, masks=50):
    """
    Predict the model's outputs, along with the uncertainty in each one.

    The uncertainty is computed as described in https://arxiv.org/abs/1703.04977.
    It involves repeating the prediction many times with different dropout masks.
    The prediction is computed as the average over all the predictions.  The
    uncertainty includes both the variation among the predicted values (epistemic
    uncertainty) and the model's own estimates for how well it fits the data
    (aleatoric uncertainty).  Not all models support uncertainty prediction.

    Parameters
    ----------
    dataset: dc.data.Dataset
      Dataset to make prediction on
    masks: int
      the number of dropout masks to average over

    Returns
    -------
    for each output, a tuple (y_pred, y_std) where y_pred is the predicted
    value of the output, and each element of y_std estimates the standard
    deviation of the corresponding element of y_pred
    """
    sum_pred = []
    sum_sq_pred = []
    sum_var = []
    for i in range(masks):
      generator = self.default_generator(
          dataset, mode='uncertainty', pad_batches=False)
      results = self._predict(generator, [], None, True)
      if len(sum_pred) == 0:
        for p, v in results:
          sum_pred.append(p)
          sum_sq_pred.append(p * p)
          sum_var.append(v)
      else:
        for j, (p, v) in enumerate(results):
          sum_pred[j] += p
          sum_sq_pred[j] += p * p
          sum_var[j] += v
    output = []
    std = []
    for i in range(len(sum_pred)):
      p = sum_pred[i] / masks
      output.append(p)
      std.append(np.sqrt(sum_sq_pred[i] / masks - p * p + sum_var[i] / masks))
    if len(output) == 1:
      return (output[0], std[0])
    else:
      return zip(output, std)

  def evaluate_generator(self,
                         generator,
                         metrics,
                         transformers=[],
                         per_task_metrics=False):
    """Evaluate the performance of this model on the data produced by a generator.

    Parameters
    ----------
    generator: generator
      this should generate batches, each represented as a tuple of the form
      (inputs, labels, weights).
    metric: deepchem.metrics.Metric
      Evaluation metric
    transformers: list of dc.trans.Transformers
      Transformers that the input data has been transformed by.  The output
      is passed through these transformers to undo the transformations.
    per_task_metrics: bool
      If True, return per-task scores.

    Returns
    -------
    dict
      Maps tasks to scores under metric.
    """
    evaluator = GeneratorEvaluator(self, generator, transformers)
    return evaluator.compute_model_performance(metrics, per_task_metrics)

  def compute_saliency(self, X):
    """Compute the saliency map for an input sample.

    This computes the Jacobian matrix with the derivative of each output element
    with respect to each input element.  More precisely,

    - If this model has a single output, it returns a matrix of shape
      (output_shape, input_shape) with the derivatives.
    - If this model has multiple outputs, it returns a list of matrices, one
      for each output.

    This method cannot be used on models that take multiple inputs.

    Parameters
    ----------
    X: ndarray
      the input data for a single sample

    Returns
    -------
    the Jacobian matrix, or a list of matrices
    """
    input_shape = X.shape
    X = np.reshape(X, [1] + list(X.shape))
    self._create_inputs([X])
    X, _, _ = self._prepare_batch(([X], None, None))
    if tf.executing_eagerly():
      # In eager mode we use a GradientTape to compute gradients.

      X = tf.constant(X[0])
      with tf.GradientTape(
          persistent=True, watch_accessed_variables=False) as tape:
        tape.watch(X)
        outputs = self.model(X)
        if isinstance(outputs, tf.Tensor):
          outputs = [outputs]
        final_result = []
        for output in outputs:
          output_shape = tuple(output.shape.as_list()[1:])
          output = tf.reshape(output, [-1])
          result = []
          for i in range(output.shape[0]):
            result.append(tape.gradient(output[i], X))
          final_result.append(
              tf.reshape(tf.stack(result), output_shape + input_shape).numpy())
    else:
      # In graph mode we use tf.gradients().

      def jacobian(y, x):
        # Adapted from https://github.com/tensorflow/tensorflow/issues/675#issuecomment-319891923.
        y = tf.reshape(tf.convert_to_tensor(y)[0], [-1])
        n = y.shape[0]
        loop_vars = [
            tf.constant(0, tf.int32),
            tf.TensorArray(tf.float32, size=n)
        ]
        _, jacobian = tf.while_loop(
            lambda j, _: j < n,
            lambda j, result: (j + 1, result.write(j, tf.gradients(y[j], x))),
            loop_vars)
        return jacobian.stack()

      grads = [
          jacobian(self._output_tensors[i], self._input_placeholders[0])
          for i in self._prediction_outputs
      ]
      feed_dict = {self._input_placeholders[0]: X[0]}
      result = self.session.run(grads, feed_dict=feed_dict)
      output_shapes = [
          tuple(o.shape.as_list()[1:]) for o in self._output_tensors
      ]
      final_result = [
          x.reshape(s + input_shape) for x, s in zip(result, output_shapes)
      ]
    if len(final_result) == 1:
      return final_result[0]
    return final_result

  def _prepare_batch(self, batch):
    inputs, labels, weights = batch
    inputs = [
        x if x.dtype == t else x.astype(t)
        for x, t in zip(inputs, self._input_dtypes)
    ]
    if labels is not None:
      labels = [
          x if x.dtype == t else x.astype(t)
          for x, t in zip(labels, self._label_dtypes)
      ]
    if weights is not None:
      weights = [
          x if x.dtype == t else x.astype(t)
          for x, t in zip(weights, self._weights_dtypes)
      ]
    for i in range(len(inputs)):
      shape = inputs[i].shape
      dims = len(shape)
      expected_dims = len(self._input_shapes[i])
      if dims < expected_dims:
        inputs[i] = inputs[i].reshape(shape + (1,) * (expected_dims - dims))
      elif dims > expected_dims and all(d == 1 for d in shape[expected_dims:]):
        inputs[i] = inputs[i].reshape(shape[:expected_dims])
    return (inputs, labels, weights)

  def default_generator(self,
                        dataset,
                        epochs=1,
                        mode='fit',
                        deterministic=True,
                        pad_batches=True):
    """Create a generator that iterates batches for a dataset.

    Subclasses may override this method to customize how model inputs are
    generated from the data.

    Parameters
    ----------
    dataset: Dataset
      the data to iterate
    epochs: int
      the number of times to iterate over the full dataset
    mode: str
      allowed values are 'fit' (called during training), 'predict' (called
      during prediction), and 'uncertainty' (called during uncertainty
      prediction)
    deterministic: bool
      whether to iterate over the dataset in order, or randomly shuffle the
      data for each epoch
    pad_batches: bool
      whether to pad each batch up to this model's preferred batch size

    Returns
    -------
    a generator that iterates batches, each represented as a tuple of lists:
    ([inputs], [outputs], [weights])
    """
    for epoch in range(epochs):
      for (X_b, y_b, w_b, ids_b) in dataset.iterbatches(
          batch_size=self.batch_size,
          deterministic=deterministic,
          pad_batches=pad_batches):
        yield ([X_b], [y_b], [w_b])

  def save_checkpoint(self, max_checkpoints_to_keep=5, model_dir=None):
    """Save a checkpoint to disk.

    Usually you do not need to call this method, since fit() saves checkpoints
    automatically.  If you have disabled automatic checkpointing during fitting,
    this can be called to manually write checkpoints.

    Parameters
    ----------
    max_checkpoints_to_keep: int
      the maximum number of checkpoints to keep.  Older checkpoints are discarded.
    model_dir: str, default None
      Model directory to save checkpoint to. If None, revert to self.model_dir
    """
    self._ensure_built()
    if model_dir is None:
      model_dir = self.model_dir
    if not os.path.exists(model_dir):
      os.makedirs(model_dir)
    manager = tf.train.CheckpointManager(self._checkpoint, model_dir,
                                         max_checkpoints_to_keep)
    self._exec_with_session(lambda: manager.save())

  def _exec_with_session(self, f):
    if tf.executing_eagerly():
      f()
    else:
      with self.session.as_default():
        f()

  def get_checkpoints(self, model_dir=None):
    """Get a list of all available checkpoint files.

    Parameters
    ----------
    model_dir: str, default None
      Directory to get list of checkpoints from. Reverts to self.model_dir if None

    """
    if model_dir is None:
      model_dir = self.model_dir
    return tf.train.get_checkpoint_state(model_dir).all_model_checkpoint_paths

  def restore(self, checkpoint=None, model_dir=None, session=None):
    """Reload the values of all variables from a checkpoint file.

    Parameters
    ----------
    checkpoint: str
      the path to the checkpoint file to load.  If this is None, the most recent
      checkpoint will be chosen automatically.  Call get_checkpoints() to get a
      list of all available checkpoints.
    model_dir: str, default None
      Directory to restore checkpoint from. If None, use self.model_dir.
    session: tf.Session(), default None
      Session to run restore ops under. If None, self.session is used.
    """
    self._ensure_built()
    if model_dir is None:
      model_dir = self.model_dir
    if checkpoint is None:
      checkpoint = tf.train.latest_checkpoint(model_dir)
    if checkpoint is None:
      raise ValueError('No checkpoint found')
    if tf.executing_eagerly():
      self._checkpoint.restore(checkpoint)
    else:
      if session is None:
        session = self.session
      self._checkpoint.restore(checkpoint).run_restore_ops(session)

  def get_global_step(self):
    """Get the number of steps of fitting that have been performed."""
    if tf.executing_eagerly():
      return int(self._global_step)
    return self._global_step.eval(session=self.session)

  def _create_assignment_map(self, source_model, include_top=True, **kwargs):
    """
    Creates a default assignment map between variables of source and current model.
    This is used only when a custom assignment map is missing. This assumes the
    model is made of different layers followed by a dense layer for mapping to
    output tasks. include_top is used to control whether or not the final dense
    layer is used. The default assignment map is useful in cases where the type
    of task is different (classification vs regression) and/or number of tasks.

    Parameters
    ----------
    source_model: dc.models.KerasModel
        Source model to copy variable values from.
    include_top: bool, default True
        if true, copies the last dense layer
    """
    assignment_map = {}
    source_vars = source_model.model.trainable_variables
    dest_vars = self.model.trainable_variables

    if not include_top:
      source_vars = source_vars[:-2]
      dest_vars = dest_vars[:-2]

    for source_var, dest_var in zip(source_vars, dest_vars):
      assignment_map[source_var] = dest_var

    return assignment_map

  def _create_value_map(self, source_model, **kwargs):
    """
    Creates a value map between variables in the source model and their
    current values. This is used only when a custom value map is missing, and
    assumes the restore method has been called under self.session.

    Parameters
    ----------
    source_model: dc.models.KerasModel
        Source model to create value map from
    """
    value_map = {}
    source_vars = source_model.model.trainable_variables

    if tf.executing_eagerly():
      for source_var in source_vars:
        value_map[source_var] = source_var.numpy()
    else:
      for source_var in source_vars:
        # self.session is used because restore was called in the same session
        value_map[source_var] = source_var.eval(session=self.session)

    return value_map

  def load_from_pretrained(self,
                           source_model,
                           assignment_map=None,
                           value_map=None,
                           checkpoint=None,
                           model_dir=None,
                           include_top=True,
                           **kwargs):
    """Copies variable values from a pretrained model. `source_model` can either
    be a pretrained model or a model with the same architecture. `value_map`
    is a variable-value dictionary. If no `value_map` is provided, the variable
    values are restored to the `source_model` from a checkpoint and a default
    `value_map` is created. `assignment_map` is a dictionary mapping variables
    from the `source_model` to the current model. If no `assignment_map` is
    provided, one is made from scratch and assumes the model is composed of
    several different layers, with the final one being a dense layer. include_top
    is used to control whether or not the final dense layer is used. The default
    assignment map is useful in cases where the type of task is different
    (classification vs regression) and/or number of tasks in the setting.

    Parameters
    ----------
    source_model: dc.KerasModel, required
      source_model can either be the pretrained model or a dc.KerasModel with
      the same architecture as the pretrained model. It is used to restore from
      a checkpoint, if value_map is None and to create a default assignment map
      if assignment_map is None
    assignment_map: Dict, default None
      Dictionary mapping the source_model variables and current model variables
    value_map: Dict, default None
      Dictionary containing source_model trainable variables mapped to numpy
      arrays. If value_map is None, the values are restored and a default
      variable map is created using the restored values
    checkpoint: str, default None
      the path to the checkpoint file to load.  If this is None, the most recent
      checkpoint will be chosen automatically.  Call get_checkpoints() to get a
      list of all available checkpoints
    model_dir: str, default None
      Restore model from custom model directory if needed
    include_top: bool, default True
        if True, copies the weights and bias associated with the final dense
        layer. Used only when assignment map is None
    """

    self._ensure_built()
    if value_map is None:
      logger.info(
          "No value map provided. Creating default value map from restored model."
      )
      if tf.executing_eagerly():
        source_model.restore(model_dir=model_dir, checkpoint=checkpoint)
      else:
        source_model.restore(
            model_dir=model_dir, checkpoint=checkpoint, session=self.session)
      value_map = self._create_value_map(source_model=source_model)

    if assignment_map is None:
      logger.info("No assignment map provided. Creating custom assignment map.")
      assignment_map = self._create_assignment_map(
          source_model=source_model, include_top=include_top)

    if tf.executing_eagerly():
      for source_var, dest_var in assignment_map.items():
        assert source_var.shape == dest_var.shape
        dest_var.assign(value_map[source_var])

    else:
      with self.session.as_default():
        for source_var, dest_var in assignment_map.items():
          assert source_var.shape == dest_var.shape
          assign_op = dest_var.assign(value_map[source_var])
          self.session.run(assign_op)

    dest_vars = list(assignment_map.values())
    self._initialized_vars.update(set(dest_vars))


class _StandardLoss(object):
  """The implements the loss function for models that use a dc.models.losses.Loss."""

  def __init__(self, model, loss):
    self.model = model
    self.loss = loss

  def __call__(self, outputs, labels, weights):
    if len(outputs) != 1 or len(labels) != 1 or len(weights) != 1:
      raise ValueError(
          "Loss functions expects exactly one each of outputs, labels, and weights"
      )
    losses = self.loss(outputs[0], labels[0])
    w = weights[0]
    if len(w.shape) < len(losses.shape):
      if isinstance(w, tf.Tensor):
        shape = tuple(w.shape.as_list())
      else:
        shape = w.shape
      shape = tuple(-1 if x is None else x for x in shape)
      w = tf.reshape(w, shape + (1,) * (len(losses.shape) - len(w.shape)))

    loss = losses * w
    return tf.reduce_mean(loss) + sum(self.model.losses)
