import numpy as np
import tensorflow as tf
import time

from deepchem.data import NumpyDataset
from deepchem.models.models import Model
from deepchem.models.tensorgraph.optimizers import Adam


class KerasModel(Model):
  """This is a DeepChem model implement by a Keras model.

  This class provides several advantages over using the Keras model's fitting
  and prediction methods directly.

  1. It provides better integration with the rest of DeepChem, such as direct
     support for Datasets and Transformers.

  2. It defines the loss in a more flexible way.  In particular, Keras does not
     support multidimensional weight matrices, which makes it impossible to
     implement most multitask models with Keras.

  3. It provides various additional features not found in the Keras Model class,
     such as uncertainty prediction and saliency mapping.
  """

  def __init__(self,
               model,
               loss_fn,
               batch_size=100,
               model_dir=None,
               learning_rate=0.001,
               optimizer=None,
               **kwargs):
    """Create a new KerasModel.

    Parameters
    ----------
    model: tf.keras.Model
      the Keras model implementing the calculation
    loss_fn: function
      a function defining the training loss for a batch.  It must be of the form
      f(inputs, labels, weights), taking the list of inputs to the model, the
      expected outputs, and weight matrices.  It should return a scalar equal to
      the value of the loss function for the batch.  (This is different from the
      loss function used by tf.keras.Model.compile(), which corresponds only to
      a single sample and does not include regularization terms.)
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
    """
    super(KerasModel, self).__init__(
        model_instance=model, model_dir=model_dir, **kwargs)
    self.model = model
    self.loss_fn = loss_fn
    self.batch_size = batch_size
    if optimizer is None:
      self.optimizer = Adam(learning_rate=learning_rate)
    else:
      self.optimizer = optimizer
    self._built = False
    self._training_ops_built = False

  def _ensure_built(self):
    if self._built:
      return
    self._built = True
    if not tf.executing_eagerly():
      self.session = tf.Session()
    self._global_step = tf.Variable(0, trainable=False)
    self._tf_optimizer = self.optimizer._create_optimizer(self._global_step)
    self._checkpoint = tf.train.Checkpoint(
        optimizer=self._tf_optimizer, model=self.model)

  def _create_training_ops(self, example_batch):
    if self._training_ops_built:
      return
    self._ensure_built()
    self._training_ops_built = True
    if tf.executing_eagerly():
      return
    if len(self.model.inputs) > 0:
      self._input_placeholders = self.model.inputs
    else:
      # The model doesn't specify inputs, so guess the input shapes based on the
      # example batch.
      input_shapes = [(None,) + i.shape[1:] for i in example_batch[0]]
      self._input_placeholders = [
          tf.placeholder(dtype=tf.float32, shape=s) for s in input_shapes
      ]
      if len(input_shapes) == 1:
        self.model.build(input_shapes[0])
      else:
        self.model.build(input_shapes)
    self._label_placeholders = [
        tf.placeholder(dtype=tf.float32, shape=t.shape)
        for t in example_batch[1]
    ]
    self._weights_placeholders = [
        tf.placeholder(dtype=tf.float32, shape=t.shape)
        for t in example_batch[2]
    ]
    self._output_tensors = self.model(self._input_placeholders, training=False)
    self._loss_tensor = self.loss_fn(self._input_placeholders,
                                     self._label_placeholders,
                                     self._weights_placeholders)
    try:
      self._train_op = self._tf_optimizer.minimize(
          self._loss_tensor, global_step=self._global_step)
    except ValueError:
      # The loss doesn't depend on any variables.
      self._train_op = 0
    self.session.run(tf.global_variables_initializer())

  def fit(self,
          dataset,
          nb_epoch=10,
          max_checkpoints_to_keep=5,
          checkpoint_interval=1000,
          deterministic=False,
          restore=False):
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
   """
    return self.fit_generator(
        self.default_generator(
            dataset, epochs=nb_epoch, deterministic=deterministic),
        max_checkpoints_to_keep, checkpoint_interval, restore)

  def fit_generator(self,
                    generator,
                    max_checkpoints_to_keep=5,
                    checkpoint_interval=1000,
                    restore=False):
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

    Returns
    -------
    the average loss over the most recent checkpoint interval
    """
    self._ensure_built()
    if restore:
      self.restore()
    if checkpoint_interval > 0:
      manager = tf.train.CheckpointManager(self._checkpoint, self.model_dir,
                                           max_checkpoints_to_keep)
    avg_loss = 0.0
    averaged_batches = 0
    time1 = time.time()

    # Main training loop.

    for batch in generator:
      self._create_training_ops(batch)
      inputs, labels, weights = batch
      if tf.executing_eagerly():

        # In eager mode we execute the loss function, accumulating the gradients.

        with tf.GradientTape() as tape:
          loss = self.loss_fn(inputs, labels, weights)
          avg_loss += loss
          grads = tape.gradient(loss, self.model.trainable_variables)
          self._tf_optimizer.apply_gradients(
              zip(grads, self.model.trainable_variables))
          current_step = self._global_step.numpy()
      else:

        # In graph mode we execute the training op.

        fetches = [self._train_op, self._loss_tensor, self._global_step]
        feed_dict = dict(zip(self._input_placeholders, inputs))
        feed_dict.update(dict(zip(self._label_placeholders, labels)))
        feed_dict.update(dict(zip(self._weights_placeholders, weights)))
        fetched_values = self.session.run(fetches, feed_dict=feed_dict)
        avg_loss += fetched_values[1]
        current_step = fetched_values[2]

      # Report progress and write checkpoints.

      averaged_batches += 1
      if checkpoint_interval > 0 and current_step % checkpoint_interval == checkpoint_interval - 1:
        self._exec_with_session(lambda: manager.save())
        avg_loss = float(avg_loss) / averaged_batches
        print(
            'Ending global_step %d: Average loss %g' % (current_step, avg_loss))
        avg_loss = 0.0
        averaged_batches = 0

    # Report final results.

    if checkpoint_interval > 0:
      if averaged_batches > 0:
        avg_loss = float(avg_loss) / averaged_batches
        print(
            'Ending global_step %d: Average loss %g' % (current_step, avg_loss))
      self._exec_with_session(lambda: manager.save())
    time2 = time.time()
    print("TIMING: model fitting took %0.3f s" % (time2 - time1))
    return avg_loss

  def fit_on_batch(self, X, y, w):
    if not self.built:
      self.build()
    dataset = NumpyDataset(X, y, w)
    return self.fit(dataset, nb_epoch=1)

  def _predict(self, generator, transformers, outputs):
    """
    Predict outputs for data provided by a generator.

    This is the private implementation of prediction.  Do not call it directly.
    Instead call one of the public prediction methods.

    Parameters
    ----------
    generator: generator
      this should generate batches, each represented as a tuple of the form
      (inputs, labels, weights).
    transformers: list
      List of dc.trans.Transformers.
    Returns:
      a NumPy array of the model produces a single output, or a list of arrays
      if it produces multiple outputs
    """
    results = None
    for batch in generator:
      self._create_training_ops(batch)
      inputs, labels, weights = batch
      if tf.executing_eagerly():

        # In eager mode we invoke the model directly.

        outputs = self.model(inputs)
        outputs = [t.numpy() for t in outputs]
      else:

        # In graph mode we execute the output tensors.

        fetches = [self._train_op, self._loss_tensor, self._global_step]
        feed_dict = dict(zip(self._input_placeholders, inputs))
        outputs = self.session.run(self._output_tensors, feed_dict=feed_dict)

      # Apply tranformers and record results.

      if len(transformers) > 0:
        if len(outputs) > 1:
          raise ValueError(
              "predict() does not support Transformers for models with multiple outputs."
          )
        elif len(outputs) == 1:
          outputs = [undo_transforms(outputs[0], transformers)]
      if results is None:
        results = [outputs]
      else:
        for i, t in enumerate(outputs):
          results[i].append(t)

    # Concatenate arrays to create the final results.

    final_results = []
    for result_list in results:
      final_results.append(np.concatenate(result_list, axis=0))
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
    transformers: list
      List of dc.trans.Transformers.
    Returns:
      a NumPy array of the model produces a single output, or a list of arrays
      if it produces multiple outputs
    """
    return self._predict(generator, transformers, outputs)

  def predict_on_batch(self, X, transformers=[], outputs=None):
    """Generates predictions for input samples, processing samples in a batch.

    Parameters
    ----------
    X: ndarray
      the input data, as a Numpy array.
    transformers: List
      List of dc.trans.Transformers

    Returns
    -------
    a NumPy array of the model produces a single output, or a list of arrays
    if it produces multiple outputs
    """
    dataset = NumpyDataset(X=X, y=None)
    return self.predict(dataset, transformers, outputs)

  def predict(self, dataset, transformers=[], outputs=None):
    """
    Uses self to make predictions on provided Dataset object.

    Parameters
    ----------
    dataset: dc.data.Dataset
      Dataset to make prediction on
    transformers: list
      List of dc.trans.Transformers.

    Returns
    -------
    a NumPy array of the model produces a single output, or a list of arrays
    if it produces multiple outputs
    """
    generator = self.default_generator(dataset, predict=True, pad_batches=False)
    return self.predict_on_generator(generator, transformers, outputs)

  def default_generator(self,
                        dataset,
                        epochs=1,
                        predict=False,
                        deterministic=True,
                        pad_batches=True):
    for epoch in range(epochs):
      for (X_b, y_b, w_b, ids_b) in dataset.iterbatches(
          batch_size=self.batch_size,
          deterministic=deterministic,
          pad_batches=pad_batches):
        yield ([X_b], [y_b], [w_b])

  def save_checkpoint(self, max_checkpoints_to_keep=5):
    """Save a checkpoint to disk.

    Usually you do not need to call this method, since fit() saves checkpoints
    automatically.  If you have disabled automatic checkpointing during fitting,
    this can be called to manually write checkpoints.

    Parameters
    ----------
    max_checkpoints_to_keep: int
      the maximum number of checkpoints to keep.  Older checkpoints are discarded.
    """
    self._ensure_built()
    manager = tf.train.CheckpointManager(self._checkpoint, self.model_dir,
                                         max_checkpoints_to_keep)
    self._exec_with_session(lambda: manager.save())

  def _exec_with_session(self, f):
    if tf.executing_eagerly():
      f()
    else:
      with self.session.as_default():
        f()

  def get_checkpoints(self):
    """Get a list of all available checkpoint files."""
    return tf.train.get_checkpoint_state(
        self.model_dir).all_model_checkpoint_paths

  def restore(self, checkpoint=None):
    """Reload the values of all variables from a checkpoint file.

    Parameters
    ----------
    checkpoint: str
      the path to the checkpoint file to load.  If this is None, the most recent
      checkpoint will be chosen automatically.  Call get_checkpoints() to get a
      list of all available checkpoints.
    """
    if checkpoint is None:
      checkpoint = tf.train.latest_checkpoint(self.model_dir)
    if checkpoint is None:
      raise ValueError('No checkpoint found')
    if tf.executing_eagerly():
      self.model.load_weights(checkpoint)
    else:
      with self.session.as_default():
        self.model.load_weights(checkpoint)
