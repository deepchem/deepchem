import numpy as np
import time
import logging
try:
  from collections.abc import Sequence as SequenceCollection
except:
  from collections import Sequence as SequenceCollection

from deepchem.data import Dataset
from deepchem.models.models import Model
from deepchem.models.losses import Loss
from deepchem.models.optimizers import Optimizer
from typing import Any, Callable, Iterable, List, Optional, Tuple, Union
from deepchem.utils.typing import LossFn

# JAX depend
import jax.numpy as jnp
import jax
import haiku as hk
import optax

import warnings

logger = logging.getLogger(__name__)


class JaxModel(Model):
  """This is a DeepChem model implemented by a Jax Model

  Here is a simple example of that uses JaxModel to train a
  Haiku (JAX Neural Network Library) based model on deepchem
  dataset.

  >> def f(x):
  >>   net = hk.nets.MLP([512, 256, 128, 1])
  >>   return net(x)
  >> model = hk.without_apply_rng(hk.transform(f))
  >> rng = jax.random.PRNGKey(500)
  >> x, _, _, _ = next(iter(train_dataset.iterbatches(batch_size=256)))
  >> params = model.init(rng, x)
  >> j_m = JaxModel(model, params, 256, 0.001, 100)
  >> j_m.fit(train_dataset)

  All optimizations will be done using the optax library.
  """

  def __init__(self,
               model: hk.State,
               params: hk.Params,
               loss: Union[Loss, LossFn],
               output_types: Optional[List[str]] = None,
               batch_size: int = 100,
               learning_rate: float = 0.001,
               optimizer: Union[optax.GradientTransformation,
                                Optimizer] = optax.adam(1e-3),
               log_frequency: int = 100,
               **kwargs):
    """
    Create a new JaxModel

    Parameters
    ----------
    model: hk.State or Function
      Any Jax based model that has a `apply` method for computing the network. Currently
      only haiku models are supported.
    params: hk.Params
      The parameter of the Jax based networks
    loss: dc.models.losses.Loss or function
      a Loss or function defining how to compute the training loss for each
      batch, as described above
    output_types: list of strings, optional (default None)
      the type of each output from the model, as described above
    batch_size: int, optional (default 100)
      default batch size for training and evaluating
    learning_rate: float or LearningRateSchedule, optional (default 0.001)
      the learning rate to use for fitting.  If optimizer is specified, this is
      ignored.
    optimizer: optax object
      For the time being, it is optax object
    log_frequency: int, optional (default 100)
      The frequency at which to log data. Data is logged using
      `logging` by default.

    Miscellanous Parameters Yet To Add
    ----------------------------------
    model_dir: str, optional (default None)
      Will be added along with the save & load method
    tensorboard: bool, optional (default False)
      whether to log progress to TensorBoard during training
    wandb: bool, optional (default False)
      whether to log progress to Weights & Biases during training

    Work in Progress
    ----------------
    [1] Integrate the optax losses, optimizers, schedulers with Deepchem
    [2] Support for saving & loading the model.
    """
    super(JaxModel, self).__init__(model=model, **kwargs)
    warnings.warn(
        'JaxModel is still in active development and all features may not yet be implemented'
    )
    self._loss_fn = loss  # lambda pred, tar: jnp.mean(optax.l2_loss(pred, tar))
    self.batch_size = batch_size
    self.learning_rate = learning_rate
    self.optimizer = optimizer
    self.model = model
    self.params = params
    self._built = False
    self.log_frequency = log_frequency

    if output_types is None:
      self._prediction_outputs = None
      self._loss_outputs = None
      self._variance_outputs = None
      self._other_outputs = None
    else:
      self._prediction_outputs = []
      self._loss_outputs = []
      self._variance_outputs = []
      self._other_outputs = []
      for i, type in enumerate(output_types):
        if type == 'prediction':
          self._prediction_outputs.append(i)
        elif type == 'loss':
          self._loss_outputs.append(i)
        elif type == 'variance':
          self._variance_outputs.append(i)
        else:
          self._other_outputs.append(i)
      if len(self._loss_outputs) == 0:
        self._loss_outputs = self._prediction_outputs

  def _ensure_built(self):
    """The first time this is called, create internal data structures.

    Work in Progress
    ----------------
    [1] Integerate the optax losses, optimizers, schedulers with Deepchem

    """
    if self._built:
      return

    self._built = True
    self._global_step = 0
    self.opt_state = self.optimizer.init(self.params)

  def fit(self,
          dataset: Dataset,
          nb_epochs: int = 10,
          deterministic: bool = False,
          loss: Union[Loss, LossFn] = None,
          callbacks: Union[Callable, List[Callable]] = [],
          all_losses: Optional[List[float]] = None) -> float:
    """Train this model on a dataset.

    Parameters
    ----------
    dataset: Dataset
      the Dataset to train on
    nb_epoch: int
      the number of epochs to train for
    deterministic: bool
      if True, the samples are processed in order.  If False, a different random
      order is used for each epoch.
    loss: function
      a function of the form f(outputs, labels, weights) that computes the loss
      for each batch.  If None (the default), the model's standard loss function
      is used.
    callbacks: function or list of functions
      one or more functions of the form f(model, step) that will be invoked after
      every step.  This can be used to perform validation, logging, etc.
    all_losses: Optional[List[float]], optional (default None)
      If specified, all logged losses are appended into this list. Note that
      you can call `fit()` repeatedly with the same list and losses will
      continue to be appended.

    Returns
    -------
    The average loss over the most recent checkpoint interval

    Miscellanous Parameters Yet To Add
    ----------------------------------
    max_checkpoints_to_keep: int
      the maximum number of checkpoints to keep.  Older checkpoints are discarded.
    checkpoint_interval: int
      the frequency at which to write checkpoints, measured in training steps.
      Set this to 0 to disable automatic checkpointing.
    restore: bool
      if True, restore the model from the most recent checkpoint and continue training
      from there.  If False, retrain the model from scratch.
    variables: list of torch.nn.Parameter
      the variables to train.  If None (the default), all trainable variables in
      the model are used.

    Work in Progress
    ----------------
    [1] Integerate the optax losses, optimizers, schedulers with Deepchem
    [2] Support for saving & loading the model.
    [3] Adding support for output types (choosing only self._loss_outputs)
   """
    return self.fit_generator(
        self.default_generator(
            dataset, epochs=nb_epochs, deterministic=deterministic), loss,
        callbacks, all_losses)

  def fit_generator(self,
                    generator: Iterable[Tuple[Any, Any, Any]],
                    loss: Union[Loss, LossFn] = None,
                    callbacks: Union[Callable, List[Callable]] = [],
                    all_losses: Optional[List[float]] = None) -> float:
    if not isinstance(callbacks, SequenceCollection):
      callbacks = [callbacks]
    self._ensure_built()
    avg_loss = 0.0
    last_avg_loss = 0.0
    averaged_batches = 0
    if loss is None:
      loss = self._loss_fn
    grad_update = self._create_gradient_fn(loss, self.model, self.optimizer,
                                           self._loss_outputs)
    params, opt_state = self._get_trainable_params()
    time1 = time.time()

    # Main training loop

    for batch in generator:
      inputs, labels, weights = self._prepare_batch(batch)

      if isinstance(inputs, list) and len(inputs) == 1:
        inputs = inputs[0]

      if isinstance(labels, list) and len(labels) == 1:
        labels = labels[0]

      if isinstance(weights, list) and len(weights) == 1:
        weights = weights[0]

      params, opt_state, batch_loss = grad_update(params, opt_state, inputs,
                                                  labels, weights)

      avg_loss += jax.device_get(batch_loss)
      self._global_step += 1
      current_step = self._global_step
      averaged_batches += 1
      should_log = (current_step % self.log_frequency == 0)

      if should_log:
        avg_loss = float(avg_loss) / averaged_batches
        logger.info(
            'Ending global_step %d: Average loss %g' % (current_step, avg_loss))
        if all_losses is not None:
          all_losses.append(avg_loss)
        # Capture the last avg_loss in case of return since we're resetting to 0 now
        last_avg_loss = avg_loss
        avg_loss = 0.0
        averaged_batches = 0
      for c in callbacks:
        c(self, current_step)

    # Report final results.
    if averaged_batches > 0:
      avg_loss = float(avg_loss) / averaged_batches
      logger.info(
          'Ending global_step %d: Average loss %g' % (current_step, avg_loss))
      if all_losses is not None:
        all_losses.append(avg_loss)
      last_avg_loss = avg_loss

    time2 = time.time()
    logger.info("TIMING: model fitting took %0.3f s" % (time2 - time1))
    self._set_trainable_params(params, opt_state)
    return last_avg_loss

  def _get_trainable_params(self):
    """
    Will be used to seperate freezing parameters while transfer learning
    """
    return self.params, self.opt_state

  def _set_trainable_params(self, params: hk.Params, opt_state: optax.OptState):
    """
    A functional approach to setting the final parameters after training
    """
    self.params = params
    self.opt_state = opt_state

  def _create_gradient_fn(self, loss, model, optimizer, loss_outputs):
    """
    This function calls the update function, to implement the backpropogation
    """

    @jax.jit
    def model_loss(params, batch, target, weights):
      predict = model.apply(params, batch)
      if loss_outputs is not None:
        predict = [predict[i] for i in loss_outputs]
      return loss(predict, target, weights)

    @jax.jit
    def update(params, opt_state, batch, target,
               weights) -> Tuple[hk.Params, optax.OptState, jnp.ndarray]:
      batch_loss, grads = jax.value_and_grad(model_loss)(params, batch, target,
                                                         weights)
      updates, opt_state = optimizer.update(grads, opt_state)
      new_params = optax.apply_updates(params, updates)
      return new_params, opt_state, batch_loss

    return update

  def _prepare_batch(self, batch):
    inputs, labels, weights = batch
    inputs = [
        x.astype(np.float32) if x.dtype == np.float64 else x for x in inputs
    ]
    if labels is not None:
      labels = [
          x.astype(np.float32) if x.dtype == np.float64 else x for x in labels
      ]
    else:
      labels = []

    if weights is not None:
      weights = [
          x.astype(np.float32) if x.dtype == np.float64 else x for x in weights
      ]
    else:
      weights = []

    return (inputs, labels, weights)

  def default_generator(
      self,
      dataset: Dataset,
      epochs: int = 1,
      mode: str = 'fit',
      deterministic: bool = True,
      pad_batches: bool = True) -> Iterable[Tuple[List, List, List]]:
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
      for (X_b, y_b, w_b, _) in dataset.iterbatches(
          batch_size=self.batch_size,
          deterministic=deterministic,
          pad_batches=pad_batches):
        yield ([X_b], [y_b], [w_b])
