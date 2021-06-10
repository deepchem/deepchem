import numpy as np
import time
import logging
import os

from deepchem.data import Dataset, NumpyDataset
from deepchem.models.models import Model

from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
from deepchem.utils.typing import ArrayLike, LossFn, OneOrMany

# JAX depend
import jax.numpy as jnp
import jax
from jax import random
import haiku as hk
import optax

logger = logging.getLogger(__name__)


class JaxModel(Model):
  """This is a DeepChem model implemented by a Jax Model
    """

  def __init__(self,
               model,
               params: hk.Params,
               loss,
               batch_size: int,
               learning_rate: float,
               optimizer,
               optimizer_state,
               log_frequency: int = 100):
    """
        model = hk.without_apply_rng(hk.transform(f))
        rng = jax.random.PRNGKey(500)
        params = model.init(rng, x)

        pass model.apply for model here
        """

    self.loss = lambda pred, tar: jnp.mean(optax.l2_loss(pred, tar))
    self.batch_size = batch_size

    self.optimizer = optax.adam(learning_rate)
    self.model = model  # this is a function, hk.apply
    self.params = params
    self._built = False
    self.log_frequency = log_frequency

  def _ensure_built(self):
    if self._built:
      return

    self._built = True
    self._global_step = 0
    self.opt_state = self.optimizer.init(self.params)

  def fit(
      self,
      dataset: Dataset,
      nb_epochs: int = 10,
      deterministic: bool = False,
  ):
    return self.fit_generator(
        self.default_generator(
            dataset, epochs=nb_epochs, deterministic=deterministic))

  def fit_generator(
      self,
      generator: Iterable[Tuple[Any, Any, Any]],
  ):
    self._ensure_built()
    avg_loss = 0.0
    last_avg_loss = 0.0
    averaged_batches = 0

    loss = self.loss
    grad_update = self._create_gradient_fn(self.loss, self.optimizer)
    params, opt_state = self._get_trainable_params()

    for batch in generator:
      inputs, labels, weights = self._prepare_batch(batch)

      if isinstance(inputs, list) and len(inputs) == 1:
        inputs = inputs[0]

      if isinstance(labels, list) and len(labels) == 1:
        labels = labels[0]

      if isinstance(weights, list) and len(weights) == 1:
        weights = weights[0]

      params, opt_state, batch_loss = grad_update(params, opt_state, inputs,
                                                  labels)

      avg_loss += jax.device_get(batch_loss)
      self._global_step += 1
      current_step = self._global_step
      averaged_batches += 1
      should_log = (current_step % self.log_frequency == 0)

      if should_log:
        avg_loss = float(avg_loss) / averaged_batches
        logger.info(
            'Ending global_step %d: Average loss %g' % (current_step, avg_loss))
        last_avg_loss = avg_loss
        avg_loss = 0.0
        averaged_batches = 0

    # Report final results.
    if averaged_batches > 0:
      avg_loss = float(avg_loss) / averaged_batches
      logger.info(
          'Ending global_step %d: Average loss %g' % (current_step, avg_loss))
      last_avg_loss = avg_loss

    self._set_trainable_params(params, opt_state)
    return last_avg_loss

  def _get_trainable_params(self):
    return self.params, self.opt_state

  def _set_trainable_params(self, params, opt_state):
    self.params = params
    self.opt_state = opt_state

  def _create_gradient_fn(self, loss, optimizer, p=None):
    """
        This function calls the update function, to implement the backpropogation
        """

    @jax.jit
    def update(params, opt_state, batch, target):
      batch_loss, grads = jax.value_and_grad(loss)(params, batch, target)
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

    for epoch in range(epochs):
      for (X_b, y_b, w_b, ids_b) in dataset.iterbatches(
          batch_size=self.batch_size,
          deterministic=deterministic,
          pad_batches=pad_batches):
        yield ([X_b], [y_b], [w_b])
