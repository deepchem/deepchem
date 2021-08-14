'''
This python consists of diffrent variations of PINNs model using the JaxModel API
'''
import numpy as np
import time
from typing import Any, Callable, Iterable, List, Optional, Tuple, Union

try:
  from collections.abc import Sequence as SequenceCollection
except:
  from collections import Sequence as SequenceCollection

from deepchem.data import Dataset
from deepchem.models.losses import Loss
from deepchem.models.optimizers import Optimizer
from deepchem.utils.typing import LossFn, OneOrMany
from deepchem.trans.transformers import Transformer, undo_transforms

# JAX dependencies
import jax
import jax.numpy as jnp
import haiku as hk
import optax
from deepchem.models import JaxModel
from deepchem.models.jax_models.jax_model import create_default_gradient_fn, create_default_eval_fn

import logging

logger = logging.getLogger(__name__)


def create_default_update_fn(optimizer, model_loss):
  """
  This function calls the update function, to implement the backpropogation
  """

  @jax.jit
  def update(params, opt_state, batch, target, weights,
             rng) -> Tuple[hk.Params, optax.OptState, jnp.ndarray]:
    batch_loss, grads = jax.value_and_grad(model_loss)(params, target, weights, rng, *batch)
    updates, opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, batch_loss

  return update


class PINN_Model(JaxModel):
  def __init__(self,
               forward_fn: hk.State,
               params: hk.Params,
               boundary_data: dict = {},
               output_types: Optional[List[str]] = None,
               batch_size: int = 100,
               learning_rate: float = 0.001,
               optimizer: Union[optax.GradientTransformation,
                                Optimizer] = optax.adam(1e-3),
               grad_fn: Callable = create_default_gradient_fn,
               update_fn: Callable = create_default_update_fn,
               eval_fn: Callable = create_default_eval_fn,
               rng=jax.random.PRNGKey(1),
               log_frequency: int = 100,
               **kwargs):

    self.boundary_data = boundary_data
    super(PINN_Model, self).__init__(
        forward_fn, params, None, output_types, batch_size, learning_rate,
        optimizer, grad_fn, update_fn, eval_fn, rng, log_frequency, **kwargs
    )

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
    model_loss_fn = self._create_gradient_fn(self.forward_fn,
                                             self._loss_outputs, self.boundary_data)
    grad_update = self._create_update_fn(self.optimizer, model_loss_fn)

    params, opt_state = self._get_trainable_params()
    rng = self.rng
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

      params, opt_state, batch_loss = grad_update(
          params, opt_state, inputs, labels, weights, rng=rng)
      rng, _ = jax.random.split(rng)
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

  def _prepare_batch(self, batch):
    inputs, labels, weights = batch
    inputs = [
        x.astype(np.float32) if x.dtype == np.float64 else x for x in inputs
    ]
    inputs = [
        np.split(x, x.shape[1], 1) for x in inputs
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

  def _predict(
      self, generator: Iterable[Tuple[Any, Any, Any]],
      transformers: List[Transformer], uncertainty: bool,
      other_output_types: Optional[OneOrMany[str]]) -> OneOrMany[np.ndarray]:
    """
    Predict outputs for data provided by a generator.
    This is the private implementation of prediction.  Do not
    call it directly. Instead call one of the public prediction
    methods.
    Parameters
    ----------
    generator: generator
      this should generate batches, each represented as a tuple of the form
      (inputs, labels, weights).
    transformers: List[dc.trans.Transformers]
      Transformers that the input data has been transformed by.  The output
      is passed through these transformers to undo the transformations.
    uncertainty: bool
      specifies whether this is being called as part of estimating uncertainty.
      If True, it sets the training flag so that dropout will be enabled, and
      returns the values of the uncertainty outputs.
    other_output_types: list, optional
      Provides a list of other output_types (strings) to predict from model.
    Returns
    -------
    A NumpyArray if the model produces a single output, or a list of arrays otherwise.
    """
    results: Optional[List[List[np.ndarray]]] = None
    variances: Optional[List[List[np.ndarray]]] = None
    if uncertainty and (other_output_types is not None):
      raise ValueError(
          'This model cannot compute uncertainties and other output types simultaneously. Please invoke one at a time.'
      )
    if uncertainty:
      if self._variance_outputs is None or len(self._variance_outputs) == 0:
        raise ValueError('This model cannot compute uncertainties')
      if len(self._variance_outputs) != len(self._prediction_outputs):
        raise ValueError(
            'The number of variances must exactly match the number of outputs')
    if other_output_types:
      if self._other_outputs is None or len(self._other_outputs) == 0:
        raise ValueError(
            'This model cannot compute other outputs since no other output_types were specified.'
        )
    self._ensure_built()
    eval_fn = self._create_eval_fn(self.forward_fn, self.params)
    rng = self.rng

    for batch in generator:
      inputs, _, _ = self._prepare_batch(batch)

      if isinstance(inputs, list) and len(inputs) == 1:
        inputs = inputs[0]

      output_values = eval_fn(*inputs, rng)
      if isinstance(output_values, jnp.ndarray):
        output_values = [output_values]
      output_values = [jax.device_get(t) for t in output_values]

      # Apply tranformers and record results.
      if uncertainty:
        var = [output_values[i] for i in self._variance_outputs]
        if variances is None:
          variances = [var]
        else:
          for i, t in enumerate(var):
            variances[i].append(t)

      access_values = []
      if other_output_types:
        access_values += self._other_outputs
      elif self._prediction_outputs is not None:
        access_values += self._prediction_outputs

      if len(access_values) > 0:
        output_values = [output_values[i] for i in access_values]

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
    if results is not None:
      for r in results:
        final_results.append(np.concatenate(r, axis=0))
    if uncertainty and variances is not None:
      for v in variances:
        final_variances.append(np.concatenate(v, axis=0))
      return zip(final_results, final_variances)
    if len(final_results) == 1:
      return final_results[0]
    else:
      return final_results
