"""
Normalizing flows for transforming probability distributions.
"""

import numpy as np
import logging
from typing import List, Iterable, Optional, Tuple, Sequence, Any, Callable

import tensorflow as tf
from tensorflow.keras.layers import Lambda

import deepchem as dc
from deepchem.models.losses import Loss
from deepchem.models.models import Model
from deepchem.models.keras_model import KerasModel
from deepchem.models.optimizers import Optimizer, Adam
from deepchem.utils.typing import OneOrMany
from deepchem.utils.data_utils import load_from_disk, save_to_disk

logger = logging.getLogger(__name__)


class NormalizingFlow(tf.keras.models.Model):
  """Base class for normalizing flow.

  The purpose of a normalizing flow is to map a simple distribution (that is
  easy to sample from and evaluate probability densities for) to a more
  complex distribution that is learned from data. The base distribution 
  p(x) is transformed by the associated normalizing flow y=g(x) to model the
  distribution p(y).

  Normalizing flows combine the advantages of autoregressive models
  (which provide likelihood estimation but do not learn features) and
  variational autoencoders (which learn feature representations but
  do not provide marginal likelihoods).

  """

  def __init__(self, base_distribution, flow_layers: Sequence,
               **kwargs) -> None:
    """Create a new NormalizingFlow.

    Parameters
    ----------
    base_distribution: tfd.Distribution
      Probability distribution to be transformed.
      Typically an N dimensional multivariate Gaussian.
    flow_layers: Sequence[tfb.Bijector]
      An iterable of bijectors that comprise the flow.
    **kwargs

    """

    try:
      import tensorflow_probability as tfp
      tfd = tfp.distributions
      tfb = tfp.bijectors
    except ModuleNotFoundError:
      raise ImportError(
          "This class requires tensorflow-probability to be installed.")

    self.base_distribution = base_distribution
    self.flow_layers = flow_layers

    # Chain of flows is also a normalizing flow
    bijector = tfb.Chain(list(reversed(self.flow_layers)))

    # An instance of tfd.TransformedDistribution
    self.flow = tfd.TransformedDistribution(
        distribution=self.base_distribution, bijector=bijector)

    super(NormalizingFlow, self).__init__(**kwargs)

  def __call__(self, *inputs, training=True):
    return self.flow.bijector.forward(*inputs)


class NormalizingFlowModel(KerasModel):
  """A base distribution and normalizing flow for applying transformations.

  Normalizing flows are effective for any application requiring 
  a probabilistic model that can both sample from a distribution and
  compute marginal likelihoods, e.g. generative modeling,
  unsupervised learning, or probabilistic inference. For a thorough review
  of normalizing flows, see [1]_.

  A distribution implements two main operations:
    1. Sampling from the transformed distribution
    2. Calculating log probabilities

  A normalizing flow implements three main operations:
    1. Forward transformation 
    2. Inverse transformation 
    3. Calculating the Jacobian

  Deep Normalizing Flow models require normalizing flow layers where
  input and output dimensions are the same, the transformation is invertible,
  and the determinant of the Jacobian is efficient to compute and
  differentiable. The determinant of the Jacobian of the transformation 
  gives a factor that preserves the probability volume to 1 when transforming
  between probability densities of different random variables.

  References
  ----------
  .. [1] Papamakarios, George et al. "Normalizing Flows for Probabilistic Modeling and Inference." (2019). https://arxiv.org/abs/1912.02762.

  """

  def __init__(self, model: NormalizingFlow, **kwargs) -> None:
    """Creates a new NormalizingFlowModel.

    In addition to the following arguments, this class also accepts all the keyword arguments from KerasModel.

    Parameters
    ----------
    model: NormalizingFlow
      An instance of NormalizingFlow.    

    Examples
    --------
    >> import tensorflow_probability as tfp
    >> tfd = tfp.distributions
    >> tfb = tfp.bijectors
    >> flow_layers = [
    ..    tfb.RealNVP(
    ..        num_masked=2,
    ..        shift_and_log_scale_fn=tfb.real_nvp_default_template(
    ..            hidden_layers=[8, 8]))
    ..]
    >> base_distribution = tfd.MultivariateNormalDiag(loc=[0., 0., 0.])
    >> nf = NormalizingFlow(base_distribution, flow_layers)
    >> nfm = NormalizingFlowModel(nf)
    >> dataset = NumpyDataset(
    ..    X=np.random.rand(5, 3).astype(np.float32),
    ..    y=np.random.rand(5,),
    ..    ids=np.arange(5))
    >> nfm.fit(dataset)

    """

    try:
      import tensorflow_probability as tfp
      tfd = tfp.distributions
      tfb = tfp.bijectors
    except ModuleNotFoundError:
      raise ImportError(
          "This class requires tensorflow-probability to be installed.")

    self.nll_loss_fn = lambda input, labels, weights: self.create_nll(input)

    super(NormalizingFlowModel, self).__init__(
        model=model, loss=self.nll_loss_fn, **kwargs)

    self.flow = self.model.flow  # normalizing flow

    # TODO: Incompability between TF and TFP means that TF doesn't track
    # trainable variables in the flow; must override `_create_gradient_fn`
    # self._variables = self.flow.trainable_variables

  def create_nll(self, input: OneOrMany[tf.Tensor]) -> tf.Tensor:
    """Create the negative log likelihood loss function.

    The default implementation is appropriate for most cases. Subclasses can
    override this if there is a need to customize it.

    Parameters
    ----------
    input: OneOrMany[tf.Tensor]
      A batch of data.

    Returns
    -------
    A Tensor equal to the loss function to use for optimization.

    """

    return -tf.reduce_mean(self.flow.log_prob(input, training=True))

  def save(self):
    """Saves model to disk using joblib."""
    save_to_disk(self.model, self.get_model_filename(self.model_dir))

  def reload(self):
    """Loads model from joblib file on disk."""
    self.model = load_from_disk(self.get_model_filename(self.model_dir))

  def _create_gradient_fn(self,
                          variables: Optional[List[tf.Variable]]) -> Callable:
    """Create a function that computes gradients and applies them to the model.

    Because of the way TensorFlow function tracing works, we need to create a
    separate function for each new set of variables.
    
    Parameters
    ----------
    variables: Optional[List[tf.Variable]]
      Variables to track during training.

    Returns
    -------
    Callable function that applies gradients for batch of training data.

    """

    @tf.function(experimental_relax_shapes=True)
    def apply_gradient_for_batch(inputs, labels, weights, loss):
      with tf.GradientTape() as tape:
        tape.watch(self.flow.trainable_variables)
        if isinstance(inputs, tf.Tensor):
          inputs = [inputs]
        if self._loss_outputs is not None:
          inputs = [inputs[i] for i in self._loss_outputs]
        batch_loss = loss(inputs, labels, weights)
      if variables is None:
        vars = self.flow.trainable_variables
      else:
        vars = variables
      grads = tape.gradient(batch_loss, vars)
      self._tf_optimizer.apply_gradients(zip(grads, vars))
      self._global_step.assign_add(1)
      return batch_loss

    return apply_gradient_for_batch


class NormalizingFlowLayer(object):
  """Base class for normalizing flow layers.

  This is an abstract base class for implementing new normalizing flow
  layers that are not available in tfb. It should not be called directly.

  A normalizing flow transforms random variables into new random variables.
  Each learnable layer is a bijection, an invertible
  transformation between two probability distributions. A simple initial
  density is pushed through the normalizing flow to produce a richer, 
  more multi-modal distribution. Normalizing flows have three main operations:

  1. Forward
    Transform a distribution. Useful for generating new samples.
  2. Inverse
    Reverse a transformation, useful for computing conditional probabilities.
  3. Log(|det(Jacobian)|) [LDJ]
    Compute the determinant of the Jacobian of the transformation, 
    which is a scaling that conserves the probability "volume" to equal 1. 

  For examples of customized normalizing flows applied to toy problems,
  see [1]_.

  References
  ----------
  .. [1] Saund, Brad. "Normalizing Flows." (2020). https://github.com/bsaund/normalizing_flows.

  Notes
  -----
  - A sequence of normalizing flows is a normalizing flow.
  - The Jacobian is the matrix of first-order derivatives of the transform.

  """

  def __init__(self, **kwargs):
    """Create a new NormalizingFlowLayer."""

    pass

  def _forward(self, x: tf.Tensor) -> tf.Tensor:
    """Forward transformation.

    x = g(y)

    Parameters
    ----------
    x: tf.Tensor
      Input tensor.

    Returns
    -------
    fwd_x: tf.Tensor
      Transformed tensor.

    """

    raise NotImplementedError("Forward transform must be defined.")

  def _inverse(self, y: tf.Tensor) -> tf.Tensor:
    """Inverse transformation.

    x = g^{-1}(y)
    
    Parameters
    ----------
    y: tf.Tensor
      Input tensor.

    Returns
    -------
    inv_y: tf.Tensor
      Inverted tensor.

    """

    raise NotImplementedError("Inverse transform must be defined.")

  def _forward_log_det_jacobian(self, x: tf.Tensor) -> tf.Tensor:
    """Log |Determinant(Jacobian(x)|

    Note x = g^{-1}(y)

    Parameters
    ----------
    x: tf.Tensor
      Input tensor.

    Returns
    -------
    ldj: tf.Tensor
      Log of absolute value of determinant of Jacobian of x.

    """

    raise NotImplementedError("LDJ must be defined.")

  def _inverse_log_det_jacobian(self, y: tf.Tensor) -> tf.Tensor:
    """Inverse LDJ.

    The ILDJ = -LDJ.

    Note x = g^{-1}(y)

    Parameters
    ----------
    y: tf.Tensor
      Input tensor.

    Returns
    -------
    ildj: tf.Tensor
      Log of absolute value of determinant of Jacobian of y.

    """

    return -self._forward_log_det_jacobian(self._inverse(y))
