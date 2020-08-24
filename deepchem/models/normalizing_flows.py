"""
Normalizing flows for transforming probability distributions.
"""

import numpy as np
import logging
from typing import List, Iterable, Optional, Tuple, Sequence, Any

import tensorflow as tf
from tensorflow.keras.layers import Lambda

import deepchem as dc
from deepchem.models.losses import Loss
from deepchem.models.models import Model
from deepchem.models.keras_model import KerasModel
from deepchem.models.optimizers import Optimizer, Adam

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

  def __init__(self, base_distribution, flow_layers, **kwargs):
    """Create a new NormalizingFlow.

    Parameters
    ----------
    base_distribution : tfd.Distribution
      Probability distribution to be transformed.
      Typically an N dimensional multivariate Gaussian.
    flow_layers : Sequence[tfb.Bijector]
      An iterable of bijectors that comprise the flow.

    """

    try:
      import tensorflow_probability as tfp
      tfd = tfp.distributions
      tfb = tfp.bijectors
    except ModuleNotFoundError:
      raise ValueError(
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

  A distribution implements two main operations:
    1. Sampling from the transformed distribution.
    2. Calculating log probabilities.

  A normalizing flow implements three main operations:
    1. Forward transformation, 2. Inverse transformation, and 
    3. Calculating the Jacobian.

  Deep Normalizing Flow models require normalizing flow layers where
  input and output dimensions are the same, the transformation is invertible,
  and the determinant of the Jacobian is efficient to compute and
  differentiable. The determinant of the Jacobian of the transformation 
  gives a factor that preserves the probability volume to 1 when transforming
  between probability densities of different random variables.

  They are effective for any application requiring a probabilistic
  model with these capabilities, e.g. generative modeling,
  unsupervised learning, or probabilistic inference. For a thorough review
  of normalizing flows, see [1]_.

  References
  ----------
  .. [1] Papamakarios, George et al. "Normalizing Flows for Probabilistic Modeling and Inference." (2019). https://arxiv.org/abs/1912.02762.

  """

  def __init__(self, model: NormalizingFlow, **kwargs):
    """Creates a new NormalizingFlowModel.

    Parameters
    ----------
    model: NormalizingFlow
      An instance of NormalizingFlow.
    loss: dc.models.losses.Loss, default NegLogLoss
      Loss function
    optimizer: dc.models.optimizers.Optimizer, default Adam
      Optimizer.
    

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
      raise ValueError(
          "This class requires tensorflow-probability to be installed.")

    self.model = model
    self.flow = model.flow  # normalizing flow
    """Initialize tf network."""
    x = self.flow.distribution.sample(self.flow.distribution.batch_shape)
    for b in reversed(self.flow.bijector.bijectors):
      x = b.forward(x)

    self.nll_loss_fn = lambda output, labels, weights: self.create_nll(output)

    super(NormalizingFlowModel, self).__init__(
        model=self.model, loss=self.nll_loss_fn, **kwargs)

  def create_nll(self, output):
    """Create the negative log loss function for density estimation.

    The default implementation is appropriate for most cases. Subclasses can
    override this if there is a need to customize it.

    Parameters
    ----------
    output: Tensor
      the output from the normalizing flow on a batch of generated data.
      This is its estimate of the probability that the sample was drawn
      from the target distribution.

    Returns
    -------
    A Tensor equal to the loss function to use for optimization.

    """

    return Lambda(
        lambda x: -tf.reduce_mean(tf.math.add(self.flow.log_prob(x), 1e-10)))(
            output)


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

  def _forward(self, x):
    """Forward transformation.

    x = g(y)

    Parameters
    ----------
    x : Tensor
      Input tensor.

    Returns
    -------
    fwd_x : Tensor
      Transformed tensor.

    """

    raise NotImplementedError("Forward transform must be defined.")

  def _inverse(self, y):
    """Inverse transformation.

    x = g^{-1}(y)
    
    Parameters
    ----------
    y : Tensor
      Input tensor.

    Returns
    -------
    inv_y : Tensor
      Inverted tensor.

    """

    raise NotImplementedError("Inverse transform must be defined.")

  def _forward_log_det_jacobian(self, x):
    """Log |Determinant(Jacobian(x)|

    Note x = g^{-1}(y)

    Parameters
    ----------
    x : Tensor
      Input tensor.

    Returns
    -------
    ldj : Tensor
      Log of absolute value of determinant of Jacobian of x.

    """

    raise NotImplementedError("LDJ must be defined.")

  def _inverse_log_det_jacobian(self, y):
    """Inverse LDJ.

    The ILDJ = -LDJ.

    Note x = g^{-1}(y)

    Parameters
    ----------
    y : Tensor
      Input tensor.

    Returns
    -------
    ildj : Tensor
      Log of absolute value of determinant of Jacobian of y.

    """

    return -self._forward_log_det_jacobian(self._inverse(y))
