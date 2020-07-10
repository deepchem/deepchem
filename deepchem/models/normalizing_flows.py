"""
Normalizing flows for transforming distributions.
"""

import numpy as np
import logging
from typing import List

from deepchem.models.models import Model

logger = logging.getLogger(__name__)


class NormalizingFlowLayer(object):
  """Base class for normalizing flow layers.

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

  They are effective for any application requiring a probabilistic
  model with these capabilities (e.g. generative modeling,
  unsupervised learning, probabilistic inference). For a thorough review
  of normalizing flows, see [1]_.

  References
  ----------
  .. [1] Papamakarios, George et al. "Normalizing Flows for Probabilistic Modeling and Inference." (2019). https://arxiv.org/abs/1912.02762.

  Notes
  -----
  - A sequence of normalizing flows is a normalizing flow.
  - The Jacobian is the matrix of first-order derivatives of the transform.

  """

  def __init__(self, model, **kwargs):
    """Create a new NormalizingFlowLayer.
    
    Parameters
    ----------
    model : object
      Model object from TensorFlowProbability, Pytorch, etc. The model
      should be a bijective transformation with forward, inverse, and 
      LDJ methods.
    kwargs : dict
      Additional keyword arguments.

    """

    self.model = model

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


class NormalizingFlow(object):
  """Base class for normalizing flow.

  A normalizing flow is a chain of NormalizingFlowLayers.

  The purpose of a normalizing flow is to map a simple distribution that is
  easy to sample from and evaluate probability densities to more complex
  distribituions that are learned with data. The base distribution p(x) is
  transformed by the associated normalizing flow y=g(x) to model the
  distribution p(y).

  Normalizing flows combine the advantages of autoregressive models
  (which provide likelihood estimation but do not learn features) and
  variational autoencoders (which learn feature representations but
  do not provide marginal likelihoods).

  The determinant of the Jacobian of the transformation gives a factor
  that preserves the probability volume to 1 when transforming between
  probability densities of different random variables.

  """

  def __init__(self, flows: List[NormalizingFlowLayer]):
    """Create a new NormalizingFlow.

    Parameters
    ----------
    flows : List[NormalizingFlowLayer]
      List of NormalizingFlowLayers.

    """

    self.flows = flows

  def _forward(self, x):
    """Apply normalizing flow.

    Parameters
    ----------
    x : Tensor
      Samples from distribution.

    Returns
    -------
    (ys, ldjs) : Tuple[Tensor, Tensor]
      Transformed samples and log det Jacobian values.

    """

    ys = [x]
    ldjs = np.zeros(x.shape[0])

    for flow in self.flows:
      x = flow._forward(x)
      ldj = flow._forward_log_det_jacobian(x)
      ldjs += ldj
      ys.append(x)

    return (ys, ldjs)

  def _inverse(self, y):
    """Invert normalizing flow.

    Parameters
    ----------
    y : Tensor
      Samples from transformed distribution.

    Returns
    -------
    (xs, ildjs) : Tuple[Tensor, Tensor]
      Transformed samples and inverse log det Jacobian values.

    """

    xs = [y]
    ildjs = np.zeros(y.shape[0])

    for flow in self.flows:
      x = flow._inverse(y)
      ildj = flow._inverse_log_det_jacobian(y)
      ildjs += ildj
      xs.append(x)

    return (xs, ildjs)


class NormalizingFlowModel(Model):
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
  differentiable. 

  """

  def __init__(self,
               base_distribution,
               normalizing_flow: NormalizingFlow,
               event_shape=None):
    """Creates a new NormalizingFlowModel.

    Parameters
    ----------
    base_distribution : Distribution
      Probability distribution to be transformed.
    normalizing_flow : NormalizingFlow
      An instance of NormalizingFlow.
    event_shape : Tensor
      Shape of single samples drawn from distribution. For scalar
      distributions the shape is []. For a 3D Multi-variate normal 
      distribution, the shape is [3].

    """

    self.base_distribution = base_distribution
    self.normalizing_flow = normalizing_flow
    self.event_shape = event_shape

  def __call__(self, x):
    """Apply `normalizing_flow` to samples from `base_distribution`.

    Parameters
    ----------
    x : Tensor
      Samples from `base_distribution`.

    Returns
    -------
    (y, ldjs) : Tuple[Tensor, Tensor]
      Samples from transformed distribution and log det Jacobian.

    """

    return self.normalizing_flow._forward(x)

  def sample(self, shape, seed=None):
    """Generate samples from the transformed distribution.

    Parameters
    ----------
    shape : Tensor
      Shape of generated samples.
    seed : int
      Random seed.

    Returns
    -------
    samples : Tensor
      Tensor of random samples from the distribution.

    """

    raise NotImplementedError("Sampling must be defined.")

  def log_prob(self, value):
    """Log probability function.

    Given a datapoint `x`, what is the probability assigned by the
    model p(x). Equivalent to probability density estimation.

    The negative log likelihood (NLL) is a common loss function for 
    fitting data to distributions.

    NLL = -mean(log_prob(x))

    Parameters
    ----------
    value : Tensor
      Value of random variable.

    Returns
    -------
    log_prob : Tensor
      Log-likelihood function.

    """

    raise NotImplementedError("Log prob must be defined.")
