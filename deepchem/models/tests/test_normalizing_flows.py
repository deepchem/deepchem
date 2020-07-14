"""
Tests for Normalizing Flows.
"""

import os
import sys
import pytest

import deepchem
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import unittest
import numpy as np

from deepchem.models.normalizing_flows import NormalizingFlowLayer, NormalizingFlow, NormalizingFlowModel


class TestNormalizingFlow(unittest.TestCase):

  def setUp(self):

    self.ef = ExpFlow()
    self.nfm = TransformedNormal()

  def test_simple_flow(self):
    """Tests a simple flow of Exp layers."""

    X = self.nfm.sample([10])

    ys, ldjs = self.nfm(X)
    xs, ildjs = self.nfm.normalizing_flow._inverse(ys[-1])

    assert len(xs) == 3
    assert len(ys) == 3
    assert xs[0].shape == 10
    assert np.isclose(self.nfm.log_prob(1), -1.4, atol=0.5)


class ExpFlow(NormalizingFlowLayer):
  """Exp(x)."""

  def __init__(self, **kwargs):
    model = tfp.bijectors.Exp()
    super(ExpFlow, self).__init__(model, **kwargs)

  def _forward(self, x):
    return self.model.forward(x)

  def _inverse(self, y):
    return self.model.inverse(y)

  def _forward_log_det_jacobian(self, x):
    return self.model.forward_log_det_jacobian(x, 1)


class TransformedNormal(NormalizingFlowModel):
  """Univariate Gaussian base distribution."""

  def __init__(self, 
    base_distribution=tfp.distributions.Normal(0, 1),
    normalizing_flow=NormalizingFlow([ExpFlow(), ExpFlow()])
    ):

    super(TransformedNormal, self).__init__(base_distribution, normalizing_flow)

  def sample(self, shape, seed=None):
    return self.base_distribution.sample(sample_shape=shape, seed=seed)

  def log_prob(self, value):
    return self.base_distribution.log_prob(value=value)

