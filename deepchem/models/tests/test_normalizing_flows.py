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

  def test_simple_flow(self):
    """Tests a simple flow of Exp layers."""

    dist = tfp.distributions.Normal(0, 1)  # univariate Gaussian
    X = dist.sample([10])
    g = self.ef
    flows = [g, g]
    nf = NormalizingFlow(flows)
    nfm = NormalizingFlowModel(dist, nf)

    ys, ldjs = nfm(X)
    xs, ildjs = nf._inverse(ys[-1])

    assert len(xs) == 3
    assert len(ys) == 3
    assert xs[0].shape == 10


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
