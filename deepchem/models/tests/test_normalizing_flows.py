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

from deepchem.data import NumpyDataset
from deepchem.models.normalizing_flows import NormalizingFlow, NormalizingFlowModel

tfd = tfp.distributions
tfb = tfp.bijectors


class TestNormalizingFlow(unittest.TestCase):

  def setUp(self):

    flow_layers = [
        tfb.RealNVP(
            num_masked=2,
            shift_and_log_scale_fn=tfb.real_nvp_default_template(
                hidden_layers=[8, 8]))
    ]
    # 3D Multivariate Gaussian base distribution
    self.nf = NormalizingFlow(
        base_distribution=tfd.MultivariateNormalDiag(loc=[0., 0., 0.]),
        flow_layers=flow_layers)

    self.nfm = NormalizingFlowModel(self.nf, batch_size=1)

    # Must be float32 for RealNVP
    self.dataset = NumpyDataset(
        X=np.random.rand(5, 3).astype(np.float32),
        y=np.random.rand(5,),
        ids=np.arange(5))

  def test_simple_flow(self):
    """Tests a simple flow of one RealNVP layer."""

    X = self.nfm.flow.sample()
    x1 = tf.zeros([3])
    x2 = self.dataset.X[0]

    # log likelihoods should be negative
    assert self.nfm.flow.log_prob(X).numpy() < 0
    assert self.nfm.flow.log_prob(x1).numpy() < 0
    assert self.nfm.flow.log_prob(x2).numpy() < 0

    # # Fit model
    final = self.nfm.fit(self.dataset, nb_epoch=5)
    assert final > 0
