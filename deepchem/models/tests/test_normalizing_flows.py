"""
Tests for Normalizing Flows.
"""

import pytest

import unittest
from deepchem.data import NumpyDataset

try:
    import tensorflow as tf
    import tensorflow_probability as tfp
    from deepchem.models.normalizing_flows import NormalizingFlow, NormalizingFlowModel
    tfd = tfp.distributions
    tfb = tfp.bijectors
    has_tensorflow_probablity = True
except:
    has_tensorflow_probablity = False


@unittest.skipIf(not has_tensorflow_probablity,
                 'tensorflow_probability not installed')
@pytest.mark.tensorflow
def test_normalizing_flow():

    flow_layers = [
        tfb.RealNVP(num_masked=1,
                    shift_and_log_scale_fn=tfb.real_nvp_default_template(
                        hidden_layers=[8, 8]))
    ]
    # 3D Multivariate Gaussian base distribution
    nf = NormalizingFlow(
        base_distribution=tfd.MultivariateNormalDiag(loc=[0., 0.]),
        flow_layers=flow_layers)

    nfm = NormalizingFlowModel(nf)

    # Must be float32 for RealNVP
    target_distribution = tfd.MultivariateNormalDiag(loc=[1., 0.])
    dataset = NumpyDataset(X=target_distribution.sample(96))

    # Tests a simple flow of one RealNVP layer.

    X = nfm.flow.sample()
    x1 = tf.zeros([2])
    x2 = dataset.X[0]

    # log likelihoods should be negative
    assert nfm.flow.log_prob(X).numpy() < 0
    assert nfm.flow.log_prob(x1).numpy() < 0
    assert nfm.flow.log_prob(x2).numpy() < 0

    # # Fit model
    final = nfm.fit(dataset, nb_epoch=5)
    print(final)
    assert final > 0
