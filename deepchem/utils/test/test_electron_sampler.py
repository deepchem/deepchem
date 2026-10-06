"""
Test for electron_sampler.py
"""

import numpy as np
from scipy.stats import norm
from deepchem.utils.electron_sampler import ElectronSampler


def f(x):
    # dummy function which can be passed as the parameter f to simultaneous_move and single_move
    return 2 * np.log(np.random.uniform(low=0, high=1.0, size=np.shape(x)[0]))


def test_mean():
    distribution = ElectronSampler(np.array([[1, 1, 3], [3, 2, 3]]), f)
    x1 = np.array([[[[1, 2, 3]]], [[[4, 5, 6]]]])
    mean = distribution.harmonic_mean(x1)
    assert (mean == np.array([[[[1.3333333333333333]]],
                              [[[4.988597077109626]]]])).all()


def test_log_prob():
    x1 = np.array([[[[1, 2, 3]]], [[[4, 5, 6]]]])
    x2 = np.array([[[[10, 6, 4]]], [[[2, 1, 7]]]])
    sigma = np.full(np.shape(x1), 1)
    distribution = ElectronSampler(np.array([[1, 1, 3], [3, 2, 3]]), f)
    move_probability = distribution.log_prob_gaussian(x1, x2, sigma)
    assert (move_probability == np.array([-49, -10.5])).all()


def test_steps():

    # test for gauss_initialize_position
    distribution = ElectronSampler(np.array([[1, 1, 3], [3, 2, 3]]),
                                   f,
                                   batch_no=2,
                                   steps=1000)
    distribution.gauss_initialize_position(np.array([[1], [2]]))
    assert ((distribution.x -
             np.array([[[[1, 1, 3]], [[3, 2, 3]], [[3, 2, 3]]],
                       [[[1, 1, 3]], [[3, 2, 3]], [[3, 2, 3]]]])) != 0).any()

    # testing symmetric simultaneous_move
    x1 = distribution.x
    distribution.move()
    assert ((distribution.x - x1) != 0).all()

    # testing asymmetric simultaneous_move
    distribution = ElectronSampler(np.array([[1, 1, 3], [3, 2, 3]]),
                                   f,
                                   batch_no=2,
                                   steps=1000,
                                   symmetric=False)
    distribution.gauss_initialize_position(np.array([[1], [2]]))
    x1 = distribution.x
    distribution.move(asymmetric_func=distribution.harmonic_mean)
    assert ((distribution.x - x1) != 0).all()
    assert np.shape(distribution.sampled_electrons) == (2000, 3, 1, 3)

    # testing symmetric single_move
    distribution = ElectronSampler(np.array([[1, 1, 3], [3, 2, 3]]),
                                   f,
                                   batch_no=2,
                                   steps=1000,
                                   simultaneous=False)
    distribution.gauss_initialize_position(np.array([[1], [2]]))
    x1 = distribution.x
    distribution.move(index=1)
    assert ((distribution.x[:, 1, :, :] - x1[:, 1, :, :]) != 0).all()
    assert ((distribution.x[:, 2, :, :] - x1[:, 2, :, :]) == 0).all()
    assert np.shape(distribution.sampled_electrons) == (2000, 3, 1, 3)

    # testing asymmetric single_move
    distribution = ElectronSampler(np.array([[1, 1, 3], [3, 2, 3]]),
                                   f,
                                   batch_no=2,
                                   steps=1000,
                                   simultaneous=False,
                                   symmetric=False)
    distribution.gauss_initialize_position(np.array([[1], [2]]))
    x1 = distribution.x
    distribution.move(asymmetric_func=distribution.harmonic_mean, index=1)
    assert ((distribution.x[:, 1, :, :] - x1[:, 1, :, :]) != 0).all()
    assert ((distribution.x[:, 2, :, :] - x1[:, 2, :, :]) == 0).all()
    assert np.shape(distribution.sampled_electrons) == (2000, 3, 1, 3)


def test_log_prob_sigma_shapes():
    # sigma can have one entry per electron, one per coordinate or one per
    # batch element. All of them must give the gaussian log-density, up to the
    # constant term that log_prob_gaussian leaves out.
    distribution = ElectronSampler(np.array([[1, 1, 3], [3, 2, 3]]), f)
    rs = np.random.RandomState(0)
    batch, n_electrons = 2, 4
    y = rs.randn(batch, n_electrons, 1, 3)
    mu = rs.randn(batch, n_electrons, 1, 3)
    const = -(n_electrons * 3) / 2 * np.log(2 * np.pi)
    for shape in [(batch, n_electrons, 1, 1), (batch, n_electrons, 1, 3),
                  (batch, 1, 1, 1)]:
        sigma = rs.uniform(0.2, 2.0, shape)
        expected = norm.logpdf(y, mu, np.broadcast_to(
            sigma, y.shape)).sum(axis=(1, 2, 3)) - const
        result = distribution.log_prob_gaussian(y, mu, sigma)
        assert np.allclose(result, expected)
