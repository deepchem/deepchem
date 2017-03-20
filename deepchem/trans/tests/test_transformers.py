"""
Tests for transformer objects. 
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "MIT"

import os
import unittest
import numpy as np
import pandas as pd
import deepchem as dc
import numpy.random as random


class TestTransformers(unittest.TestCase):
  """
  Test top-level API for transformer objects.
  """

  def setUp(self):
    super(TestTransformers, self).setUp()
    self.current_dir = os.path.dirname(os.path.abspath(__file__))

  def test_y_log_transformer(self):
    """Tests logarithmic data transformer."""
    solubility_dataset = dc.data.tests.load_solubility_data()
    log_transformer = dc.trans.LogTransformer(
        transform_y=True, dataset=solubility_dataset)
    X, y, w, ids = (solubility_dataset.X, solubility_dataset.y,
                    solubility_dataset.w, solubility_dataset.ids)
    solubility_dataset = log_transformer.transform(solubility_dataset)
    X_t, y_t, w_t, ids_t = (solubility_dataset.X, solubility_dataset.y,
                            solubility_dataset.w, solubility_dataset.ids)

    # Check ids are unchanged.
    for id_elt, id_t_elt in zip(ids, ids_t):
      assert id_elt == id_t_elt
    # Check X is unchanged since this is a y transformer
    np.testing.assert_allclose(X, X_t)
    # Check w is unchanged since this is a y transformer
    np.testing.assert_allclose(w, w_t)
    # Check y is now a logarithmic version of itself
    np.testing.assert_allclose(y_t, np.log(y + 1))

    # Check that untransform does the right thing.
    np.testing.assert_allclose(log_transformer.untransform(y_t), y)

  def test_transform_unlabelled(self):
    ul_dataset = dc.data.tests.load_unlabelled_data()
    # transforming y should raise an exception
    with self.assertRaises(ValueError) as context:
      dc.trans.transformers.Transformer(transform_y=True).transform(ul_dataset)

    # transforming w should raise an exception
    with self.assertRaises(ValueError) as context:
      dc.trans.transformers.Transformer(transform_w=True).transform(ul_dataset)

    # transforming X should be okay
    dc.trans.NormalizationTransformer(
        transform_X=True, dataset=ul_dataset).transform(ul_dataset)

  def test_X_log_transformer(self):
    """Tests logarithmic data transformer."""
    solubility_dataset = dc.data.tests.load_solubility_data()
    log_transformer = dc.trans.LogTransformer(
        transform_X=True, dataset=solubility_dataset)
    X, y, w, ids = (solubility_dataset.X, solubility_dataset.y,
                    solubility_dataset.w, solubility_dataset.ids)
    solubility_dataset = log_transformer.transform(solubility_dataset)
    X_t, y_t, w_t, ids_t = (solubility_dataset.X, solubility_dataset.y,
                            solubility_dataset.w, solubility_dataset.ids)

    # Check ids are unchanged.
    for id_elt, id_t_elt in zip(ids, ids_t):
      assert id_elt == id_t_elt
    # Check y is unchanged since this is a X transformer
    np.testing.assert_allclose(y, y_t)
    # Check w is unchanged since this is a y transformer
    np.testing.assert_allclose(w, w_t)
    # Check y is now a logarithmic version of itself
    np.testing.assert_allclose(X_t, np.log(X + 1))

    # Check that untransform does the right thing.
    np.testing.assert_allclose(log_transformer.untransform(X_t), X)

  def test_y_log_transformer_select(self):
    """Tests logarithmic data transformer with selection."""
    multitask_dataset = dc.data.tests.load_feat_multitask_data()
    dfe = pd.read_csv(
        os.path.join(self.current_dir,
                     "../../models/tests/feat_multitask_example.csv"))
    tid = []
    tasklist = ["task0", "task3", "task4", "task5"]
    first_task = "task0"
    for task in tasklist:
      tiid = dfe.columns.get_loc(task) - dfe.columns.get_loc(first_task)
      tid = np.concatenate((tid, np.array([tiid])))
    tasks = tid.astype(int)
    log_transformer = dc.trans.LogTransformer(
        transform_y=True, tasks=tasks, dataset=multitask_dataset)
    X, y, w, ids = (multitask_dataset.X, multitask_dataset.y,
                    multitask_dataset.w, multitask_dataset.ids)
    multitask_dataset = log_transformer.transform(multitask_dataset)
    X_t, y_t, w_t, ids_t = (multitask_dataset.X, multitask_dataset.y,
                            multitask_dataset.w, multitask_dataset.ids)

    # Check ids are unchanged.
    for id_elt, id_t_elt in zip(ids, ids_t):
      assert id_elt == id_t_elt
    # Check X is unchanged since this is a y transformer
    np.testing.assert_allclose(X, X_t)
    # Check w is unchanged since this is a y transformer
    np.testing.assert_allclose(w, w_t)
    # Check y is now a logarithmic version of itself
    np.testing.assert_allclose(y_t[:, tasks], np.log(y[:, tasks] + 1))

    # Check that untransform does the right thing.
    np.testing.assert_allclose(log_transformer.untransform(y_t), y)

  def test_X_log_transformer_select(self):
    #Tests logarithmic data transformer with selection.
    multitask_dataset = dc.data.tests.load_feat_multitask_data()
    dfe = pd.read_csv(
        os.path.join(self.current_dir,
                     "../../models/tests/feat_multitask_example.csv"))
    fid = []
    featurelist = ["feat0", "feat1", "feat2", "feat3", "feat5"]
    first_feature = "feat0"
    for feature in featurelist:
      fiid = dfe.columns.get_loc(feature) - dfe.columns.get_loc(first_feature)
      fid = np.concatenate((fid, np.array([fiid])))
    features = fid.astype(int)
    log_transformer = dc.trans.LogTransformer(
        transform_X=True, features=features, dataset=multitask_dataset)
    X, y, w, ids = (multitask_dataset.X, multitask_dataset.y,
                    multitask_dataset.w, multitask_dataset.ids)
    multitask_dataset = log_transformer.transform(multitask_dataset)
    X_t, y_t, w_t, ids_t = (multitask_dataset.X, multitask_dataset.y,
                            multitask_dataset.w, multitask_dataset.ids)

    # Check ids are unchanged.
    for id_elt, id_t_elt in zip(ids, ids_t):
      assert id_elt == id_t_elt
    # Check y is unchanged since this is a X transformer
    np.testing.assert_allclose(y, y_t)
    # Check w is unchanged since this is a y transformer
    np.testing.assert_allclose(w, w_t)
    # Check y is now a logarithmic version of itself
    np.testing.assert_allclose(X_t[:, features], np.log(X[:, features] + 1))

    # Check that untransform does the right thing.
    np.testing.assert_allclose(log_transformer.untransform(X_t), X)

  def test_y_normalization_transformer(self):
    """Tests normalization transformer."""
    solubility_dataset = dc.data.tests.load_solubility_data()
    normalization_transformer = dc.trans.NormalizationTransformer(
        transform_y=True, dataset=solubility_dataset)
    X, y, w, ids = (solubility_dataset.X, solubility_dataset.y,
                    solubility_dataset.w, solubility_dataset.ids)
    solubility_dataset = normalization_transformer.transform(solubility_dataset)
    X_t, y_t, w_t, ids_t = (solubility_dataset.X, solubility_dataset.y,
                            solubility_dataset.w, solubility_dataset.ids)
    # Check ids are unchanged.
    for id_elt, id_t_elt in zip(ids, ids_t):
      assert id_elt == id_t_elt
    # Check X is unchanged since this is a y transformer
    np.testing.assert_allclose(X, X_t)
    # Check w is unchanged since this is a y transformer
    np.testing.assert_allclose(w, w_t)
    # Check that y_t has zero mean, unit std.
    assert np.isclose(y_t.mean(), 0.)
    assert np.isclose(y_t.std(), 1.)

    # Check that untransform does the right thing.
    np.testing.assert_allclose(normalization_transformer.untransform(y_t), y)

  def test_X_normalization_transformer(self):
    """Tests normalization transformer."""
    solubility_dataset = dc.data.tests.load_solubility_data()
    normalization_transformer = dc.trans.NormalizationTransformer(
        transform_X=True, dataset=solubility_dataset)
    X, y, w, ids = (solubility_dataset.X, solubility_dataset.y,
                    solubility_dataset.w, solubility_dataset.ids)
    solubility_dataset = normalization_transformer.transform(solubility_dataset)
    X_t, y_t, w_t, ids_t = (solubility_dataset.X, solubility_dataset.y,
                            solubility_dataset.w, solubility_dataset.ids)
    # Check ids are unchanged.
    for id_elt, id_t_elt in zip(ids, ids_t):
      assert id_elt == id_t_elt
    # Check y is unchanged since this is a X transformer
    np.testing.assert_allclose(y, y_t)
    # Check w is unchanged since this is a y transformer
    np.testing.assert_allclose(w, w_t)
    # Check that X_t has zero mean, unit std.
    #np.set_printoptions(threshold='nan')
    mean = X_t.mean(axis=0)
    assert np.amax(np.abs(mean - np.zeros_like(mean))) < 1e-7
    orig_std_array = X.std(axis=0)
    std_array = X_t.std(axis=0)
    # Entries with zero std are not normalized
    for orig_std, std in zip(orig_std_array, std_array):
      if not np.isclose(orig_std, 0):
        assert np.isclose(std, 1)

    # TODO(rbharath): Untransform doesn't work properly for binary feature
    # vectors. Need to figure out what's wrong here. (low priority)
    ## Check that untransform does the right thing.
    #np.testing.assert_allclose(normalization_transformer.untransform(X_t), X)

  def test_cdf_X_transformer(self):
    """Test CDF transformer on Gaussian normal dataset."""
    target = np.array(np.transpose(np.linspace(0., 1., 1001)))
    target = np.transpose(np.array(np.append([target], [target], axis=0)))
    gaussian_dataset = dc.data.tests.load_gaussian_cdf_data()
    bins = 1001
    cdf_transformer = dc.trans.CDFTransformer(
        transform_X=True, dataset=gaussian_dataset, bins=bins)
    X, y, w, ids = (gaussian_dataset.X, gaussian_dataset.y, gaussian_dataset.w,
                    gaussian_dataset.ids)
    gaussian_dataset = cdf_transformer.transform(gaussian_dataset, bins=bins)
    X_t, y_t, w_t, ids_t = (gaussian_dataset.X, gaussian_dataset.y,
                            gaussian_dataset.w, gaussian_dataset.ids)

    # Check ids are unchanged.
    for id_elt, id_t_elt in zip(ids, ids_t):
      assert id_elt == id_t_elt
    # Check y is unchanged since this is an X transformer
    np.testing.assert_allclose(y, y_t)
    # Check w is unchanged since this is an X transformer
    np.testing.assert_allclose(w, w_t)
    # Check X is now holding the proper values when sorted.
    sorted = np.sort(X_t, axis=0)
    np.testing.assert_allclose(sorted, target)

  def test_cdf_y_transformer(self):
    #Test CDF transformer on Gaussian normal dataset.
    target = np.array(np.transpose(np.linspace(0., 1., 1001)))
    target = np.transpose(np.array(np.append([target], [target], axis=0)))
    gaussian_dataset = dc.data.tests.load_gaussian_cdf_data()
    bins = 1001
    cdf_transformer = dc.trans.CDFTransformer(
        transform_y=True, dataset=gaussian_dataset, bins=bins)
    X, y, w, ids = (gaussian_dataset.X, gaussian_dataset.y, gaussian_dataset.w,
                    gaussian_dataset.ids)
    gaussian_dataset = cdf_transformer.transform(gaussian_dataset, bins=bins)
    X_t, y_t, w_t, ids_t = (gaussian_dataset.X, gaussian_dataset.y,
                            gaussian_dataset.w, gaussian_dataset.ids)

    # Check ids are unchanged.
    for id_elt, id_t_elt in zip(ids, ids_t):
      assert id_elt == id_t_elt
    # Check X is unchanged since this is an y transformer
    np.testing.assert_allclose(X, X_t)
    # Check w is unchanged since this is an y transformer
    np.testing.assert_allclose(w, w_t)
    # Check y is now holding the proper values when sorted.
    sorted = np.sort(y_t, axis=0)
    np.testing.assert_allclose(sorted, target)

    # Check that untransform does the right thing.
    np.testing.assert_allclose(cdf_transformer.untransform(y_t), y)

  def test_clipping_X_transformer(self):
    """Test clipping transformer on X of singletask dataset."""
    n_samples = 10
    n_features = 3
    n_tasks = 1
    ids = np.arange(n_samples)
    X = np.ones((n_samples, n_features))
    target = 5. * X
    X *= 6.
    y = np.zeros((n_samples, n_tasks))
    w = np.ones((n_samples, n_tasks))
    dataset = dc.data.NumpyDataset(X, y, w, ids)
    transformer = dc.trans.ClippingTransformer(transform_X=True, x_max=5.)
    clipped_dataset = transformer.transform(dataset)
    X_t, y_t, w_t, ids_t = (clipped_dataset.X, clipped_dataset.y,
                            clipped_dataset.w, clipped_dataset.ids)
    # Check ids are unchanged.
    for id_elt, id_t_elt in zip(ids, ids_t):
      assert id_elt == id_t_elt
    # Check y is unchanged since this is an X transformer
    np.testing.assert_allclose(y, y_t)
    # Check w is unchanged since this is an X transformer
    np.testing.assert_allclose(w, w_t)
    # Check X is now holding the proper values when sorted.
    np.testing.assert_allclose(X_t, target)

  def test_clipping_y_transformer(self):
    """Test clipping transformer on y of singletask dataset."""
    n_samples = 10
    n_features = 3
    n_tasks = 1
    ids = np.arange(n_samples)
    X = np.zeros((n_samples, n_features))
    y = np.ones((n_samples, n_tasks))
    target = 5. * y
    y *= 6.
    w = np.ones((n_samples, n_tasks))
    dataset = dc.data.NumpyDataset(X, y, w, ids)
    transformer = dc.trans.ClippingTransformer(transform_y=True, y_max=5.)
    clipped_dataset = transformer.transform(dataset)
    X_t, y_t, w_t, ids_t = (clipped_dataset.X, clipped_dataset.y,
                            clipped_dataset.w, clipped_dataset.ids)
    # Check ids are unchanged.
    for id_elt, id_t_elt in zip(ids, ids_t):
      assert id_elt == id_t_elt
    # Check X is unchanged since this is a y transformer
    np.testing.assert_allclose(X, X_t)
    # Check w is unchanged since this is a y transformer
    np.testing.assert_allclose(w, w_t)
    # Check y is now holding the proper values when sorted.
    np.testing.assert_allclose(y_t, target)

  def test_power_X_transformer(self):
    """Test Power transformer on Gaussian normal dataset."""
    gaussian_dataset = dc.data.tests.load_gaussian_cdf_data()
    powers = [1, 2, 0.5]
    power_transformer = dc.trans.PowerTransformer(
        transform_X=True, powers=powers)
    X, y, w, ids = (gaussian_dataset.X, gaussian_dataset.y, gaussian_dataset.w,
                    gaussian_dataset.ids)
    gaussian_dataset2 = power_transformer.transform(gaussian_dataset)
    X_t, y_t, w_t, ids_t = (gaussian_dataset2.X, gaussian_dataset2.y,
                            gaussian_dataset2.w, gaussian_dataset2.ids)

    # Check ids are unchanged.
    for id_elt, id_t_elt in zip(ids, ids_t):
      assert id_elt == id_t_elt
    # Check y is unchanged since this is an X transformer
    np.testing.assert_allclose(y, y_t)
    # Check w is unchanged since this is an X transformer
    np.testing.assert_allclose(w, w_t)
    # Check X is now holding the proper values in each column.
    np.testing.assert_allclose(X_t.shape[1], len(powers) * X.shape[1])
    np.testing.assert_allclose(X, X_t[:, :2])
    np.testing.assert_allclose(np.power(X, 2), X_t[:, 2:4])
    np.testing.assert_allclose(np.power(X, 0.5), X_t[:, 4:])

  def test_power_y_transformer(self):
    """Test Power transformer on Gaussian normal dataset."""
    gaussian_dataset = dc.data.tests.load_gaussian_cdf_data()
    powers = [1, 2, 0.5]
    power_transformer = dc.trans.PowerTransformer(
        transform_y=True, powers=powers)
    X, y, w, ids = (gaussian_dataset.X, gaussian_dataset.y, gaussian_dataset.w,
                    gaussian_dataset.ids)
    gaussian_dataset2 = power_transformer.transform(gaussian_dataset)
    X_t, y_t, w_t, ids_t = (gaussian_dataset2.X, gaussian_dataset2.y,
                            gaussian_dataset2.w, gaussian_dataset2.ids)

    # Check ids are unchanged.
    for id_elt, id_t_elt in zip(ids, ids_t):
      assert id_elt == id_t_elt
    # Check X is unchanged since this is an X transformer
    np.testing.assert_allclose(X, X_t)
    # Check w is unchanged since this is an X transformer
    np.testing.assert_allclose(w, w_t)
    # Check y is now holding the proper values in each column.
    np.testing.assert_allclose(y_t.shape[1], len(powers) * y.shape[1])
    np.testing.assert_allclose(y, y_t[:, :2])
    np.testing.assert_allclose(np.power(y, 2), y_t[:, 2:4])
    np.testing.assert_allclose(np.power(y, 0.5), y_t[:, 4:])

    # Check that untransform does the right thing.
    np.testing.assert_allclose(power_transformer.untransform(y_t), y)

  def test_singletask_balancing_transformer(self):
    """Test balancing transformer on single-task dataset."""

    classification_dataset = dc.data.tests.load_classification_data()
    balancing_transformer = dc.trans.BalancingTransformer(
        transform_w=True, dataset=classification_dataset)
    X, y, w, ids = (classification_dataset.X, classification_dataset.y,
                    classification_dataset.w, classification_dataset.ids)
    classification_dataset = balancing_transformer.transform(
        classification_dataset)
    X_t, y_t, w_t, ids_t = (classification_dataset.X, classification_dataset.y,
                            classification_dataset.w,
                            classification_dataset.ids)
    # Check ids are unchanged.
    for id_elt, id_t_elt in zip(ids, ids_t):
      assert id_elt == id_t_elt
    # Check X is unchanged since this is a w transformer
    np.testing.assert_allclose(X, X_t)
    # Check y is unchanged since this is a w transformer
    np.testing.assert_allclose(y, y_t)
    for ind, task in enumerate(classification_dataset.get_task_names()):
      y_task = y_t[:, ind]
      w_task = w_t[:, ind]
      w_orig_task = w[:, ind]
      # Assert that entries with zero weight retain zero weight
      np.testing.assert_allclose(w_task[w_orig_task == 0],
                                 np.zeros_like(w_task[w_orig_task == 0]))
      # Check that sum of 0s equals sum of 1s in transformed for each task
      assert np.isclose(
          np.sum(w_task[y_task == 0]), np.sum(w_task[y_task == 1]))

  def test_multitask_balancing_transformer(self):
    """Test balancing transformer on multitask dataset."""
    multitask_dataset = dc.data.tests.load_multitask_data()
    balancing_transformer = dc.trans.BalancingTransformer(
        transform_w=True, dataset=multitask_dataset)
    X, y, w, ids = (multitask_dataset.X, multitask_dataset.y,
                    multitask_dataset.w, multitask_dataset.ids)
    multitask_dataset = balancing_transformer.transform(multitask_dataset)
    X_t, y_t, w_t, ids_t = (multitask_dataset.X, multitask_dataset.y,
                            multitask_dataset.w, multitask_dataset.ids)
    # Check ids are unchanged.
    for id_elt, id_t_elt in zip(ids, ids_t):
      assert id_elt == id_t_elt
    # Check X is unchanged since this is a w transformer
    np.testing.assert_allclose(X, X_t)
    # Check y is unchanged since this is a w transformer
    np.testing.assert_allclose(y, y_t)
    for ind, task in enumerate(multitask_dataset.get_task_names()):
      y_task = y_t[:, ind]
      w_task = w_t[:, ind]
      w_orig_task = w[:, ind]
      # Assert that entries with zero weight retain zero weight
      np.testing.assert_allclose(w_task[w_orig_task == 0],
                                 np.zeros_like(w_task[w_orig_task == 0]))
      # Check that sum of 0s equals sum of 1s in transformed for each task
      assert np.isclose(
          np.sum(w_task[y_task == 0]), np.sum(w_task[y_task == 1]))

  def test_coulomb_fit_transformer(self):
    """Test coulomb fit transformer on singletask dataset."""
    n_samples = 10
    n_features = 3
    n_tasks = 1
    ids = np.arange(n_samples)
    X = np.random.rand(n_samples, n_features, n_features)
    y = np.zeros((n_samples, n_tasks))
    w = np.ones((n_samples, n_tasks))
    dataset = dc.data.NumpyDataset(X, y, w, ids)
    fit_transformer = dc.trans.CoulombFitTransformer(dataset)
    X_t = fit_transformer.X_transform(dataset.X)
    assert len(X_t.shape) == 2

  def test_IRV_transformer(self):
    n_features = 128
    n_samples = 20
    test_samples = 5
    n_tasks = 2
    X = np.random.randint(2, size=(n_samples, n_features))
    y = np.zeros((n_samples, n_tasks))
    w = np.ones((n_samples, n_tasks))
    dataset = dc.data.NumpyDataset(X, y, w, ids=None)
    X_test = np.random.randint(2, size=(test_samples, n_features))
    y_test = np.zeros((test_samples, n_tasks))
    w_test = np.ones((test_samples, n_tasks))
    test_dataset = dc.data.NumpyDataset(X_test, y_test, w_test, ids=None)
    sims = np.sum(
        X_test[0, :] * X, axis=1, dtype=float) / np.sum(
            np.sign(X_test[0, :] + X), axis=1, dtype=float)
    sims = sorted(sims, reverse=True)
    IRV_transformer = dc.trans.IRVTransformer(10, n_tasks, dataset)
    test_dataset_trans = IRV_transformer.transform(test_dataset)
    dataset_trans = IRV_transformer.transform(dataset)
    assert test_dataset_trans.X.shape == (test_samples, 20 * n_tasks)
    assert np.allclose(test_dataset_trans.X[0, :10], sims[:10])
    assert np.allclose(test_dataset_trans.X[0, 10:20], [0] * 10)
    assert not np.isclose(dataset_trans.X[0, 0], 1.)
