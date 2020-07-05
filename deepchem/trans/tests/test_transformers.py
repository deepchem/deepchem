"""
Tests for transformer objects.
"""
from deepchem.molnet import load_delaney
from deepchem.trans.transformers import FeaturizationTransformer
from deepchem.trans.transformers import DataTransforms

__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "MIT"

import os
import unittest
import numpy as np
import pandas as pd
import deepchem as dc
import tensorflow as tf
import scipy.ndimage


def load_classification_data():
  """Loads classification data from example.csv"""
  current_dir = os.path.dirname(os.path.abspath(__file__))
  featurizer = dc.feat.CircularFingerprint(size=1024)
  tasks = ["outcome"]
  task_type = "classification"
  input_file = os.path.join(current_dir,
                            "../../models/tests/example_classification.csv")
  loader = dc.data.CSVLoader(
      tasks=tasks, smiles_field="smiles", featurizer=featurizer)
  return loader.featurize(input_file)


def load_multitask_data():
  """Load example multitask data."""
  current_dir = os.path.dirname(os.path.abspath(__file__))
  featurizer = dc.feat.CircularFingerprint(size=1024)
  tasks = [
      "task0", "task1", "task2", "task3", "task4", "task5", "task6", "task7",
      "task8", "task9", "task10", "task11", "task12", "task13", "task14",
      "task15", "task16"
  ]
  input_file = os.path.join(current_dir,
                            "../../models/tests/multitask_example.csv")
  loader = dc.data.CSVLoader(
      tasks=tasks, smiles_field="smiles", featurizer=featurizer)
  return loader.featurize(input_file)


def load_solubility_data():
  """Loads solubility dataset"""
  current_dir = os.path.dirname(os.path.abspath(__file__))
  featurizer = dc.feat.CircularFingerprint(size=1024)
  tasks = ["log-solubility"]
  task_type = "regression"
  input_file = os.path.join(current_dir, "../../models/tests/example.csv")
  loader = dc.data.CSVLoader(
      tasks=tasks, smiles_field="smiles", featurizer=featurizer)

  return loader.create_dataset(input_file)


def load_feat_multitask_data():
  """Load example with numerical features, tasks."""
  current_dir = os.path.dirname(os.path.abspath(__file__))
  features = ["feat0", "feat1", "feat2", "feat3", "feat4", "feat5"]
  featurizer = dc.feat.UserDefinedFeaturizer(features)
  tasks = ["task0", "task1", "task2", "task3", "task4", "task5"]
  input_file = os.path.join(current_dir,
                            "../../models/tests/feat_multitask_example.csv")
  loader = dc.data.UserCSVLoader(
      tasks=tasks, featurizer=featurizer, id_field="id")
  return loader.featurize(input_file)


def load_gaussian_cdf_data():
  """Load example with numbers sampled from Gaussian normal distribution.
     Each feature and task is a column of values that is sampled
     from a normal distribution of mean 0, stdev 1."""
  current_dir = os.path.dirname(os.path.abspath(__file__))
  features = ["feat0", "feat1"]
  featurizer = dc.feat.UserDefinedFeaturizer(features)
  tasks = ["task0", "task1"]
  input_file = os.path.join(current_dir,
                            "../../models/tests/gaussian_cdf_example.csv")
  loader = dc.data.UserCSVLoader(
      tasks=tasks, featurizer=featurizer, id_field="id")
  return loader.featurize(input_file)


def load_unlabelled_data():
  current_dir = os.path.dirname(os.path.abspath(__file__))
  featurizer = dc.feat.CircularFingerprint(size=1024)
  tasks = []
  input_file = os.path.join(current_dir, "../../data/tests/no_labels.csv")
  loader = dc.data.CSVLoader(
      tasks=tasks, smiles_field="smiles", featurizer=featurizer)
  return loader.featurize(input_file)


class TestTransformers(unittest.TestCase):
  """
  Test top-level API for transformer objects.
  """

  def setUp(self):
    super(TestTransformers, self).setUp()
    self.current_dir = os.path.dirname(os.path.abspath(__file__))
    '''
       init to load the MNIST data for DataTransforms Tests
      '''
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    train = dc.data.NumpyDataset(x_train, y_train)
    # extract only the images (no need of the labels)
    data = (train.X)[0]
    # reshaping the vector to image
    data = np.reshape(data, (28, 28))
    self.d = data

  def test_y_log_transformer(self):
    """Tests logarithmic data transformer."""
    solubility_dataset = load_solubility_data()
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
    ul_dataset = load_unlabelled_data()
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
    solubility_dataset = load_solubility_data()
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
    multitask_dataset = load_feat_multitask_data()
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
    # Tests logarithmic data transformer with selection.
    multitask_dataset = load_feat_multitask_data()
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

  def test_y_minmax_transformer(self):
    """Tests MinMax transformer. """
    solubility_dataset = load_solubility_data()
    minmax_transformer = dc.trans.MinMaxTransformer(
        transform_y=True, dataset=solubility_dataset)
    X, y, w, ids = (solubility_dataset.X, solubility_dataset.y,
                    solubility_dataset.w, solubility_dataset.ids)
    solubility_dataset = minmax_transformer.transform(solubility_dataset)
    X_t, y_t, w_t, ids_t = (solubility_dataset.X, solubility_dataset.y,
                            solubility_dataset.w, solubility_dataset.ids)

    # Check ids are unchanged before and after transformation
    for id_elt, id_t_elt in zip(ids, ids_t):
      assert id_elt == id_t_elt

    # Check X is unchanged since transform_y is true
    np.testing.assert_allclose(X, X_t)
    # Check w is unchanged since transform_y is true
    np.testing.assert_allclose(w, w_t)

    # Check minimum and maximum values of transformed y are 0 and 1
    np.testing.assert_allclose(y_t.min(), 0.)
    np.testing.assert_allclose(y_t.max(), 1.)

    # Check untransform works correctly
    np.testing.assert_allclose(minmax_transformer.untransform(y_t), y)

    # Test on random example
    n_samples = 100
    n_features = 10
    n_tasks = 10

    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples, n_tasks)
    dataset = dc.data.NumpyDataset(X, y)

    minmax_transformer = dc.trans.MinMaxTransformer(
        transform_y=True, dataset=dataset)
    w, ids = dataset.w, dataset.ids

    dataset = minmax_transformer.transform(dataset)
    X_t, y_t, w_t, ids_t = (dataset.X, dataset.y, dataset.w, dataset.ids)
    # Check ids are unchanged before and after transformation
    for id_elt, id_t_elt in zip(ids, ids_t):
      assert id_elt == id_t_elt

    # Check X is unchanged since transform_y is true
    np.testing.assert_allclose(X, X_t)
    # Check w is unchanged since transform_y is true
    np.testing.assert_allclose(w, w_t)

    # Check minimum and maximum values of transformed y are 0 and 1
    np.testing.assert_allclose(y_t.min(), 0.)
    np.testing.assert_allclose(y_t.max(), 1.)

    # Test if dimensionality expansion is handled correctly by untransform
    y_t = np.expand_dims(y_t, axis=-1)
    y_restored = minmax_transformer.untransform(y_t)
    assert y_restored.shape == y.shape + (1,)
    np.testing.assert_allclose(np.squeeze(y_restored, axis=-1), y)

  def test_X_minmax_transformer(self):
    solubility_dataset = load_solubility_data()
    minmax_transformer = dc.trans.MinMaxTransformer(
        transform_X=True, dataset=solubility_dataset)
    X, y, w, ids = (solubility_dataset.X, solubility_dataset.y,
                    solubility_dataset.w, solubility_dataset.ids)
    solubility_dataset = minmax_transformer.transform(solubility_dataset)
    X_t, y_t, w_t, ids_t = (solubility_dataset.X, solubility_dataset.y,
                            solubility_dataset.w, solubility_dataset.ids)

    # Check ids are unchanged before and after transformation
    for id_elt, id_t_elt in zip(ids, ids_t):
      assert id_elt == id_t_elt

    # Check X is unchanged since transform_y is true
    np.testing.assert_allclose(y, y_t)
    # Check w is unchanged since transform_y is true
    np.testing.assert_allclose(w, w_t)

    # Check minimum and maximum values of transformed y are 0 and 1
    np.testing.assert_allclose(X_t.min(), 0.)
    np.testing.assert_allclose(X_t.max(), 1.)

    # Check untransform works correctly
    np.testing.assert_allclose(minmax_transformer.untransform(X_t), X)

  def test_y_normalization_transformer(self):
    """Tests normalization transformer."""
    solubility_dataset = load_solubility_data()
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
    solubility_dataset = load_solubility_data()
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
    # np.set_printoptions(threshold='nan')
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
    # np.testing.assert_allclose(normalization_transformer.untransform(X_t), X)

  def test_cdf_X_transformer(self):
    """Test CDF transformer on Gaussian normal dataset."""
    target = np.array(np.transpose(np.linspace(0., 1., 1001)))
    target = np.transpose(np.array(np.append([target], [target], axis=0)))
    gaussian_dataset = load_gaussian_cdf_data()
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
    # Test CDF transformer on Gaussian normal dataset.
    target = np.array(np.transpose(np.linspace(0., 1., 1001)))
    target = np.transpose(np.array(np.append([target], [target], axis=0)))
    gaussian_dataset = load_gaussian_cdf_data()
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
    gaussian_dataset = load_gaussian_cdf_data()
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
    gaussian_dataset = load_gaussian_cdf_data()
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

    classification_dataset = load_classification_data()
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
    multitask_dataset = load_multitask_data()
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

  def test_featurization_transformer(self):
    fp_size = 2048
    tasks, all_dataset, transformers = load_delaney('Raw')
    train = all_dataset[0]
    transformer = FeaturizationTransformer(
        transform_X=True,
        dataset=train,
        featurizer=dc.feat.CircularFingerprint(size=fp_size))
    new_train = transformer.transform(train)

    self.assertEqual(new_train.y.shape, train.y.shape)
    self.assertEqual(new_train.X.shape[-1], fp_size)

  def test_blurring(self):
    # Check Blurring
    dt = DataTransforms(self.d)
    blurred = dt.gaussian_blur(sigma=1.5)
    check_blur = scipy.ndimage.gaussian_filter(self.d, 1.5)
    assert np.allclose(check_blur, blurred)

  def test_center_crop(self):
    # Check center crop
    dt = DataTransforms(self.d)
    x_crop = 50
    y_crop = 50
    crop = dt.center_crop(x_crop, y_crop)
    y = self.d.shape[0]
    x = self.d.shape[1]
    x_start = x // 2 - (x_crop // 2)
    y_start = y // 2 - (y_crop // 2)
    check_crop = self.d[y_start:y_start + y_crop, x_start:x_start + x_crop]
    assert np.allclose(check_crop, crop)

  def test_crop(self):
    #Check crop
    dt = DataTransforms(self.d)
    crop = dt.crop(0, 10, 0, 10)
    y = self.d.shape[0]
    x = self.d.shape[1]
    check_crop = self.d[10:y - 10, 0:x - 0]
    assert np.allclose(crop, check_crop)

  def test_convert2gray(self):
    # Check convert2gray
    dt = DataTransforms(self.d)
    gray = dt.convert2gray()
    check_gray = np.dot(self.d[..., :3], [0.2989, 0.5870, 0.1140])
    assert np.allclose(check_gray, gray)

  def test_rotation(self):
    # Check rotation
    dt = DataTransforms(self.d)
    angles = [0, 5, 10, 90]
    for ang in angles:
      rotate = dt.rotate(ang)
      check_rotate = scipy.ndimage.rotate(self.d, ang)
      assert np.allclose(rotate, check_rotate)

    # Some more test cases for flip
    rotate = dt.rotate(-90)
    check_rotate = scipy.ndimage.rotate(self.d, 270)
    assert np.allclose(rotate, check_rotate)

  def test_flipping(self):
    # Check flip
    dt = DataTransforms(self.d)
    flip_lr = dt.flip(direction="lr")
    flip_ud = dt.flip(direction="ud")
    check_lr = np.fliplr(self.d)
    check_ud = np.flipud(self.d)
    assert np.allclose(flip_ud, check_ud)
    assert np.allclose(flip_lr, check_lr)

  def test_scaling(self):
    from PIL import Image
    # Check Scales
    dt = DataTransforms(self.d)
    h = 150
    w = 150
    scale = Image.fromarray(self.d).resize((h, w))
    check_scale = dt.scale(h, w)
    np.allclose(scale, check_scale)

  def test_shift(self):
    # Check shift
    dt = DataTransforms(self.d)
    height = 5
    width = 5
    if len(self.d.shape) == 2:
      shift = scipy.ndimage.shift(self.d, [height, width])
    if len(self.d.shape) == 3:
      shift = scipy.ndimage.shift(self.d, [height, width, 0])
    check_shift = dt.shift(width, height)
    assert np.allclose(shift, check_shift)

  def test_gaussian_noise(self):
    # check gaussian noise
    dt = DataTransforms(self.d)
    np.random.seed(0)
    random_noise = self.d
    random_noise = random_noise + np.random.normal(
        loc=0, scale=25.5, size=self.d.shape)
    np.random.seed(0)
    check_random_noise = dt.gaussian_noise(mean=0, std=25.5)
    assert np.allclose(random_noise, check_random_noise)

  def test_salt_pepper_noise(self):
    # check salt and pepper noise
    dt = DataTransforms(self.d)
    np.random.seed(0)
    prob = 0.05
    random_noise = self.d
    noise = np.random.random(size=self.d.shape)
    random_noise[noise < (prob / 2)] = 0
    random_noise[noise > (1 - prob / 2)] = 255
    np.random.seed(0)
    check_random_noise = dt.salt_pepper_noise(prob, salt=255, pepper=0)
    assert np.allclose(random_noise, check_random_noise)

  def test_DAG_transformer(self):
    """Tests the DAG transformer."""
    np.random.seed(123)
    tf.random.set_seed(123)
    n_tasks = 1

    # Load mini log-solubility dataset.
    featurizer = dc.feat.ConvMolFeaturizer()
    tasks = ["outcome"]
    input_file = os.path.join(self.current_dir,
                              "../../models/tests/example_regression.csv")
    loader = dc.data.CSVLoader(
        tasks=tasks, smiles_field="smiles", featurizer=featurizer)
    dataset = loader.create_dataset(input_file)
    transformer = dc.trans.DAGTransformer(max_atoms=50)
    dataset = transformer.transform(dataset)
    # The transformer generates n DAGs for a molecule with n
    # atoms. These are denoted the "parents"
    for idm, mol in enumerate(dataset.X):
      assert dataset.X[idm].get_num_atoms() == len(dataset.X[idm].parents)

  def test_median_filter(self):
    #Check median filter
    from PIL import Image, ImageFilter
    dt = DataTransforms(self.d)
    filtered = dt.median_filter(size=3)
    image = Image.fromarray(self.d)
    image = image.filter(ImageFilter.MedianFilter(size=3))
    check_filtered = np.array(image)
    assert np.allclose(check_filtered, filtered)
