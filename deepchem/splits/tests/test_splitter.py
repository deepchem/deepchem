"""
Tests for splitter objects.
"""
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

__author__ = "Bharath Ramsundar, Aneesh Pappu"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "GPL"

import tempfile
import numpy as np
from deepchem.datasets import DiskDataset
from deepchem.splits import RandomSplitter
from deepchem.splits import IndexSplitter
from deepchem.splits import ScaffoldSplitter
from deepchem.splits import RandomStratifiedSplitter
from deepchem.datasets.tests import TestDatasetAPI


class TestSplitters(TestDatasetAPI):
  """
  Test some basic splitters.
  """

  def test_singletask_random_split(self):
    """
    Test singletask RandomSplitter class.
    """
    solubility_dataset = self.load_solubility_data()
    random_splitter = RandomSplitter()
    train_data, valid_data, test_data = \
        random_splitter.train_valid_test_split(
            solubility_dataset,
            self.train_dir, self.valid_dir, self.test_dir,
            frac_train=0.8, frac_valid=0.1, frac_test=0.1)
    assert len(train_data) == 8
    assert len(valid_data) == 1
    assert len(test_data) == 1

    merge_dir = tempfile.mkdtemp()
    merged_dataset = DiskDataset.merge(
        merge_dir, [train_data, valid_data, test_data])
    assert sorted(merged_dataset.ids) == (
           sorted(solubility_dataset.ids))

  def test_singletask_index_split(self):
    """
    Test singletask RandomSplitter class.
    """
    solubility_dataset = self.load_solubility_data()
    random_splitter = IndexSplitter()
    train_data, valid_data, test_data = \
        random_splitter.train_valid_test_split(
            solubility_dataset,
            self.train_dir, self.valid_dir, self.test_dir,
            frac_train=0.8, frac_valid=0.1, frac_test=0.1)
    assert len(train_data) == 8
    assert len(valid_data) == 1
    assert len(test_data) == 1

    merge_dir = tempfile.mkdtemp()
    merged_dataset = DiskDataset.merge(
        merge_dir, [train_data, valid_data, test_data])
    assert sorted(merged_dataset.ids) == (
           sorted(solubility_dataset.ids))

  def test_singletask_scaffold_split(self):
    """
    Test singletask ScaffoldSplitter class.
    """
    solubility_dataset = self.load_solubility_data()
    scaffold_splitter = ScaffoldSplitter()
    train_data, valid_data, test_data = \
        scaffold_splitter.train_valid_test_split(
            solubility_dataset,
            self.train_dir, self.valid_dir, self.test_dir,
            frac_train=0.8, frac_valid=0.1, frac_test=0.1)
    assert len(train_data) == 8
    assert len(valid_data) == 1
    assert len(test_data) == 1

  def test_singletask_random_k_fold_split(self):
    """
    Test singletask RandomSplitter class.
    """
    solubility_dataset = self.load_solubility_data()
    random_splitter = RandomSplitter()
    ids_set = set(solubility_dataset.ids)

    K = 5
    fold_dirs = [tempfile.mkdtemp() for i in range(K)]
    fold_datasets = random_splitter.k_fold_split(solubility_dataset, fold_dirs)
    for fold in range(K):
      fold_dataset = fold_datasets[fold]
      # Verify lengths is 10/k == 2
      assert len(fold_dataset) == 2
      # Verify that compounds in this fold are subset of original compounds
      fold_ids_set = set(fold_dataset.ids)
      assert fold_ids_set.issubset(ids_set)
      # Verify that no two folds have overlapping compounds.
      for other_fold in range(K):
        if fold == other_fold:
          continue
        other_fold_dataset = fold_datasets[other_fold]
        other_fold_ids_set = set(other_fold_dataset.ids)
        assert fold_ids_set.isdisjoint(other_fold_ids_set)

    merge_dir = tempfile.mkdtemp()
    merged_dataset = DiskDataset.merge(merge_dir, fold_datasets)
    assert len(merged_dataset) == len(solubility_dataset)
    assert sorted(merged_dataset.ids) == (
           sorted(solubility_dataset.ids))

  def test_singletask_index_k_fold_split(self):
    """
    Test singletask IndexSplitter class.
    """
    solubility_dataset = self.load_solubility_data()
    index_splitter = IndexSplitter()
    ids_set = set(solubility_dataset.ids)

    K = 5
    fold_dirs = [tempfile.mkdtemp() for i in range(K)]
    fold_datasets = index_splitter.k_fold_split(solubility_dataset, fold_dirs)

    for fold in range(K):
      fold_dataset = fold_datasets[fold]
      # Verify lengths is 10/k == 2
      assert len(fold_dataset) == 2
      # Verify that compounds in this fold are subset of original compounds
      fold_ids_set = set(fold_dataset.ids)
      assert fold_ids_set.issubset(ids_set)
      # Verify that no two folds have overlapping compounds.
      for other_fold in range(K):
        if fold == other_fold:
          continue
        other_fold_dataset = fold_datasets[other_fold]
        other_fold_ids_set = set(other_fold_dataset.ids)
        assert fold_ids_set.isdisjoint(other_fold_ids_set)

    merge_dir = tempfile.mkdtemp()
    merged_dataset = DiskDataset.merge(merge_dir, fold_datasets)
    assert len(merged_dataset) == len(solubility_dataset)
    assert sorted(merged_dataset.ids) == (
           sorted(solubility_dataset.ids))
    
  def test_singletask_scaffold_k_fold_split(self):
    """
    Test singletask ScaffoldSplitter class.
    """
    solubility_dataset = self.load_solubility_data()
    scaffold_splitter = ScaffoldSplitter()
    ids_set = set(solubility_dataset.ids)

    K = 5
    fold_dirs = [tempfile.mkdtemp() for i in range(K)]
    fold_datasets = scaffold_splitter.k_fold_split(
        solubility_dataset, fold_dirs)

    for fold in range(K):
      fold_dataset = fold_datasets[fold]
      # Verify lengths is 10/k == 2
      assert len(fold_dataset) == 2
      # Verify that compounds in this fold are subset of original compounds
      fold_ids_set = set(fold_dataset.ids)
      assert fold_ids_set.issubset(ids_set)
      # Verify that no two folds have overlapping compounds.
      for other_fold in range(K):
        if fold == other_fold:
          continue
        other_fold_dataset = fold_datasets[other_fold]
        other_fold_ids_set = set(other_fold_dataset.ids)
        assert fold_ids_set.isdisjoint(other_fold_ids_set)

    merge_dir = tempfile.mkdtemp()
    merged_dataset = DiskDataset.merge(merge_dir, fold_datasets)
    assert len(merged_dataset) == len(solubility_dataset)
    assert sorted(merged_dataset.ids) == (
           sorted(solubility_dataset.ids))

  def test_singletask_stratified_column_indices(self):
    """
    Test RandomStratifiedSplitter's split method on simple singletas.
    """
    # Test singletask case. 
    n_samples = 100
    n_positives = 20
    n_features = 10
    n_tasks = 1

    X = np.random.rand(n_samples, n_features)
    y = np.zeros((n_samples, n_tasks))
    y[:n_positives] = 1
    w = np.ones((n_samples, n_tasks))
    ids = np.arange(n_samples)
    stratified_splitter = RandomStratifiedSplitter()
    column_indices = stratified_splitter.get_task_split_indices(
        y, w, frac_split=.5)

    split_index = column_indices[0]
    # The split index should partition dataset in half.
    assert np.count_nonzero(y[:split_index]) == 10

  def test_singletask_stratified_column_indices_mask(self):
    """
    Test RandomStratifiedSplitter's split method on dataset with mask.
    """
    # Test singletask case. 
    n_samples = 100
    n_positives = 20
    n_features = 10
    n_tasks = 1

    # Test case where some weights are zero (i.e. masked)
    X = np.random.rand(n_samples, n_features)
    y = np.zeros((n_samples, n_tasks))
    y[:n_positives] = 1
    w = np.ones((n_samples, n_tasks))
    # Set half the positives to have zero weight
    w[:n_positives/2] = 0
    ids = np.arange(n_samples)

    stratified_splitter = RandomStratifiedSplitter()
    column_indices = stratified_splitter.get_task_split_indices(
        y, w, frac_split=.5)

    split_index = column_indices[0]
    # There are 10 nonzero actives.
    # The split index should partition this into half, so expect 5
    w_present = (w != 0)
    y_present = y * w_present
    assert np.count_nonzero(y_present[:split_index]) == 5

  def test_multitask_stratified_column_indices(self):
    """
    Test RandomStratifiedSplitter split on multitask dataset.
    """
    n_samples = 100
    n_features = 10
    n_tasks = 10
    X = np.random.rand(n_samples, n_features)
    p = .05 # proportion actives
    y = np.random.binomial(1, p, size=(n_samples, n_tasks))
    w = np.ones((n_samples, n_tasks))

    stratified_splitter = RandomStratifiedSplitter()
    split_indices = stratified_splitter.get_task_split_indices(
        y, w, frac_split=.5)

    for task in range(n_tasks):
      split_index = split_indices[task]
      task_actives = np.count_nonzero(y[:, task])
      # The split index should partition dataset in half.
      assert np.count_nonzero(y[:split_index, task]) == int(task_actives/2)

  def test_multitask_stratified_column_indices_masked(self):
    """
    Test RandomStratifiedSplitter split on multitask dataset.
    """
    n_samples = 200
    n_features = 10
    n_tasks = 10
    X = np.random.rand(n_samples, n_features)
    p = .05 # proportion actives
    y = np.random.binomial(1, p, size=(n_samples, n_tasks))
    w = np.ones((n_samples, n_tasks))
    # Mask half the examples
    w[:n_samples/2] = 0

    stratified_splitter = RandomStratifiedSplitter()
    split_indices = stratified_splitter.get_task_split_indices(
        y, w, frac_split=.5)

    w_present = (w != 0)
    y_present = y * w_present
    for task in range(n_tasks):
      split_index = split_indices[task]
      task_actives = np.count_nonzero(y_present[:, task])
      # The split index should partition dataset in half.
      assert np.count_nonzero(y_present[:split_index, task]) == int(task_actives/2)

  def test_singletask_stratified_split(self):
    """
    Test RandomStratifiedSplitter on a singletask split.
    """
    np.random.seed(2314)
    # Test singletask case. 
    n_samples = 20
    n_positives = 10
    n_features = 10
    n_tasks = 1

    X = np.random.rand(n_samples, n_features)
    y = np.zeros((n_samples, n_tasks))
    y[:n_positives] = 1
    w = np.ones((n_samples, n_tasks))
    ids = np.arange(n_samples)
    data_dir = tempfile.mkdtemp()
    dataset = DiskDataset.from_numpy(data_dir, X, y, w, ids)

    stratified_splitter = RandomStratifiedSplitter()
    split_dirs = [tempfile.mkdtemp(), tempfile.mkdtemp()]
    dataset_1, dataset_2 = stratified_splitter.split(
        dataset, split_dirs, frac_split=.5)
  
    # Should have split cleanly in half (picked random seed to ensure this)
    assert len(dataset_1) == 10
    assert len(dataset_2) == 10

    # Check positives are correctly distributed
    y_1 = dataset_1.y
    assert np.count_nonzero(y_1) == n_positives/2

    y_2 = dataset_2.y
    assert np.count_nonzero(y_2) == n_positives/2

  def test_singletask_stratified_k_fold_split(self):
    """
    Test RandomStratifiedSplitter k-fold class.
    """
    n_samples = 100
    n_positives = 20
    n_features = 10
    n_tasks = 1

    X = np.random.rand(n_samples, n_features)
    y = np.zeros(n_samples)
    y[:n_positives] = 1
    w = np.ones(n_samples)
    ids = np.arange(n_samples)

    data_dir = tempfile.mkdtemp()
    dataset = DiskDataset.from_numpy(data_dir, X, y, w, ids)
    
    stratified_splitter = RandomStratifiedSplitter()
    ids_set = set(dataset.ids)

    K = 5
    fold_dirs = [tempfile.mkdtemp() for i in range(K)]
    fold_datasets = stratified_splitter.k_fold_split(
        dataset, fold_dirs)

    for fold in range(K):
      fold_dataset = fold_datasets[fold]
      # Verify lengths is 100/k == 20
      # Note: This wouldn't work for multitask str
      # assert len(fold_dataset) == n_samples/K
      fold_labels = fold_dataset.y
      # Verify that each fold has n_positives/K = 4 positive examples.
      assert np.count_nonzero(fold_labels == 1) == n_positives/K
      # Verify that compounds in this fold are subset of original compounds
      fold_ids_set = set(fold_dataset.ids)
      assert fold_ids_set.issubset(ids_set)
      # Verify that no two folds have overlapping compounds.
      for other_fold in range(K):
        if fold == other_fold:
          continue
        other_fold_dataset = fold_datasets[other_fold]
        other_fold_ids_set = set(other_fold_dataset.ids)
        assert fold_ids_set.isdisjoint(other_fold_ids_set)

    merge_dir = tempfile.mkdtemp()
    merged_dataset = DiskDataset.merge(merge_dir, fold_datasets)
    assert len(merged_dataset) == len(dataset)
    assert sorted(merged_dataset.ids) == (
           sorted(dataset.ids))


  def test_multitask_random_split(self):
    """
    Test multitask RandomSplitter class.
    """
    multitask_dataset = self.load_multitask_data()
    random_splitter = RandomSplitter()
    train_data, valid_data, test_data = \
        random_splitter.train_valid_test_split(
            multitask_dataset,
            self.train_dir, self.valid_dir, self.test_dir,
            frac_train=0.8, frac_valid=0.1, frac_test=0.1)
    assert len(train_data) == 8
    assert len(valid_data) == 1
    assert len(test_data) == 1

  def test_multitask_index_split(self):
    """
    Test multitask IndexSplitter class.
    """
    multitask_dataset = self.load_multitask_data()
    index_splitter = IndexSplitter()
    train_data, valid_data, test_data = \
        index_splitter.train_valid_test_split(
            multitask_dataset,
            self.train_dir, self.valid_dir, self.test_dir,
            frac_train=0.8, frac_valid=0.1, frac_test=0.1)
    assert len(train_data) == 8
    assert len(valid_data) == 1
    assert len(test_data) == 1

  def test_multitask_scaffold_split(self):
    """
    Test multitask ScaffoldSplitter class.
    """
    multitask_dataset = self.load_multitask_data()
    scaffold_splitter = ScaffoldSplitter()
    train_data, valid_data, test_data = \
        scaffold_splitter.train_valid_test_split(
            multitask_dataset,
            self.train_dir, self.valid_dir, self.test_dir,
            frac_train=0.8, frac_valid=0.1, frac_test=0.1)
    assert len(train_data) == 8
    assert len(valid_data) == 1
    assert len(test_data) == 1

  def test_stratified_multitask_split(self):
    """
    Test multitask RandomStratifiedSplitter class
    """
    # sparsity is determined by number of w weights that are 0 for a given
    # task structure of w np array is such that each row corresponds to a
    # sample. The loaded sparse dataset has many rows with only zeros
    sparse_dataset = self.load_sparse_multitask_dataset()
    
    stratified_splitter = RandomStratifiedSplitter()
    datasets = stratified_splitter.train_valid_test_split(
        sparse_dataset,
        self.train_dir, self.valid_dir, self.test_dir,
        frac_train=0.8, frac_valid=0.1, frac_test=0.1)
    train_data, valid_data, test_data = datasets

    for dataset_index, dataset in enumerate(datasets):
      w = dataset.w
      # verify that there are no rows (samples) in weights matrix w
      # that have no hits.
      assert len(np.where(~w.any(axis=1))[0]) == 0
