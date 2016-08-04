"""
Contains an abstract base class that supports chemically aware data splits.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Bharath Ramsundar, Aneesh Pappu "
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "GPL"

import tempfile
import numpy as np
from rdkit import Chem
from deepchem.utils import ScaffoldGenerator
from deepchem.utils.save import log
from deepchem.datasets import NumpyDataset
from deepchem.featurizers.featurize import load_data

def generate_scaffold(smiles, include_chirality=False):
  """Compute the Bemis-Murcko scaffold for a SMILES string."""
  mol = Chem.MolFromSmiles(smiles)
  engine = ScaffoldGenerator(include_chirality=include_chirality)
  scaffold = engine.get_scaffold(mol)
  return scaffold

def randomize_arrays(array_list):
  # assumes that every array is of the same dimension
  num_rows = array_list[0].shape[0]
  perm = np.random.permutation(num_rows)
  permuted_arrays = []
  for array in array_list:
    permuted_arrays.append(array[perm])
  return permuted_arrays 

class Splitter(object):
  """
  Abstract base class for chemically aware splits..
  """
  def __init__(self, verbosity=None):
    """Creates splitter object."""
    self.verbosity = verbosity

  def k_fold_split(self, dataset, directories=None, compute_feature_statistics=True):
    """Does K-fold split of dataset."""
    log("Computing K-fold split", self.verbosity)
    k = len(directories)
    fold_datasets = []
    # rem_dataset is remaining portion of dataset
    rem_dataset = dataset
    for fold in range(k):
      # Note starts as 1/k since fold starts at 0. Ends at 1 since fold goes up
      # to k-1.
      frac_fold = 1./(k-fold)
      fold_dir = directories[fold]
      fold_inds, rem_inds, _ = self.split(
          rem_dataset,
          frac_train=frac_fold, frac_valid=1-frac_fold, frac_test=0)
      fold_dataset = rem_dataset.select( 
          fold_dir, fold_inds,
          compute_feature_statistics=compute_feature_statistics)
      rem_dir = tempfile.mkdtemp()
      rem_dataset = rem_dataset.select( 
          rem_dir, rem_inds,
          compute_feature_statistics=compute_feature_statistics)
      fold_datasets.append(fold_dataset)
    return fold_datasets

  def train_valid_test_split(self, dataset, train_dir=None,
                             valid_dir=None, test_dir=None, frac_train=.8,
                             frac_valid=.1, frac_test=.1, seed=None,
                             log_every_n=1000,
                             compute_feature_statistics=True):
    """
    Splits self into train/validation/test sets.

    Returns Dataset objects.
    """
    log("Computing train/valid/test indices", self.verbosity)
    train_inds, valid_inds, test_inds = self.split(
      dataset,
      frac_train=frac_train, frac_test=frac_test,
      frac_valid=frac_valid, log_every_n=log_every_n)
    train_dataset = dataset.select( 
        train_dir, train_inds,
        compute_feature_statistics=compute_feature_statistics)
    if valid_dir is not None:
      valid_dataset = dataset.select(
          valid_dir, valid_inds,
          compute_feature_statistics=compute_feature_statistics)
    else:
      valid_dataset = None
    test_dataset = dataset.select(
        test_dir, test_inds,
        compute_feature_statistics=compute_feature_statistics)

    return train_dataset, valid_dataset, test_dataset

  def train_test_split(self, samples, train_dir, test_dir, seed=None,
                       frac_train=.8, compute_feature_statistics=True):
    """
    Splits self into train/test sets.
    Returns Dataset objects.
    """
    valid_dir = tempfile.mkdtemp()
    train_samples, _, test_samples = self.train_valid_test_split(
      samples, train_dir, valid_dir, test_dir,
      frac_train=frac_train, frac_test=1-frac_train, frac_valid=0.,
      compute_feature_statistics=compute_feature_statistics)
    return train_samples, test_samples

  def split(self, dataset, frac_train=None, frac_valid=None, frac_test=None,
            log_every_n=None):
    """
    Stub to be filled in by child classes.
    """
    raise NotImplementedError

  
class RandomStratifiedSplitter(Splitter):
  """
  RandomStratified Splitter class.

  For sparse multitask datasets, a standard split offers no guarantees that the
  splits will have any activate compounds. This class guarantees that each task
  will have a proportional split of the activates in a split. TO do this, a
  ragged split is performed with different numbers of compounds taken from each
  task. Thus, the length of the split arrays may exceed the split of the
  original array. That said, no datapoint is copied to more than one split, so
  correctness is still ensured.

  Note that this splitter is only valid for boolean label data.

  TODO(rbharath): This splitter should be refactored to match style of other
  splitter classes.
  """

  def __randomize_arrays(self, array_list):
    # assumes that every array is of the same dimension
    num_rows = array_list[0].shape[0]
    perm = np.random.permutation(num_rows)
    array_list = [array[perm] for array in array_list]
    return array_list
  
  def __generate_required_hits(self, w, frac_split):
    # returns list of per column sum of non zero elements
    required_hits = (w != 0).sum(axis=0)  
    for col_hits in required_hits:
      col_hits = int(frac_split * col_hits)
    return required_hits

  def get_task_split_indices(self, y, w, frac_split):
    """Returns num datapoints needed per task to split properly."""
    w_present = (w != 0)
    y_present = y * w_present

    # Compute number of actives needed per task.
    task_actives = np.sum(y_present, axis=0)
    task_split_actives = (frac_split*task_actives).astype(int)
    
    # loop through each column and obtain index required to splice out for
    # required fraction of hits
    split_indices = []
    n_tasks = np.shape(y)[1]
    for task in range(n_tasks):
      actives_count = task_split_actives[task]
      cum_task_actives = np.cumsum(y_present[:, task])
      # Find the first index where the cumulative number of actives equals
      # the actives_count
      split_index = np.amin(np.where(cum_task_actives >= actives_count)[0])
      # Note that np.where tells us last index required to exceed
      # actives_count, so we actually want the following location
      split_indices.append(split_index+1)
    return split_indices 

  # TODO(rbharath): Refactor this split method to match API of other splits (or
  # potentially refactor those to match this.
  def split(self, dataset, split_dirs, frac_split):
    """
    Method that does bulk of splitting dataset.
    """
    assert len(split_dirs) == 2
    # Handle edge case where frac_split is 1
    if frac_split == 1:
      dataset_1 = NumpyDataset(dataset.X, dataset.y, dataset.w, dataset.ids)
      dataset_2 = None 
      return dataset_1, dataset_2
    X, y, w, ids = randomize_arrays((dataset.X, dataset.y, dataset.w, dataset.ids))
    split_indices = self.get_task_split_indices(y, w, frac_split)

    # Create weight matrices fpor two haves. 
    w_1, w_2 = np.zeros_like(w), np.zeros_like(w)
    for task, split_index in enumerate(split_indices):
      # copy over up to required index for weight first_split
      w_1[:split_index, task] = w[:split_index, task]
      w_2[split_index:, task] = w[split_index:, task]

    # check out if any rows in either w_1 or w_2 are just zeros
    rows_1 = w_1.any(axis=1)
    X_1, y_1, w_1, ids_1 = X[rows_1], y[rows_1], w_1[rows_1], ids[rows_1]
    dataset_1 = NumpyDataset(X_1, y_1, w_1, ids_1)

    rows_2 = w_2.any(axis=1)
    X_2, y_2, w_2, ids_2 = X[rows_2], y[rows_2], w_2[rows_2], ids[rows_2]
    dataset_2 = NumpyDataset(X_2, y_2, w_2, ids_2)

    return dataset_1, dataset_2 

  def train_valid_test_split(self, dataset, train_dir,
                             valid_dir, test_dir, frac_train=.8,
                             frac_valid=.1, frac_test=.1, seed=None,
                             log_every_n=1000):
    """Custom split due to raggedness in original split.
    """
    # Obtain original x, y, and w arrays and shuffle
    X, y, w, ids = randomize_arrays((dataset.X, dataset.y, dataset.w, dataset.ids))
    rem_dir = tempfile.mkdtemp()
    train_dataset, rem_dataset = self.split(
        dataset, [train_dir, rem_dir], frac_train)

    # calculate percent split for valid (out of test and valid)
    if frac_valid + frac_test > 0:
      valid_percentage = frac_valid / (frac_valid + frac_test)
    else:
      return train_dataset, None, None
    # split test data into valid and test, treating sub test set also as sparse
    valid_dataset, test_dataset = self.split(
        dataset, [valid_dir, test_dir], valid_percentage)

    return train_dataset, valid_dataset, test_dataset

  def k_fold_split(self, dataset, directories, compute_feature_statistics=True):
    """Needs custom implementation due to ragged splits for stratification."""
    log("Computing K-fold split", self.verbosity)
    k = len(directories)
    fold_datasets = []
    # rem_dataset is remaining portion of dataset
    rem_dataset = dataset
    for fold in range(k):
      # Note starts as 1/k since fold starts at 0. Ends at 1 since fold goes up
      # to k-1.
      frac_fold = 1./(k-fold)
      fold_dir = directories[fold]
      rem_dir = tempfile.mkdtemp()
      fold_dataset, rem_dataset = self.split(
          rem_dataset, [fold_dir, rem_dir], frac_split=frac_fold)
      fold_datasets.append(fold_dataset)
    return fold_datasets


class MolecularWeightSplitter(Splitter):
  """
  Class for doing data splits by molecular weight.
  """

  def split(self, dataset, seed=None, frac_train=.8, frac_valid=.1,
            frac_test=.1, log_every_n=None):
    """
    Splits internal compounds into train/validation/test using the MW calculated
    by SMILES string.
    """

    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.)
    np.random.seed(seed)

    mws = []
    for smiles in dataset.ids:
      mol = Chem.MolFromSmiles(smiles)
      mw = Chem.rdMolDescriptors.CalcExactMolWt(mol)
      mws.append(mw)

    # Sort by increasing MW
    mws = np.array(mws)
    sortidx = np.argsort(mws)

    train_cutoff = frac_train * len(sortidx)
    valid_cutoff = (frac_train + frac_valid) * len(sortidx)

    return (sortidx[:train_cutoff], sortidx[train_cutoff:valid_cutoff],
            sortidx[valid_cutoff:])


class RandomSplitter(Splitter):
  """
  Class for doing random data splits.
  """

  def split(self, dataset, seed=None, frac_train=.8, frac_valid=.1,
            frac_test=.1, log_every_n=None):
    """
    Splits internal compounds randomly into train/validation/test.
    """
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.)
    np.random.seed(seed)
    num_datapoints = len(dataset)
    train_cutoff = int(frac_train * num_datapoints)
    valid_cutoff = int((frac_train + frac_valid) * num_datapoints)
    shuffled = np.random.permutation(range(num_datapoints))
    return (shuffled[:train_cutoff], shuffled[train_cutoff:valid_cutoff],
            shuffled[valid_cutoff:])

class IndexSplitter(Splitter):
  """
  Class for simple order based splits. 
  """

  def split(self, dataset, seed=None, frac_train=.8, frac_valid=.1,
            frac_test=.1, log_every_n=None):
    """
    Splits internal compounds into train/validation/test in provided order.
    """
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.)
    num_datapoints = len(dataset)
    train_cutoff = int(frac_train * num_datapoints)
    valid_cutoff = int((frac_train + frac_valid) * num_datapoints)
    indices = range(num_datapoints)
    return (indices[:train_cutoff], indices[train_cutoff:valid_cutoff],
            indices[valid_cutoff:])


class ScaffoldSplitter(Splitter):
  """
  Class for doing data splits based on the scaffold of small molecules.
  """

  def split(self, dataset, frac_train=.8, frac_valid=.1, frac_test=.1,
            log_every_n=1000):
    """
    Splits internal compounds into train/validation/test by scaffold.
    """
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.)
    scaffolds = {}
    log("About to generate scaffolds", self.verbosity)
    data_len = len(dataset)
    for ind, smiles in enumerate(dataset.ids):
      if ind % log_every_n == 0:
        log("Generating scaffold %d/%d" % (ind, data_len), self.verbosity)
      scaffold = generate_scaffold(smiles)
      if scaffold not in scaffolds:
        scaffolds[scaffold] = [ind]
      else:
        scaffolds[scaffold].append(ind)
    # Sort from largest to smallest scaffold sets
    scaffold_sets = [scaffold_set for (scaffold, scaffold_set) in
                     sorted(scaffolds.items(), key=lambda x: -len(x[1]))]
    train_cutoff = frac_train * len(dataset)
    valid_cutoff = (frac_train + frac_valid) * len(dataset)
    train_inds, valid_inds, test_inds = [], [], []
    log("About to sort in scaffold sets", self.verbosity)
    for scaffold_set in scaffold_sets:
      if len(train_inds) + len(scaffold_set) > train_cutoff:
        if len(train_inds) + len(valid_inds) + len(scaffold_set) > valid_cutoff:
          test_inds += scaffold_set
        else:
          valid_inds += scaffold_set
      else:
        train_inds += scaffold_set
    return train_inds, valid_inds, test_inds


class SpecifiedSplitter(Splitter):
  """
  Class that splits data according to user specification.
  """

  def __init__(self, input_file, split_field, verbosity=None):
    """Provide input information for splits."""
    raw_df = next(load_data([input_file], shard_size=None))
    self.splits = raw_df[split_field].values
    self.verbosity = verbosity

  def split(self, dataset, frac_train=.8, frac_valid=.1, frac_test=.1,
            log_every_n=1000):
    """
    Splits internal compounds into train/validation/test by user-specification.
    """
    train_inds, valid_inds, test_inds = [], [], []
    for ind, split in enumerate(self.splits):
      split = split.lower()
      if split == "train":
        train_inds.append(ind)
      elif split in ["valid", "validation"]:
        valid_inds.append(ind)
      elif split == "test":
        test_inds.append(ind)
      else:
        raise ValueError("Missing required split information.")
    return train_inds, valid_inds, test_inds
