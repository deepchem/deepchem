"""
Contains an abstract base class that supports chemically aware data splits.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Bharath Ramsundar, Aneesh Pappu "
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "MIT"

import tempfile
import numpy as np
import itertools
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina
import deepchem as dc
from deepchem.data import DiskDataset
from deepchem.utils import ScaffoldGenerator
from deepchem.utils.save import log
from deepchem.data import NumpyDataset
from deepchem.utils.save import load_data


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

  def __init__(self, verbose=False):
    """Creates splitter object."""
    self.verbose = verbose

  def k_fold_split(self, dataset, k, directories=None):
    """Does K-fold split of dataset."""
    log("Computing K-fold split", self.verbose)
    if directories is None:
      directories = [tempfile.mkdtemp() for _ in range(k)]
    else:
      assert len(directories) == k
    fold_datasets = []
    # rem_dataset is remaining portion of dataset
    rem_dataset = dataset
    for fold in range(k):
      # Note starts as 1/k since fold starts at 0. Ends at 1 since fold goes up
      # to k-1.
      frac_fold = 1. / (k - fold)
      fold_dir = directories[fold]
      fold_inds, rem_inds, _ = self.split(
          rem_dataset,
          frac_train=frac_fold,
          frac_valid=1 - frac_fold,
          frac_test=0)
      fold_dataset = rem_dataset.select(fold_inds, fold_dir)
      rem_dir = tempfile.mkdtemp()
      rem_dataset = rem_dataset.select(rem_inds, rem_dir)
      fold_datasets.append(fold_dataset)
    return fold_datasets

  def train_valid_test_split(self,
                             dataset,
                             train_dir=None,
                             valid_dir=None,
                             test_dir=None,
                             frac_train=.8,
                             frac_valid=.1,
                             frac_test=.1,
                             seed=None,
                             log_every_n=1000,
                             verbose=True):
    """
    Splits self into train/validation/test sets.

    Returns Dataset objects.
    """
    if (isinstance(dataset, NumpyDataset)):
      raise ValueError(
          "Only possible with DiskDataset.  NumpyDataset doesn't support .select"
      )
    log("Computing train/valid/test indices", self.verbose)
    train_inds, valid_inds, test_inds = self.split(
        dataset,
        frac_train=frac_train,
        frac_test=frac_test,
        frac_valid=frac_valid,
        log_every_n=log_every_n)
    if train_dir is None:
      train_dir = tempfile.mkdtemp()
    if valid_dir is None:
      valid_dir = tempfile.mkdtemp()
    if test_dir is None:
      test_dir = tempfile.mkdtemp()
    train_dataset = dataset.select(train_inds, train_dir)
    if frac_valid != 0:
      valid_dataset = dataset.select(valid_inds, valid_dir)
    else:
      valid_dataset = None
    test_dataset = dataset.select(test_inds, test_dir)

    return train_dataset, valid_dataset, test_dataset

  def train_valid_test_indices(self,
                               dataset,
                               train_dir=None,
                               valid_dir=None,
                               test_dir=None,
                               frac_train=.8,
                               frac_valid=.1,
                               frac_test=.1,
                               seed=None,
                               log_every_n=1000,
                               verbose=True):
    """
    Splits self into train/validation/test sets.

    Returns Dataset objects.
    """
    log("Computing train/valid/test indices", self.verbose)
    train_inds, valid_inds, test_inds = self.split(
        dataset,
        frac_train=frac_train,
        frac_test=frac_test,
        frac_valid=frac_valid,
        log_every_n=log_every_n)
    return train_inds, valid_inds, test_inds

  def train_test_split(self,
                       dataset,
                       train_dir=None,
                       test_dir=None,
                       seed=None,
                       frac_train=.8,
                       verbose=True):
    """
    Splits self into train/test sets.
    Returns Dataset objects.
    """
    valid_dir = tempfile.mkdtemp()
    train_dataset, _, test_dataset = self.train_valid_test_split(
        dataset,
        train_dir,
        valid_dir,
        test_dir,
        frac_train=frac_train,
        frac_test=1 - frac_train,
        frac_valid=0.,
        verbose=verbose)
    return train_dataset, test_dataset

  def train_valid_test_indices(self,
                               dataset,
                               train_dir=None,
                               valid_dir=None,
                               test_dir=None,
                               frac_train=.8,
                               frac_valid=.1,
                               frac_test=.1,
                               seed=None,
                               log_every_n=1000,
                               verbose=True):
    """
    Splits self into train/validation/test sets.

    Returns Dataset objects.
    """
    log("Computing train/valid/test indices", self.verbose)
    train_inds, valid_inds, test_inds = self.split(
        dataset,
        frac_train=frac_train,
        frac_test=frac_test,
        frac_valid=frac_valid,
        log_every_n=log_every_n)
    return train_inds, valid_inds, test_inds

  def train_test_indices(self,
                         dataset,
                         train_dir=None,
                         test_dir=None,
                         seed=None,
                         frac_train=.8,
                         verbose=True):
    """
    Splits self into train/test sets.
    Returns Dataset objects.
    """
    valid_dir = None
    train_inds, _, test_inds = self.train_valid_test_indices(
        dataset,
        train_dir,
        valid_dir,
        test_dir,
        frac_train=frac_train,
        frac_test=1 - frac_train,
        frac_valid=0.,
        verbose=verbose)
    return train_inds, test_inds

  def split(self,
            dataset,
            frac_train=None,
            frac_valid=None,
            frac_test=None,
            log_every_n=None,
            verbose=False):
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
    task_split_actives = (frac_split * task_actives).astype(int)

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
      split_indices.append(split_index + 1)
    return split_indices

    # TODO(rbharath): Refactor this split method to match API of other splits (or

  # potentially refactor those to match this.

  def split(self, dataset, frac_split, split_dirs=None):
    """
    Method that does bulk of splitting dataset.
    """
    if split_dirs is not None:
      assert len(split_dirs) == 2
    else:
      split_dirs = [tempfile.mkdtemp(), tempfile.mkdtemp()]

    # Handle edge case where frac_split is 1
    if frac_split == 1:
      dataset_1 = NumpyDataset(dataset.X, dataset.y, dataset.w, dataset.ids)
      dataset_2 = None
      return dataset_1, dataset_2
    X, y, w, ids = randomize_arrays((dataset.X, dataset.y, dataset.w,
                                     dataset.ids))
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

  def train_valid_test_split(self,
                             dataset,
                             train_dir=None,
                             valid_dir=None,
                             test_dir=None,
                             frac_train=.8,
                             frac_valid=.1,
                             frac_test=.1,
                             seed=None,
                             log_every_n=1000):
    """Custom split due to raggedness in original split.
    """
    if train_dir is None:
      train_dir = tempfile.mkdtemp()
    if valid_dir is None:
      valid_dir = tempfile.mkdtemp()
    if test_dir is None:
      test_dir = tempfile.mkdtemp()
    # Obtain original x, y, and w arrays and shuffle
    X, y, w, ids = randomize_arrays((dataset.X, dataset.y, dataset.w,
                                     dataset.ids))
    rem_dir = tempfile.mkdtemp()
    train_dataset, rem_dataset = self.split(dataset, frac_train,
                                            [train_dir, rem_dir])

    # calculate percent split for valid (out of test and valid)
    if frac_valid + frac_test > 0:
      valid_percentage = frac_valid / (frac_valid + frac_test)
    else:
      return train_dataset, None, None
    # split test data into valid and test, treating sub test set also as sparse
    valid_dataset, test_dataset = self.split(dataset, valid_percentage,
                                             [valid_dir, test_dir])

    return train_dataset, valid_dataset, test_dataset

  def k_fold_split(self, dataset, k, directories=None):
    """Needs custom implementation due to ragged splits for stratification."""
    log("Computing K-fold split", self.verbose)
    if directories is None:
      directories = [tempfile.mkdtemp() for _ in range(k)]
    else:
      assert len(directories) == k
    fold_datasets = []
    # rem_dataset is remaining portion of dataset
    rem_dataset = dataset
    for fold in range(k):
      # Note starts as 1/k since fold starts at 0. Ends at 1 since fold goes up
      # to k-1.
      frac_fold = 1. / (k - fold)
      fold_dir = directories[fold]
      rem_dir = tempfile.mkdtemp()
      fold_dataset, rem_dataset = self.split(rem_dataset, frac_fold,
                                             [fold_dir, rem_dir])
      fold_datasets.append(fold_dataset)
    return fold_datasets


class SingletaskStratifiedSplitter(Splitter):
  """ 
  Class for doing data splits by stratification on a single task.

  Example:

  >>> n_samples = 100
  >>> n_features = 10
  >>> n_tasks = 10
  >>> X = np.random.rand(n_samples, n_features)
  >>> y = np.random.rand(n_samples, n_tasks)
  >>> w = np.ones_like(y)
  >>> dataset = DiskDataset.from_numpy(np.ones((100,n_tasks)), np.ones((100,n_tasks)), verbose=False)
  >>> splitter = SingletaskStratifiedSplitter(task_number=5, verbose=False)
  >>> train_dataset, test_dataset = splitter.train_test_split(dataset)

  """

  def __init__(self, task_number=0, verbose=False):
    """
    Creates splitter object.

    Parameters
    ----------
    task_number: int (Optional, Default 0)
      Task number for stratification.
    verbose: bool (Optional, Default False)
      Controls logging frequency.
    """
    self.task_number = task_number
    self.verbose = verbose

  def k_fold_split(self, dataset, k, seed=None, log_every_n=None):
    """
    Splits compounds into k-folds using stratified sampling.
    Overriding base class k_fold_split.

    Parameters
    ----------
    dataset: dc.data.Dataset object
      Dataset.
    k: int
      Number of folds.
    seed: int (Optional, Default None)
      Random seed.
    log_every_n: int (Optional, Default None)
      Log every n examples (not currently used).

    Returns
    -------
    fold_datasets: List
      List containing dc.data.Dataset objects
    """
    log("Computing K-fold split", self.verbose)
    if directories is None:
      directories = [tempfile.mkdtemp() for _ in range(k)]
    else:
      assert len(directories) == k

    y_s = dataset.y[:, self.task_number]
    sortidx = np.argsort(y_s)
    sortidx_list = np.array_split(sortidx, k)

    fold_datasets = []
    for fold in range(k):
      fold_dir = directories[fold]
      fold_ind = sortidx_list[fold]
      fold_dataset = dataset.select(fold_ind, fold_dir)
      fold_datasets.append(fold_dataset)
    return fold_datasets

  def split(self,
            dataset,
            seed=None,
            frac_train=.8,
            frac_valid=.1,
            frac_test=.1,
            log_every_n=None):
    """
    Splits compounds into train/validation/test using stratified sampling.

    Parameters
    ----------
    dataset: dc.data.Dataset object
      Dataset.
    seed: int (Optional, Default None)
      Random seed.
    frac_train: float (Optional, Default .8)
      Fraction of dataset put into training data.
    frac_valid: float (Optional, Default .1)
      Fraction of dataset put into validation data.
    frac_test: float (Optional, Default .1)
      Fraction of dataset put into test data.
    log_every_n: int (Optional, Default None)
      Log every n examples (not currently used).

    Returns
    -------
    retval: Tuple
      Tuple containing train indices, valid indices, and test indices    
    """
    # JSG Assert that split fractions can be written as proper fractions over 10.
    # This can be generalized in the future with some common demoninator determination.
    # This will work for 80/20 train/test or 80/10/10 train/valid/test (most use cases).
    np.testing.assert_equal(frac_train + frac_valid + frac_test, 1.)
    np.testing.assert_equal(10 * frac_train + 10 * frac_valid + 10 * frac_test,
                            10.)

    if not seed is None:
      np.random.seed(seed)

    y_s = dataset.y[:, self.task_number]
    sortidx = np.argsort(y_s)

    split_cd = 10
    train_cutoff = int(frac_train * split_cd)
    valid_cutoff = int(frac_valid * split_cd) + train_cutoff
    test_cutoff = int(frac_test * split_cd) + valid_cutoff

    train_idx = np.array([])
    valid_idx = np.array([])
    test_idx = np.array([])

    while sortidx.shape[0] >= split_cd:
      sortidx_split, sortidx = np.split(sortidx, [split_cd])
      shuffled = np.random.permutation(range(split_cd))
      train_idx = np.hstack([train_idx, sortidx_split[shuffled[:train_cutoff]]])
      valid_idx = np.hstack(
          [valid_idx, sortidx_split[shuffled[train_cutoff:valid_cutoff]]])
      test_idx = np.hstack([test_idx, sortidx_split[shuffled[valid_cutoff:]]])

    # Append remaining examples to train
    if sortidx.shape[0] > 0: np.hstack([train_idx, sortidx])

    return (train_idx, valid_idx, test_idx)


class MolecularWeightSplitter(Splitter):
  """
  Class for doing data splits by molecular weight.
  """

  def split(self,
            dataset,
            seed=None,
            frac_train=.8,
            frac_valid=.1,
            frac_test=.1,
            log_every_n=None):
    """
    Splits internal compounds into train/validation/test using the MW calculated
    by SMILES string.
    """

    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.)
    if not seed is None:
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

  def split(self,
            dataset,
            seed=None,
            frac_train=.8,
            frac_valid=.1,
            frac_test=.1,
            log_every_n=None):
    """
    Splits internal compounds randomly into train/validation/test.
    """
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.)
    if not seed is None:
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

  def split(self,
            dataset,
            seed=None,
            frac_train=.8,
            frac_valid=.1,
            frac_test=.1,
            log_every_n=None):
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


class IndiceSplitter(Splitter):
  """
  Class for splits based on input order. 
  """

  def __init__(self, verbose=False, valid_indices=None, test_indices=None):
    """
    Parameters
    -----------
    valid_indices: list of int
        indices of samples in the valid set
    test_indices: list of int
        indices of samples in the test set
    """
    self.verbose = verbose
    self.valid_indices = valid_indices
    self.test_indices = test_indices

  def split(self,
            dataset,
            seed=None,
            frac_train=.8,
            frac_valid=.1,
            frac_test=.1,
            log_every_n=None):
    """
    Splits internal compounds into train/validation/test in designated order.
    """
    num_datapoints = len(dataset)
    indices = np.arange(num_datapoints).tolist()
    train_indices = []
    if self.valid_indices is None:
      self.valid_indices = []
    if self.test_indices is None:
      self.test_indices = []
    valid_test = self.valid_indices
    valid_test.extend(self.test_indices)
    for indice in indices:
      if not indice in valid_test:
        train_indices.append(indice)

    return (train_indices, self.valid_indices, self.test_indices)


def ClusterFps(fps, cutoff=0.2):
  # (ytz): this is directly copypasta'd from Greg Landrum's clustering example.
  dists = []
  nfps = len(fps)
  for i in range(1, nfps):
    sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
    dists.extend([1 - x for x in sims])
  cs = Butina.ClusterData(dists, nfps, cutoff, isDistData=True)
  return cs


class ButinaSplitter(Splitter):
  """
  Class for doing data splits based on the butina clustering of a bulk tanimoto
  fingerprint matrix.
  """

  def split(self,
            dataset,
            frac_train=None,
            frac_valid=None,
            frac_test=None,
            log_every_n=1000,
            cutoff=0.18):
    """
    Splits internal compounds into train and validation based on the butina
    clustering algorithm. This splitting algorithm has an O(N^2) run time, where N
    is the number of elements in the dataset. The dataset is expected to be a classification
    dataset.

    This algorithm is designed to generate validation data that are novel chemotypes.
    
    Note that this function entirely disregards the ratios for frac_train, frac_valid,
    and frac_test. Furthermore, it does not generate a test set, only a train and valid set.
  
    Setting a small cutoff value will generate smaller, finer clusters of high similarity,
    whereas setting a large cutoff value will generate larger, coarser clusters of low similarity.
    """
    print("Performing butina clustering with cutoff of", cutoff)
    mols = []
    for ind, smiles in enumerate(dataset.ids):
      mols.append(Chem.MolFromSmiles(smiles))
    n_mols = len(mols)
    fps = [AllChem.GetMorganFingerprintAsBitVect(x, 2, 1024) for x in mols]

    scaffold_sets = ClusterFps(fps, cutoff=cutoff)
    scaffold_sets = sorted(scaffold_sets, key=lambda x: -len(x))

    ys = dataset.y
    valid_inds = []
    for c_idx, cluster in enumerate(scaffold_sets):
      # for m_idx in cluster:
      valid_inds.extend(cluster)
      # continue until we find an active in all the tasks, otherwise we can't
      # compute a meaningful AUC
      # TODO (ytz): really, we want at least one active and inactive in both scenarios.
      # TODO (Ytz): for regression tasks we'd stop after only one cluster.
      active_populations = np.sum(ys[valid_inds], axis=0)
      if np.all(active_populations):
        print("# of actives per task in valid:", active_populations)
        print("Total # of validation points:", len(valid_inds))
        break

    train_inds = list(itertools.chain.from_iterable(scaffold_sets[c_idx + 1:]))
    test_inds = []

    return train_inds, valid_inds, []


class ScaffoldSplitter(Splitter):
  """
  Class for doing data splits based on the scaffold of small molecules.
  """

  def split(self,
            dataset,
            frac_train=.8,
            frac_valid=.1,
            frac_test=.1,
            log_every_n=1000):
    """
    Splits internal compounds into train/validation/test by scaffold.
    """
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.)
    scaffolds = {}
    log("About to generate scaffolds", self.verbose)
    data_len = len(dataset)
    for ind, smiles in enumerate(dataset.ids):
      if ind % log_every_n == 0:
        log("Generating scaffold %d/%d" % (ind, data_len), self.verbose)
      scaffold = generate_scaffold(smiles)
      if scaffold not in scaffolds:
        scaffolds[scaffold] = [ind]
      else:
        scaffolds[scaffold].append(ind)
    # Sort from largest to smallest scaffold sets
    scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
    scaffold_sets = [
        scaffold_set
        for (scaffold, scaffold_set) in sorted(
            scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
    ]
    train_cutoff = frac_train * len(dataset)
    valid_cutoff = (frac_train + frac_valid) * len(dataset)
    train_inds, valid_inds, test_inds = [], [], []
    log("About to sort in scaffold sets", self.verbose)
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

  def __init__(self, input_file, split_field, verbose=False):
    """Provide input information for splits."""
    raw_df = next(load_data([input_file], shard_size=None))
    self.splits = raw_df[split_field].values
    self.verbose = verbose

  def split(self,
            dataset,
            frac_train=.8,
            frac_valid=.1,
            frac_test=.1,
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
