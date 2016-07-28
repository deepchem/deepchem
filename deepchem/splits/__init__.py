"""
Contains an abstract base class that supports chemically aware data splits.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "GPL"

import os
import numpy as np
import pandas as pd
from rdkit import Chem
from deepchem.utils import ScaffoldGenerator
from deepchem.utils.save import log
from deepchem.datasets import Dataset
from deepchem.featurizers.featurize import load_data


def generate_scaffold(smiles, include_chirality=False):
  """Compute the Bemis-Murcko scaffold for a SMILES string."""
  mol = Chem.MolFromSmiles(smiles)
  engine = ScaffoldGenerator(include_chirality=include_chirality)
  scaffold = engine.get_scaffold(mol)
  return scaffold


class Splitter(object):
  """
  Abstract base class for chemically aware splits..
  """

  def __init__(self, verbosity=None):
    """Creates splitter object."""
    self.verbosity = verbosity

  def train_valid_test_split(self, dataset, train_dir,
                             valid_dir, test_dir, frac_train=.8,
                             frac_valid=.1, frac_test=.1, seed=None,
                             log_every_n=1000):
    """
    Splits self into train/validation/test sets.

    Returns Dataset objects.
    """
    log("Computing train/valid/test indices", self.verbosity)
    train_inds, valid_inds, test_inds = self.split(
        dataset,
        frac_train=frac_train, frac_test=frac_test,
        frac_valid=frac_valid, log_every_n=log_every_n)
    train_dataset = dataset.select(train_dir, train_inds)
    if valid_dir is not None:
      valid_dataset = dataset.select(valid_dir, valid_inds)
    else:
      valid_dataset = None
    test_dataset = dataset.select(test_dir, test_inds)

    return train_dataset, valid_dataset, test_dataset

  def train_test_split(self, samples, train_dir, test_dir, seed=None,
                       frac_train=.8):
    """
    Splits self into train/test sets.

    Returns Dataset objects.
    """
    valid_dir = None
    train_samples, _, test_samples = self.train_valid_test_split(
        samples, train_dir, valid_dir, test_dir,
        frac_train=frac_train, frac_test=1 - frac_train, frac_valid=0.)
    return train_samples, test_samples

  def split(self, samples, frac_train=None, frac_valid=None, frac_test=None,
            log_every_n=None):
    """
    Stub to be filled in by child classes.
    """
    raise NotImplementedError


class StratifiedSplitter(Splitter):
  """
  Class for doing stratified splits -- where data is too sparse to do regular splits
  """

  def __randomizeArrays(self, arraylist):
    generator_state = numpy.random.get_state()
    for array in arrayList:
      numpy.random.shuffle(array)
      numpy.random.set_state(generator_state)
    return arrayList

  def __generate_required_hits(self, y_df, frac_train):
    colIndex = 0
    required_hit_dict = {}
    totalCount = len(y_df.index)
    for col in y_df:
      NaN_count = y_df[col].isnull().sum()
      notNaN = totalCount - NaN_count
      requiredNotNaN = frac_train * notNaN
      required_hit_dict[colIndex] = requiredNotNaN
      colIndex += 1
    return required_hit_dict

  def __generate_required_index(self, y_df, required_hit_dict):
    index_dict = {}
    colIndex = 0
    for col in y_df:
      column = y_df[col]
      num_hit = 0
      num_required = required_hit_dict[colIndex]
      colIndex += 1
      for index, value in y_df[col].iteritems():
        if pd.notnull(value):
          num_hit += 1
          # check to see if number of hits has been hit
          if num_hit >= num_required:
            index_dict[colIndex] = index
            break
    return index_dict

  def train_valid_test_split(self, dataset, train_dir,
                             valid_dir, test_dir, frac_train=.8,
                             frac_valid=.1, frac_test=.1, seed=None,
                             log_every_n=1000):
   # Obtain original x, y, and w arrays
    numpyArrayList = dataset.to_numpy();

    numpyArrayList = randomizeArrays(numpyArrayList)
    X = numpyArrayList[0]
    y = numpyArrayList[1]
    w = numpyArrayList[2]
    ids = numpyArrayList[3]

    """
    frac_train identifies percentage of datapoints that need to be present in split -- so 80% training data may actually be 90% of data (but 80% of actual datapoints, not NaN, will be present in split)
    """
    # find, for each task, the total number of hits and calculate the required
    # number of hits for valid split based on frac_train
    x_df = pd.DataFrame(data=x)
    y_df = pd.DataFrame(data=y)
    w_df = pd.DataFrame(data=w)
    id_df = pd.DataFrame(data=ids)

    required_hit_dict = __generate_required_hits(y_df, frac_train)
    index_dict = __generate_required_index(y_df, required_hit_dict)
    X_train, X_test, y_train, y_test, w_train, w_test, id_train, id_test = []

    # cycle through rows in y, copy over rows as appropriate
    for rowIndex, row in y_df.iterrows():
     weight_row = w_df.iloc[rowIndex].tolist() #get corresponding weight row as list
     weight_train_row = []
     weight_test_row = []
     for index, value in row.iteritems():
       # test if should be test or train data
       if rowIndex <= index_dict[index]: #train data
         weight_train_row.append(weight_row[index]) #add corresponding weight
         weight_test_row.append(0)
       else: #index is past test index -- this datapoint is test data
         weight_train_row.append(0)
         weight_test_row.append(weight_row[index])
     x_row = x_df.iloc[rowIndex].tolist()
     id_row = id_df.iloc[rowIndex].tolist()
     # check to see if any weight vectors are just zero
     if weight_train_row.count(0) == len(weight_train_row): #entire example is a test example
       # Add entire row to appropriate test arrays
       X_test.append(x_row) #get corresponding row from original x df
       y_test.append(row)
       w_test.append(weight_test_row)
       id_test.append(id_row)
     elif weight_test_row.count(0) == len(weight_test_row): #entirely train example
       X_train.append(x_row)
       y_train.append(row)
       w_train.append(weight_train_row)
       id_train.append(id_row)
     else: #hybrid example -- feature X, results y, and smiles id are appended to both test and train. Weight vectors for train and row are appended as appropriately to dictate whether value is train or test
       X_train.append(x_row)
       X_test.append(x_row)
       y_train.append(row)
       y_test.append(row)
       w_train.append(weight_train_row)
       w_test.append(weight_test_row)
       id_train.append(id_row)
       id_test.append(id_row)


    X_train_np = np.array(X_train)
    X_test_np = np.array(X_test)
    y_train_np = np.array(y_train)
    y_test_np = np.array(y_test)
    w_train_np = np.array(w_train)
    w_test_np = np.array(w_test)
    id_train_np = np.array(id_train)
    id_test_np = np.array(id_test)

    # make valid split - 50/50 split of test
    X_split_list = np.vsplit(X_test_np, 2)
    y_split_list = np.vsplit(y_test_np, 2)
    w_split_list = np.vsplit(w_test_np, 2)
    id_split_list = np.vsplit(id_test_np, 2)

    X_test_np = X_split_list[0]
    X_valid_np = X_split_list[1]
    y_test_np = y_split_list[0]
    y_valid_np = y_split_list[1]
    w_test_np = w_split_list[0]
    w_valid_np = w_split_list[1]
    id_test_np = id_split_list[0]
    id_valid_np = id_split_list[1]

    # turn back into dataset objects
    train_data = Dataset.from_numpy(train_dir, X_train_np, y_train_np, w_train_np, id_train_np)
    valid_data = Dataset.from_numpy(valid_dir, X_valid_np, y_valid_np, w_valid_np, id_valid_np)
    test_data = Dataset.from_numpy(test_dir, X_test_np, y_test_np, w_test_np, id_test_np)
    return (train_data, valid_data, test_data)

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
    for smiles in dataset.get_ids():
      mol = Chem.MolFromSmiles(smiles)
      mw = Chem.rdMolDescriptors.CalcExactMolWt(mol)
      mws.append(mw)

    # Sort by increasing MW
    mws = np.array(mws)
    sortidx = np.argsort(mws)

    train_cutoff = frac_train * len(sortidx)
    valid_cutoff = (frac_train+frac_valid) * len(sortidx)

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
    valid_cutoff = int((frac_train+frac_valid) * num_datapoints )
    shuffled = np.random.permutation(range(num_datapoints))
    return (shuffled[:train_cutoff], shuffled[train_cutoff:valid_cutoff],
            shuffled[valid_cutoff:])

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
    for ind, smiles in enumerate(dataset.get_ids()):
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
    valid_cutoff = (frac_train+frac_valid) * len(dataset)
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
    raw_df = load_data([input_file], shard_size=None).next()
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
