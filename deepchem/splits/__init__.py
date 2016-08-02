"""
Contains an abstract base class that supports chemically aware data splits.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Bharath Ramsundar, Aneesh Pappu "
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

    def __randomize_arrays(self, array_list):
      #assumes that every array is of the same dimension
      numRows = array_list[0].shape[0]
      perm = np.random.permutation(numRows)
      for array in array_list:
          array = array[perm]
      return array_list

    def __generate_required_hits(self, w, frac_split):
      required_hits = (w != 0).sum(0)#returns list of per column sum of non zero elements
      for colHits in required_hits:
        colHits = int(frac_split * colHits)
      return required_hits

    def __generate_required_index(self, w, required_hit_list):
      colIndex = 0
      index_hits = []
      #loop through each column and obtain index required to splice out for required fraction of hits
      for col in w.T:
          num_hit = 0
          num_required = required_hit_list[colIndex]
          for index, value in enumerate(col):
              if value != 0:
                num_hit +=1
                if num_hit >= num_required:
                  index_hits.append(index)
                  break
          colIndex += 1
      return index_hits

    def __split(self, X, y, w, ids, frac_split):
      """
      Method that does bulk of splitting dataset appropriately based on desired split percentage
      """
      #find he total number of hits for each task and calculate the required
      #number of hits for split based on frac_split
      required_hits_list = self.__generate_required_hits(w, frac_split)
      #finds index cutoff per task in array to get required split calculated
      index_list = self.__generate_required_index(w, required_hits_list)

      X_first_split = X_second_split = X
      y_first_split = y_second_split = y
      w_first_split = w_second_split = np.zeros(w.shape)
      ids_first_split = ids_second_split = ids

      # chunk appropriate values into weights matrices
      for colIndex, index in enumerate(index_list):
        # copy over up to required idnex for weight first_split
        w_first_split[:index, colIndex] = w[:index, colIndex]
        w_first_split[index:, colIndex] = np.zeros(w_first_split[index:, colIndex].shape)
        w_second_split[:index, colIndex] = np.zeros(w_second_split[:index, colIndex].shape)
        w_second_split[index:, colIndex] = w[index:, colIndex]

      # check out if any rows in either w_first_split or w_second_split are just zeros
      rowsToKeepFirstSplit = w_first_split.any(axis=1)
      rowsToKeepSecondSplit = w_second_split.any(axis=1)

      # prune first set
      w_first_split = w_first_split[rowsToKeepFirstSplit]
      X_first_split = X_first_split[rowsToKeepFirstSplit]
      y_first_split = y_first_split[rowsToKeepFirstSplit]
      ids_first_split = ids_first_split[rowsToKeepFirstSplit]

      # prune second sets
      w_second_split = w_second_split[rowsToKeepSecondSplit]
      X_second_split = X_second_split[rowsToKeepSecondSplit]
      y_second_split = y_second_split[rowsToKeepSecondSplit]
      ids_second_split = ids_second_split[rowsToKeepSecondSplit]
      return X_first_split, y_first_split, w_first_split, ids_first_split, X_second_split, \
             y_second_split, w_second_split, ids_second_split

    def train_valid_test_split(self, dataset, train_dir,
                               valid_dir, test_dir, frac_train=.8,
                               frac_valid=.1, frac_test=.1, seed=None,
                               log_every_n=1000):

      # Obtain original x, y, and w arrays and shuffle
      X, y, w, ids = self.__randomize_arrays(dataset.to_numpy())
      X_train, y_train,w_train, ids_train, X_test, y_test, w_test, ids_test = self.__split(X, y, w, ids, frac_train)

      #calculate percent split for valid (out of test and valid)
      valid_percentage = frac_valid/(frac_valid + frac_test)
      #split test data into valid and test, treating sub test set also as sparse
      X_valid, y_valid, w_valid, ids_valid, X_test, y_test, w_test, ids_test = self.__split(X_test, y_test, w_test, ids_test, valid_percentage)

      # turn back into dataset objects
      train_data = Dataset.from_numpy(train_dir, X_train, y_train, w_train, ids_train)
      valid_data = Dataset.from_numpy(valid_dir, X_valid, y_valid, w_valid, ids_valid)
      test_data = Dataset.from_numpy(test_dir, X_test, y_test, w_test, ids_test)
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
