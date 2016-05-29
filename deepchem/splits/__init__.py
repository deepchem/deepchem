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
from rdkit import Chem
from deepchem.utils import ScaffoldGenerator
from deepchem.utils.save import log
from deepchem.datasets import Dataset

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

  def _check_populated(self, sample_dirs):
    """Check that the provided sample directories are valid."""
    for given_dir in sample_dirs:
      if given_dir is None:
        continue
        
      compounds_filename = os.path.join(given_dir, "datasets.joblib")
      if not os.path.exists(compounds_filename):
        return False
    return True


  def train_valid_test_split(self, samples, train_dir,
                             valid_dir, test_dir, frac_train=.8,
                             frac_valid=.1, frac_test=.1, seed=None,
                             log_every_n=1000, reload=False):
    """
    Splits self into train/validation/test sets.

    Returns Dataset objects.
    """
    compute_split = (
        not reload
        or not self._check_populated([train_dir, test_dir, valid_dir]))
    if compute_split:
      log("Computing train/valid/test indices", self.verbosity)
      train_inds, valid_inds, test_inds = self.split(
          samples,
          frac_train=frac_train, frac_test=frac_test,
          frac_valid=frac_valid, log_every_n=log_every_n)
    train_samples, valid_samples, test_samples = None, None, None
    dataset_files = samples.dataset_files

    # Generate train dir
    train_samples = Dataset(samples_dir=train_dir, 
                            dataset_files=dataset_files,
                            featurizers=samples.featurizers,
                            verbosity=self.verbosity,
                            reload=reload)
    if compute_split:
      train_samples._set_compound_df(samples.compounds_df.iloc[train_inds])
    # Generate test dir
    test_samples = Dataset(samples_dir=test_dir, 
                           dataset_files=dataset_files,
                           featurizers=samples.featurizers,
                           verbosity=self.verbosity,
                           reload=reload)
    if compute_split:
      test_samples._set_compound_df(samples.compounds_df.iloc[test_inds])
    # if requested, generated valid_dir
    if valid_dir is not None:
      valid_samples = Dataset(samples_dir=valid_dir, 
                              dataset_files=dataset_files,
                              featurizers=samples.featurizers,
                              verbosity=self.verbosity,
                              reload=reload)
      if compute_split:
        valid_samples._set_compound_df(samples.compounds_df.iloc[valid_inds])

    return train_samples, valid_samples, test_samples

  def train_test_split(self, samples, train_dir, test_dir, seed=None,
                       frac_train=.8, reload=False):
    """
    Splits self into train/test sets.

    Returns Dataset objects.
    """
    valid_dir = None
    train_samples, _, test_samples = self.train_valid_test_split(
        samples, train_dir, valid_dir, test_dir,
        frac_train=frac_train, frac_test=1-frac_train, frac_valid=0.,
        reload=False)
    return train_samples, test_samples

  def split(self, samples, frac_train=None, frac_valid=None, frac_test=None,
            log_every_n=None):
    """
    Stub to be filled in by child classes.
    """
    raise NotImplementedError

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
    train_cutoff = frac_train * len(dataset)
    valid_cutoff = (frac_train+frac_valid) * len(dataset)
    shuffled = np.random.permutation(range(len(dataset)))
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
    for smiles in dataset.get_ids():
      if self.verbosity is not None and ind % log_every_n == 0:
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
  def split(self, dataset, frac_train=.8, frac_valid=.1, frac_test=.1,
            log_every_n=1000):
    """
    Splits internal compounds into train/validation/test by user-specification.
    """
    train_inds, valid_inds, test_inds = [], [], []
    for ind, row in samples.compounds_df.iterrows():
      if row["split"].lower() == "train":
        train_inds.append(ind)
      elif row["split"].lower() in ["valid", "validation"]:
        valid_inds.append(ind)
      elif row["split"].lower() == "test":
        test_inds.append(ind)
      else:
        raise ValueError("Missing required split information.")
    return train_inds, valid_inds, test_inds
