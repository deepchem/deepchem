"""
Contains an abstract base class that supports chemically aware data splits.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "LGPL"

import os
import numpy as np
from rdkit import Chem
from deepchem.utils import ScaffoldGenerator
from deepchem.utils.save import log
from deepchem.featurizers.featurize import FeaturizedSamples

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

  def train_valid_test_split(self, samples, train_dir=None,
                             valid_dir=None, test_dir=None, frac_train=.8,
                             frac_valid=.1, frac_test=.1, seed=None,
                             log_every_n=1000, reload=False):
    """
    Splits self into train/validation/test sets.

    Returns FeaturizedDataset objects.
    """
    if not reload:
      train_inds, valid_inds, test_inds = self.split(
          samples,
          frac_train=frac_train, frac_test=frac_test,
          frac_valid=frac_valid, log_every_n=log_every_n)
    train_samples, valid_samples, test_samples = None, None, None
    dataset_files = samples.dataset_files
    if train_dir is not None:
      train_samples = FeaturizedSamples(samples_dir=train_dir, 
                                        dataset_files=dataset_files,
                                        featurizers=samples.featurizers,
                                        verbosity=self.verbosity,
                                        reload=False)
      if not reload:
        train_samples._set_compound_df(samples.compounds_df.iloc[train_inds])
    if test_dir is not None:
      test_samples = FeaturizedSamples(samples_dir=test_dir, 
                                       dataset_files=dataset_files,
                                       featurizers=samples.featurizers,
                                       verbosity=self.verbosity,
                                       reload=False)
      if not reload:
        test_samples._set_compound_df(samples.compounds_df.iloc[test_inds])
    if valid_dir is not None:
      valid_samples = FeaturizedSamples(samples_dir=valid_dir, 
                                       dataset_files=dataset_files,
                                       featurizers=samples.featurizers,
                                       verbosity=self.verbosity,
                                       reload=False)
      if not reload:
        valid_samples._set_compound_df(samples.compounds_df.iloc[valid_inds])

    return train_samples, valid_samples, test_samples

  def train_test_split(self, samples, train_dir, test_dir, seed=None,
                       frac_train=.8, reload=False):
    """
    Splits self into train/test sets.

    Returns FeaturizedDataset objects.
    """
    train_samples, _, test_samples = self.train_valid_test_split(
        samples, train_dir, valid_dir=None, test_dir=test_dir,
        frac_train=frac_train, frac_test=1-frac_train, frac_valid=0.,
        reload=False)
    return train_samples, test_samples

  def split(self, samples, frac_train=None, frac_valid=None, frac_test=None,
            log_every_n=None):
    """
    Stub to be filled in by child classes.
    """
    raise NotImplementedError

class RandomSplitter(Splitter):
  """
  Class for doing random data splits.
  """
  def split(self, samples, seed=None, frac_train=.8, frac_valid=.1,
            frac_test=.1, log_every_n=None):
    """
    Splits internal compounds randomly into train/validation/test.
    """
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.)
    np.random.seed(seed)
    train_cutoff = frac_train * len(samples.compounds_df)
    valid_cutoff = (frac_train+frac_valid) * len(samples.compounds_df)
    shuffled = np.random.permutation(range(len(samples.compounds_df)))
    return (shuffled[:train_cutoff], shuffled[train_cutoff:valid_cutoff],
            shuffled[valid_cutoff:])

class ScaffoldSplitter(Splitter):
  """
  Class for doing data splits based on the scaffold of small molecules.
  """
  def split(self, samples, frac_train=.8, frac_valid=.1, frac_test=.1,
            log_every_n=1000):
    """
    Splits internal compounds into train/validation/test by scaffold.
    """
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.)
    scaffolds = {}
    log("About to generate scaffolds", self.verbosity)
    for ind, row in samples.compounds_df.iterrows():
      if self.verbosity is not None and ind % log_every_n == 0:
        log("Generating scaffold %d/%d" % (ind, len(samples.compounds_df)),
            self.verbosity)
      scaffold = generate_scaffold(row["smiles"])
      if scaffold not in scaffolds:
        scaffolds[scaffold] = [ind]
      else:
        scaffolds[scaffold].append(ind)
    # Sort from largest to smallest scaffold sets
    scaffold_sets = [scaffold_set for (scaffold, scaffold_set) in
                     sorted(scaffolds.items(), key=lambda x: -len(x[1]))]
    train_cutoff = frac_train * len(samples.compounds_df)
    valid_cutoff = (frac_train+frac_valid) * len(samples.compounds_df)
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
  def split(self, samples, frac_train=.8, frac_valid=.1, frac_test=.1,
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
