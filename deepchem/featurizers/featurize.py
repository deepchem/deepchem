"""
Process an input dataset into a format suitable for machine learning.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import gzip
import pandas as pd
import numpy as np
import csv
import numbers
from rdkit import Chem
from deepchem.featurizers.fingerprints import CircularFingerprint
from deepchem.featurizers.basic import RDKitDescriptors
from deepchem.utils.save import log
from deepchem.utils.save import save_to_disk
from deepchem.utils.save import load_from_disk
from deepchem.utils.save import load_pandas_from_disk
from deepchem.utils import ScaffoldGenerator
from deepchem.featurizers.nnscore import NNScoreComplexFeaturizer
import multiprocessing as mp
from functools import partial


def generate_scaffold(smiles, include_chirality=False):
  """Compute the Bemis-Murcko scaffold for a SMILES string."""
  mol = Chem.MolFromSmiles(smiles)
  engine = ScaffoldGenerator(include_chirality=include_chirality)
  scaffold = engine.get_scaffold(mol)
  return scaffold

def _check_validity(compounds_df):
  """Ensure that columns of compound_df contain required elements."""
  if not set(FeaturizedSamples.colnames).issubset(compounds_df.keys()):
    raise ValueError("Compound dataframe does not contain required columns")

def _process_field(val):
  """Parse data in a field."""
  if (isinstance(val, numbers.Number) or isinstance(val, np.ndarray)):
    return val
  elif isinstance(val, list):
    return [_process_field(elt) for elt in val]
  elif isinstance(val, str):
    try:
      return float(val)
    except ValueError:
      return val
  else:
    raise ValueError("Field of unrecognized type: %s" % str(val))

def _get_input_type(input_file):
  """Get type of input file. Must be csv/pkl.gz/sdf file."""
  filename, file_extension = os.path.splitext(input_file)
  # If gzipped, need to compute extension again
  if file_extension == ".gz":
    filename, file_extension = os.path.splitext(filename)
  if file_extension == ".csv":
    return "csv"
  elif file_extension == ".pkl":
    return "pandas-pickle"
  elif file_extension == ".joblib":
    return "pandas-joblib"
  else:
    raise ValueError("Unrecognized extension %s" % file_extension)

def _get_fields(input_file):
  """Get the names of fields and field_types for input data."""
  # If CSV input, assume that first row contains labels
  input_type = _get_input_type(input_file)
  if input_type == "csv":
    with open(input_file, "rb") as inp_file_obj:
      return csv.reader(inp_file_obj).next()
  elif input_type == "pandas-joblib":
    df = load_from_disk(input_file)
    return df.keys()
  elif input_type == "pandas-pickle":
    df = load_pickle_from_disk(input_file)
    return df.keys()
  else:
    raise ValueError("Unrecognized extension for %s" % input_file)

class DataFeaturizer(object):
  """
  Handles loading/featurizing of chemical samples (datapoints).

  Currently knows how to load csv-files/pandas-dataframes/SDF-files. Writes a
  dataframe object to disk as output.
  """

  def __init__(self, tasks, smiles_field, split_field=None,
               id_field=None, threshold=None, user_specified_features=None,
               protein_pdb_field=None, ligand_pdb_field=None,
               ligand_mol2_field=None, 
               compound_featurizers=[], complex_featurizers=[],
               verbose=False, log_every_n=1000):
    """Extracts data from input as Pandas data frame"""
    if not isinstance(tasks, list):
      raise ValueError("tasks must be a list.")
    self.tasks = tasks
    self.smiles_field = smiles_field
    self.split_field = split_field
    if id_field is None:
      self.id_field = smiles_field
    else:
      self.id_field = id_field
    self.threshold = threshold
    self.protein_pdb_field = protein_pdb_field
    self.ligand_pdb_field = ligand_pdb_field
    self.ligand_mol2_field = ligand_mol2_field
    self.user_specified_features = user_specified_features
    self.compound_featurizers = compound_featurizers
    self.complex_featurizers = complex_featurizers
    self.verbose = verbose
    self.log_every_n = log_every_n

  def featurize(self, input_file, feature_dir, samples_dir, shard_size=128):
    """Featurize provided file and write to specified location."""
    input_type = _get_input_type(input_file)

    log("Loading raw samples now.", self.verbose)
    raw_df = load_pandas_from_disk(input_file)
    fields = raw_df.keys()
    log("Loaded raw data frame from file.", self.verbose)

    def process_raw_sample_helper(row, fields, input_type):
      return self._process_raw_sample(input_type, row, fields)
    process_raw_sample_helper_partial = partial(process_raw_sample_helper,
                                                fields=fields,
                                                input_type=input_type)


    raw_df = raw_df.apply(process_raw_sample_helper_partial, axis=1, reduce=False)
    nb_sample = raw_df.shape[0]
    interval_points = np.linspace(
        0, nb_sample, np.ceil(float(nb_sample)/shard_size)+1, dtype=int)
    shard_files = []
    for j in range(len(interval_points)-1):
      log("Sharding and standardizing into shard-%s / %s shards" % (str(j+1), len(interval_points)-1), self.verbose)
      raw_df_shard = raw_df.iloc[range(interval_points[j], interval_points[j+1])]
      
      df = self._standardize_df(raw_df_shard) 
      log("Aggregating User-Specified Features", self.verbose)
      self._add_user_specified_features(df)

      for compound_featurizer in self.compound_featurizers:
        log("Currently feauturizing feature_type: %s"
            % compound_featurizer.__class__.__name__, self.verbose)
        self._featurize_compounds(df, compound_featurizer)

      for complex_featurizer in self.complex_featurizers:
        log("Currently feauturizing feature_type: %s"
            % complex_featurizer.__class__.__name__, self.verbose)
        self._featurize_complexes(df, complex_featurizer)

      shard_out = os.path.join(feature_dir, "features_shard%d.joblib" % j)
      save_to_disk(df, shard_out)
      shard_files.append(shard_out)

    featurizers = self.compound_featurizers + self.complex_featurizers
    samples = FeaturizedSamples(samples_dir=samples_dir, featurizers=featurizers, 
                                dataset_files=shard_files,
                                reload_data=False)

    return samples

  def _process_raw_sample(self, input_type, row, fields):
    """Extract information from row data."""
    data = {}
    if input_type == "csv":
      for ind, field in enumerate(fields):
        data[field] = _process_field(row[ind])
    elif input_type in ["pandas-pickle", "pandas-joblib"]:
      for field in fields:
        data[field] = _process_field(row[field])
    else:
      raise ValueError("Unrecognized input_type")
    if self.threshold is not None:
      for task in self.tasks:
        raw = _process_field(data[task])
        if not isinstance(raw, float):
          raise ValueError("Cannot threshold non-float fields.")
        data[field] = 1 if raw > self.threshold else 0
    return data

  def _standardize_df(self, ori_df):
    """Copy specified columns to new df with standard column names."""
    df = pd.DataFrame(ori_df[[self.id_field]])
    df.columns = ["mol_id"]
    df["smiles"] = ori_df[[self.smiles_field]]
    for task in self.tasks:
      df[task] = ori_df[[task]]
    if self.split_field is not None:
      df["split"] = ori_df[[self.split_field]]
    if self.protein_pdb_field is not None:
      df["protein_pdb"] = ori_df[[self.protein_pdb_field]]
    if self.ligand_pdb_field is not None:
      df["ligand_pdb"] = ori_df[[self.ligand_pdb_field]]
    if self.ligand_mol2_field is not None:
      df["ligand_mol2"] = ori_df[[self.ligand_mol2_field]]
    return df

  def _featurize_complexes(self, df, featurizer):
    """Generates circular fingerprints for dataset."""
    protein_pdbs = list(df["protein_pdb"])
    ligand_pdbs = list(df["ligand_pdb"])
    complexes = zip(ligand_pdbs, protein_pdbs)

    features = featurizer.featurize_complexes(ligand_pdbs, protein_pdbs)
    df[featurizer.__class__.__name__] = list(features)

  def _featurize_compounds(self, df, featurizer):    
    """Featurize individual compounds.

       Given a featurizer that operates on individual chemical compounds 
       or macromolecules, compute & add features for that compound to the 
       features dataframe
    """
    features = []
    sample_smiles = df["smiles"].tolist()
    for ind, smiles in enumerate(sample_smiles):
      if ind % self.log_every_n == 0:
        log("Featurizing sample %d" % ind, self.verbose)
      mol = Chem.MolFromSmiles(smiles)
      features.append(featurizer.featurize([mol]))
    df[featurizer.__class__.__name__] = features

  def _add_user_specified_features(self, df):
    """Merge user specified features. 

      Merge features included in dataset provided by user
      into final features dataframe
    """
    if self.user_specified_features is not None:
      log("Adding user-defined features.", self.verbose)
      features_data = []
      for _, row in df.iterrows():
        # pandas rows are tuples (row_num, row_data)
        feature_list = []
        for feature_name in self.user_specified_features:
          feature_list.append(row[feature_name])
        features_data.append({"user-specified-features": np.array(feature_list)})
      df["user-specified-features"] = pd.DataFrame(features_data)


def map_function(data_tuple, featurizer):
  featurizer = NNScoreComplexFeaturizer()
  ind, ligand_pdb, protein_pdb = data_tuple
  print("Mapping on ind %d" % ind)
  print("ind, type(ligand_pdb), type(protein_pdb): %s " % str((ind, type(ligand_pdb), type(protein_pdb))))
  return featurizer.featurize_complexes([ligand_pdb], [protein_pdb])

class FeaturizedSamples(object):
  """
  Wrapper class for featurized data on disk.
  """
  # The standard columns for featurized data.
  colnames = ["mol_id", "smiles", "split"]
  optional_colnames = ["protein_pdb", "ligand_pdb", "ligand_mol2"]

  def __init__(self, samples_dir, featurizers, dataset_files=None, 
               overwrite=True, reload_data=False):
    """
    Initialiize FeaturizedSamples

    If samples_dir does not exist, must specify dataset_files. Then samples_dir
    is created and populated. If samples_dir exists (created by previous call to
    FeaturizedSamples), then dataset_files cannot be specified. If overwrite is
    set and dataset_files is provided, will overwrite old dataset_files with
    new.
    """
    self.dataset_files = dataset_files
    self.feature_types = (
        ["user-specified-features"] + 
        [featurizer.__class__.__name__ for featurizer in featurizers])

    self.featurizers = featurizers

    if not os.path.exists(samples_dir):
      os.makedirs(samples_dir)
    self.samples_dir = samples_dir
    if os.path.exists(self._get_compounds_filename()) and reload_data:
      compounds_df = load_from_disk(self._get_compounds_filename())
    else:
      compounds_df = self._get_compounds()
      # compounds_df is not altered by any method after initialization, so it's
      # safe to keep a copy in memory and on disk.
      save_to_disk(compounds_df, self._get_compounds_filename())
    _check_validity(compounds_df)
    self.compounds_df = compounds_df
    self.num_samples = len(compounds_df)

    if os.path.exists(self._get_dataset_paths_filename()):
      if dataset_files is not None:
        if overwrite:
          save_to_disk(dataset_files, self._get_dataset_paths_filename())
        else:
          raise ValueError("Can't change dataset_files already stored on disk")
      self.dataset_files = load_from_disk(self._get_dataset_paths_filename())
    else:
      save_to_disk(dataset_files, self._get_dataset_paths_filename())

  def _get_compounds_filename(self):
    """
    Get standard location for file listing compounds in this dataframe.
    """
    return os.path.join(self.samples_dir, "compounds.joblib")

  def _get_dataset_paths_filename(self):
    """
    Get standard location for file listing dataset_files.
    """
    return os.path.join(self.samples_dir, "datasets.joblib")

  def _get_compounds(self):
    """
    Create dataframe containing metadata about compounds.
    """
    compound_rows = []
    for dataset_file in self.dataset_files:
      df = load_from_disk(dataset_file)
      compound_ids = list(df["mol_id"])
      smiles = list(df["smiles"])
      if "split" in df.keys():
        splits = list(df["split"])
      else:
        splits = [None] * len(smiles)
      compound_rows += [list(elt) for elt in zip(compound_ids, smiles, splits)]

    compounds_df = pd.DataFrame(compound_rows,
                                columns=("mol_id", "smiles", "split"))
    return compounds_df

  def _set_compound_df(self, df):
    """Internal method used to replace compounds_df."""
    _check_validity(df)
    save_to_disk(df, self._get_compounds_filename())
    self.compounds_df = df
    self.num_samples = len(df)

  def __len__(self):
    """Returns size of internal dataset."""
    return self.num_samples

  # TODO(rbharath): Might this be inefficient?
  def itersamples(self):
    """
    Provides an iterator over samples.

    Each sample from the iterator is a dataframe of samples.
    """
    compound_ids = set(list(self.compounds_df["mol_id"]))
    for df_file in self.dataset_files:
      df = load_from_disk(df_file)
      visible_inds = []
      for ind, row in df.iterrows():
        if row["mol_id"] in compound_ids:
          visible_inds.append(ind)
      yield df.loc[visible_inds]

  def train_valid_test_split(self, splittype, train_dir, test_dir,
                             valid_dir=None, frac_train=.8, frac_valid=.1,
                             frac_test=.1, seed=None):
    """
    Splits self into train/validation/test sets.

    Returns FeaturizedDataset objects.
    """
    if splittype == "random":
      train_inds, valid_inds, test_inds = self._random_split(
          seed=seed, frac_train=frac_train, frac_test=frac_test,
          frac_valid=frac_valid)
    elif splittype == "scaffold":
      train_inds, valid_inds, test_inds = self._scaffold_split(
          frac_train=frac_train, frac_test=frac_test,
          frac_valid=frac_valid)
    elif splittype == "specified":
      train_inds, valid_inds, test_inds = self._specified_split()
    else:
      raise ValueError("improper splittype.")
    train_samples = FeaturizedSamples(samples_dir=train_dir, 
                                      dataset_files=self.dataset_files,
                                      featurizers=self.featurizers)
    train_samples._set_compound_df(self.compounds_df.iloc[train_inds])
    test_samples = FeaturizedSamples(samples_dir=test_dir, 
                                     dataset_files=self.dataset_files,
                                     featurizers=self.featurizers)
    test_samples._set_compound_df(self.compounds_df.iloc[test_inds])
    if valid_dir is None:
      valid_samples = None
    else:
      valid_samples = FeaturizedSamples(samples_dir=valid_dir, 
                                       dataset_files=self.dataset_files,
                                       featurizers=self.featurizers)
      valid_samples._set_compound_df(self.compounds_df.iloc[valid_inds])

    return train_samples, valid_samples, test_samples

  def train_test_split(self, splittype, train_dir, test_dir, seed=None,
                       frac_train=.8):
    """
    Splits self into train/test sets.

    Returns FeaturizedDataset objects.
    """
    train_samples, _, test_samples = self.train_valid_test_split(
        splittype, train_dir, test_dir, valid_dir=None,
        frac_train=frac_train, frac_test=1-frac_train, frac_valid=0.)
    return train_samples, test_samples

  def _random_split(self, seed=None, frac_train=.8, frac_valid=.1,
                    frac_test=.1):
    """
    Splits internal compounds randomly into train/validation/test.
    """
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.)
    np.random.seed(seed)
    train_cutoff = frac_train * len(self.compounds_df)
    valid_cutoff = (frac_train+frac_valid) * len(self.compounds_df)
    shuffled = np.random.permutation(range(len(self.compounds_df)))
    return (shuffled[:train_cutoff], shuffled[train_cutoff:valid_cutoff],
            shuffled[valid_cutoff:])

  def _scaffold_split(self, frac_train=.8, frac_valid=.1, frac_test=.1):
    """
    Splits internal compounds into train/validation/test by scaffold.
    """
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.)
    scaffolds = {}
    for ind, row in self.compounds_df.iterrows():
      scaffold = generate_scaffold(row["smiles"])
      if scaffold not in scaffolds:
        scaffolds[scaffold] = [ind]
      else:
        scaffolds[scaffold].append(ind)
    # Sort from largest to smallest scaffold sets
    scaffold_sets = [scaffold_set for (scaffold, scaffold_set) in
                     sorted(scaffolds.items(), key=lambda x: -len(x[1]))]
    train_cutoff = frac_train * len(self.compounds_df)
    valid_cutoff = (frac_train+frac_valid) * len(self.compounds_df)
    train_inds, valid_inds, test_inds = [], [], []
    for scaffold_set in scaffold_sets:
      if len(train_inds) + len(scaffold_set) > train_cutoff:
        if len(train_inds) + len(valid_inds) + len(scaffold_set) > valid_cutoff:
          test_inds += scaffold_set
        else:
          valid_inds += scaffold_set
      else:
        train_inds += scaffold_set
    return train_inds, valid_inds, test_inds

  def _specified_split(self):
    """
    Splits internal compounds into train/validation/test by user-specification.
    """
    train_inds, valid_inds, test_inds = [], [], []
    for ind, row in self.compounds_df.iterrows():
      if row["split"].lower() == "train":
        train_inds.append(ind)
      elif row["split"].lower() == "validation":
        valid_inds.append(ind)
      elif row["split"].lower() == "test":
        test_inds.append(ind)
      else:
        raise ValueError("Missing required split information.")
    return train_inds, valid_inds, test_inds
