"""
Process an input dataset into a format suitable for machine learning.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import multiprocessing as mp
import os
import cPickle as pickle
import gzip
import pandas as pd
import numpy as np
import csv
from rdkit import Chem
from vs_utils.features.fingerprints import CircularFingerprint
from vs_utils.features.basic import SimpleDescriptors
from deepchem.utils.save import save_to_disk
from deepchem.utils.save import load_from_disk

def _process_field(val):
  """Parse data in a field."""
  if isinstance(val, float) or isinstance(val, np.ndarray):
    return val
  elif isinstance(val, list):
    return [process_field(elt) for elt in val]
  elif isinstance(val, str):
    try:
      return float(val)
    except ValueError:
      return val
  else:
    raise ValueError("Field of unrecognized type: %s" % str(val))

class DataFeaturizer(object):
  """
  Handles loading/featurizing of chemical samples (datapoints).

  Currently knows how to load csv-files/pandas-dataframes/SDF-files. Writes a
  dataframe object to disk as output.
  """
    
  def __init__(self, tasks, smiles_field, split_field=None,
               id_field=None, threshold=None, user_specified_features=None,
               verbose=False, log_every_n=1000):
    """Extracts data from input as Pandas data frame"""
    self.tasks = tasks
    self.smiles_field = smiles_field
    self.split_field = split_field
    if id_field is None:
      self.id_field = smiles_field
    else:
      self.id_field = id_field
    self.threshold = threshold
    self.user_specified_features = user_specified_features
    self.verbose = verbose
    self.log_every_n = log_every_n

  def featurize(self, input_file, feature_types, out):
    """Featurize provided file and write to specified location."""
    fields = self._get_fields(input_file)
    input_type = self._get_input_type(input_file)

    rows = []
    for ind, row in enumerate(self._get_raw_samples(input_file)):
      if ind % self.log_every_n == 0:
        print("Loading sample %d" % ind)
      rows.append(self._process_raw_sample(input_type, row, fields))
    df = self._standardize_df(pd.DataFrame(rows))
    for feature_type in feature_types:
      self._featurize_df(df, feature_type)
    print("featurize()")
    print("len(df)")
    print(len(df))
    print("out")
    print(out)
    save_to_disk(df, out)
    df_loaded = load_from_disk(out)

  def _get_fields(self, input_file):
    """Get the names of fields and field_types for input data."""
    # If CSV input, assume that first row contains labels
    input_type = self._get_input_type(input_file)
    if input_type == "csv":
      with open(input_file, "rb") as inp_file_obj:
        return csv.reader(inp_file_obj).next()
    elif input_type == "pandas":
      df = load_from_disk(input_file)
      return df.keys()
    elif input_type == "sdf":
      sample_mol = self._get_raw_samples(input_file).next()
      return list(sample_mol.GetPropNames())
    else:
      raise ValueError("Unrecognized extension for %s" % input_file)

  def _get_input_type(self, input_file):
    """Get type of input file. Must be csv/pkl.gz/sdf file."""
    filename, file_extension = os.path.splitext(input_file)
    # If gzipped, need to compute extension again
    if file_extension == ".gz":
      filename, file_extension = os.path.splitext(filename)
    if file_extension == ".csv":
      return "csv"
    elif file_extension == ".pkl":
      return "pandas"
    elif file_extension == ".sdf":
      return "sdf"
    else:
      raise ValueError("Unrecognized extension %s" % file_extension)

  def _get_raw_samples(self, input_file):
    """Returns an iterator over all rows in input_file"""
    input_type = self._get_input_type(input_file)
    if input_type == "csv":
      with open(input_file, "rb") as inp_file_obj:
        for ind, row in enumerate(csv.reader(inp_file_obj)):
          # Skip labels
          if ind == 0:
            continue
          if row is not None:
            yield row
    elif input_type == "pandas":
      dataframe = load_from_disk(input_file)
      for row in dataframe.iterrows():
        yield row
    elif input_type == "sdf":
      if ".gz" in input_file:
        with gzip.open(input_file) as inp_file_obj:
          supp = Chem.ForwardSDMolSupplier(inp_file_obj)
          for mol in supp:
            if mol is not None:
              yield mol
      else:
        with open(input_file) as inp_file_obj:
          supp = Chem.ForwardSDMolSupplier(inp_file_obj)
          mols = [mol for mol in supp if mol is not None]
          for mol in supp:
            if mol is not None:
              yield mol

  def _process_raw_sample(self, input_type, row, fields):
    """Extract information from row data."""
    data = {}
    if input_type == "csv":
      for ind, field in enumerate(fields):
        data[field] = _process_field(row[ind])
      return data
    elif input_type == "pandas":
      # pandas rows are tuples (row_num, data)
      row = row[1]
      for field in fields:
        data[field] = _process_field(row[field])
    elif input_type == "sdf":
      mol = row
      for field in fields:
        if not mol.HasProp(field):
          data[field] = None
        else:
          data[field] = _process_field(mol.GetProp(field))
      data["smiles"] = Chem.MolToSmiles(mol)
    else:
      raise ValueError("Unrecognized input_type")
    if self.threshold is not None:
      for task in self.tasks:
        raw = _process_field(data[task])
        if not isinstance(raw, float):
          raise ValueError("Cannot threshold non-float fields.")
        data[field] = 1 if raw > threshold else 0
    return data

  def _standardize_df(self, ori_df):
    df = pd.DataFrame([])
    df["mol_id"] = ori_df[[self.id_field]]
    df["smiles"] = ori_df[[self.smiles_field]]
    for task in self.tasks:
      df[task] = ori_df[[task]]
    if self.split_field is not None:
      df["split"] = ori_df[[self.split_field]]

    return df

  def _featurize_df(self, df, feature_type):
    """Generates circular fingerprints for dataset."""
    if feature_type == "user-specified-features":
      if self.user_specified_features is not None:
        if self.verbose:
          print("Adding user-defined features.")
        features_data = []
        for row in df.iterrows():
          # pandas rows are tuples (row_num, row_data)
          row, feature_list = row[1], []
          for feature in user_specified_features:
            feature_list.append(row[feature])
          features_data.append({"row": np.array(feature_list)})
        df[feature_type] = pd.DataFrame(features_data)
        return
    elif feature_type in ["ECFP", "RDKIT-descriptors"]:
      if feature_type == "ECFP":
        if self.verbose:
          print("Generating ECFP circular fingerprints.")
        featurizer = CircularFingerprint(size=1024)
      elif feature_type == "RDKIT-descriptors":
        if self.verbose:
          print("Generating RDKIT descriptors.")
        featurizer = SimpleDescriptors()
      features = []
      sample_smiles = df["smiles"].tolist()
      for ind, smiles in enumerate(sample_smiles):
        if ind % self.log_every_n == 0:
          print("Featurizing sample %d" % ind)
        mol = Chem.MolFromSmiles(smiles)
        features.append(featurizer.featurize([mol]))
      df[feature_type] = features
    else:
      raise ValueError("Unsupported feature_type requested.")

class FeaturizedSamples(object):
  """
  Wrapper class for featurized data on disk.
  """
  # The standard columns for featurized data.
  colnames = ["mol_id", "smiles", "split"]
  feature_types = ["user-specified-features", "RDKIT-descriptors", "ECFP"]

  @staticmethod
  def get_sorted_task_names(df):
    """
    Given metadata df, return sorted names of tasks.
    """
    column_names = df.keys()
    task_names = (set(column_names) - 
                  set(FeaturizedSamples.colnames) -
                  set(FeaturizedSamples.feature_types))
    return sorted(list(task_names))

  def __init__(self, feature_dir, dataset_files=None, overwrite=True, reload=False):
    """
    Initialiize FeaturizedSamples

    If feature_dir does not exist, must specify dataset_files. Then feature_dir
    is created and populated. If feature_dir exists (created by previous call to
    FeaturizedSamples), then dataset_files cannot be specified. If overwrite is
    set and dataset_files is provided, will overwrite old dataset_files with
    new.
    """
    self.dataset_files = dataset_files

    if not os.path.exists(feature_dir):
      os.makedirs(feature_dir)
    self.feature_dir = feature_dir
    print("FeaturizedSamples()")
    if os.path.exists(self._get_compounds_filename()) and reload:
      print("compounds loaded from disk")
      compounds_df = load_from_disk(self._get_compounds_filename())
    else:
      print("compounds recomputed")
      compounds_df = self._get_compounds()
      # compounds_df is not altered by any method after initialization, so it's
      # safe to keep a copy in memory and on disk.
      save_to_disk(compounds_df, self._get_compounds_filename())
    print("len(compounds_df)")
    print(len(compounds_df))
    self._check_validity(compounds_df)
    self.compounds_df = compounds_df
    
    if os.path.exists(self._get_dataset_paths_filename()):
      if dataset_files is not None:
        if overwrite:
          save_to_disk(dataset_files, self._get_dataset_paths_filename())
        else:
          raise ValueError("Can't change dataset_files already stored on disk")
      self.dataset_files = load_from_disk(self._get_dataset_paths_filename())
    else:
      save_to_disk(dataset_files, self._get_dataset_paths_filename())

  def _check_validity(self, compounds_df):
    if not set(FeaturizedSamples.colnames).issubset(compounds_df.keys()):
      raise ValueError("Compound dataframe does not contain required columns")

  def _get_compounds_filename(self):
    """
    Get standard location for file listing compounds in this dataframe.
    """
    return os.path.join(self.feature_dir, "compounds.joblib")

  def _get_dataset_paths_filename(self):
    """
    Get standard location for file listing dataset_files.
    """
    return os.path.join(self.feature_dir, "datasets.joblib")

  def _get_compounds(self):
    """
    Create dataframe containing metadata about compounds.
    """
    compound_rows = []
    for dataset_file in self.dataset_files:
      df = load_from_disk(dataset_file)
      compound_ids = list(df["mol_id"])
      smiles = list(df["smiles"])
      splits = list(df["split"])
      compound_rows += [list(elt) for elt in zip(compound_ids, smiles, splits)]
    compounds_df = pd.DataFrame(compound_rows,
                                columns=("mol_id", "smiles", "split"))
    return compounds_df

  def _set_compound_df(self, df):
    """Internal method used to replace compounds_df."""
    self._check_validity(df)
    save_to_disk(df, self._get_compounds_filename())
    self.compounsd_df = df

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
      yield df.iloc[visible_inds]

  def train_test_split(self, splittype, train_dir, test_dir, seed=None,
                       frac_train=.8):
    """
    Splits self into train/test sets and returns two FeaturizedDatsets
    """
    if splittype == "random":
      train_inds, test_inds = self._train_test_random_split(seed=seed, frac_train=frac_train)
    elif splittype == "scaffold":
      train_inds, test_inds = self.train_test_scaffold_split(frac_train=frac_train)
    elif splittype == "specified":
      train_inds, test_inds = self.train_test_specified_split()
    else:
      raise ValueError("improper splittype.")
    train_dataset = FeaturizedSamples(train_dir, self.dataset_files)
    train_dataset._set_compound_df(self.compounds_df.iloc[train_inds])
    test_dataset = FeaturizedSamples(test_dir, self.dataset_files)
    test_dataset._set_compound_df(self.compounds_df.iloc[test_inds])

    return train_dataset, test_dataset

  def _train_test_random_split(self, seed=None, frac_train=.8):
    """
    Splits internal compounds randomly into train/test.
    """
    np.random.seed(seed)
    train_cutoff = frac_train * len(self.compounds_df)
    shuffled = np.random.permutation(range(len(self.compounds_df)))
    return shuffled[:train_cutoff], shuffled[train_cutoff:]

  def _train_test_scaffold_split(self, frac_train=.8):
    """
    Splits internal compounds into train/test by scaffold.
    """
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
    train_inds, test_inds = [], []
    for scaffold_set in scaffold_sets:
      if len(train_inds) + len(scaffold_set) > train_cutoff:
        test_inds += scaffold_set
      else:
        train_inds += scaffold_set
    return train_inds, test_inds

  def _train_test_specified_split(self):
    """
    Splits internal compounds into train/test by user-specification.
    """
    train_inds, test_inds = [], []
    for ind, row in self.compounds_df.iterrows():
      if row["split"].lower() == "train":
        train_inds.append(ind)
      elif row["split"].lower() == "test":
        test_inds.append(ind)
      else:
        raise ValueError("Missing required split information.")
    return train_inds, test_inds
