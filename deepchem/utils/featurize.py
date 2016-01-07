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
from functools import partial
from rdkit import Chem
from vs_utils.features.fingerprints import CircularFingerprint
from vs_utils.features.basic import SimpleDescriptors
from deepchem.utils.dataset import save_to_disk
from deepchem.utils.dataset import load_from_disk

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

class Samples(object):
  """
  Handles loading/featurizing of chemical samples (datapoints).

  Currently knows how to load csv-files/pandas-dataframes/SDF-files.
  """
    
  def __init__(self, input_file, tasks, smiles_field, threshold,
               log_every_n=1000):
    """Extracts data from input as Pandas data frame"""
    rows = []
    self.tasks = tasks
    self.threshold = threshold
    self.input_file = input_file
    self.input_type = self._get_input_type(input_file)
    self.fields = self._get_fields(input_file)

    for ind, row in enumerate(self._get_raw_samples()):
      if ind % log_every_n == 0:
        print("Loading sample %d" % row_index)
      row.append(self._process_raw_sample(row))
    self.df = pd.DataFrame(rows)

  def get_samples(self):
    """Accessor for samples in this object."""
    return self.df.iterrows()

  def _get_fields(self):
    """Get the names of fields and field_types for input data."""
    # If CSV input, assume that first row contains labels
    if self.input_type == "csv":
      return self._get_raw_samples(self.input_file).next()
    elif self.input_type == "pandas":
      df = load_from_disk(self.input_file)
      return df.keys()
    elif self.input_type == "sdf":
      sample_mol = self.get_rows(self.input_file).next()
      return list(sample_mol.GetPropNames())
    else:
      raise ValueError("Unrecognized extension for %s" % self.input_file)

  def _get_input_type(self):
    """Get type of input file. Must be csv/pkl.gz/sdf file."""
    filename, file_extension = os.path.splitext(self.input_file)
    # If gzipped, need to compute extension again
    if file_extension == ".gz":
      filename, file_extension = os.path.splitext(filename)
    if file_extension == "csv":
      return "csv"
    elif file_extension == "pkl":
      return "pandas"
    elif file_extension == "sdf":
      return "sdf"
    else:
      raise ValueError("Unrecognized extension for %s" % input_file)

  def _get_raw_samples(self):
    """Returns an iterator over all rows in input_file"""
    input_type = self.get_input_type(self.input_file)
    if input_type == "csv":
      with open(self.input_file, "rb") as inp_file_obj:
        for row in csv.reader(inp_file_obj):
          if row is not None:
            yield row
    elif input_type == "pandas":
      dataframe = load_from_disk(self.input_file)
      for row in dataframe.iterrows():
        yield row
    elif input_type == "sdf":
      if ".gz" in self.input_file:
        with gzip.open(self.input_file) as inp_file_obj:
          supp = Chem.ForwardSDMolSupplier(inp_file_obj)
          for mol in supp:
            if mol is not None:
              yield mol
      else:
        with open(self.input_file) as inp_file_obj:
          supp = Chem.ForwardSDMolSupplier(inp_file_obj)
          mols = [mol for mol in supp if mol is not None]
          for mol in supp:
            if mol is not None:
              yield mol

  def _process_raw_sample(self, row):
    """Extract information from row data."""
    data = {}
    if self.input_type == "csv":
      for ind, field in enumerate(self.fields):
        data[field] = _process_field(row[ind])
      return data
    elif self.input_type == "pandas":
      # pandas rows are tuples (row_num, data)
      row = row[1]
      for field in self.fields:
        data[field] = _process_field(row[field])
    elif self.input_type == "sdf":
      mol = row
      for field in self.fields:
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


def add_vs_utils_features(df, featuretype, log_every_n=1000):
  """Generates circular fingerprints for dataset."""
  if featuretype == "fingerprints":
    featurizer = CircularFingerprint(size=1024)
  elif featuretype == "descriptors":
    featurizer = SimpleDescriptors()
  else:
    raise ValueError("Unsupported featuretype requested.")
  print("About to generate features for molecules")
  features, mol = [], None
  smiles = df["smiles"].tolist()
  for row_ind, row_data in enumerate(smiles):
    if row_ind % log_every_n == 0:
      print("Featurizing molecule %d" % row_ind)
    mol = Chem.MolFromSmiles(row_data)
    features.append(featurizer.featurize([mol]))
  df[featuretype] = features
  return

'''
Files to save:

(1) mol_id, smiles, [all feature fields] (X-ish)
(2) mol_id, smiles, split, [all task labels] (Y-ish)

'''

def standardize_df(ori_df, feature_fields, task_fields, smiles_field,
  split_field, id_field):
  df = pd.DataFrame([])
  df["mol_id"] = ori_df[[id_field]]
  df["smiles"] = ori_df[[smiles_field]]
  for task in task_fields:
    df[task] = ori_df[[task]]
  if split_field is not None:
    df["split"] = ori_df[[split_field]]

  if feature_fields is None:
    print("No feature field specified by user.")
  else:
    features_data = []
    for row in ori_df.iterrows():
      # pandas rows are tuples (row_num, row_data)
      row, feature_list = row[1], []
      for feature in feature_fields:
        feature_list.append(row[feature])
      features_data.append({"row": np.array(feature_list)})
    df["features"] = pd.DataFrame(features_data)

  return(df)

def featurize_input(self, feature_dir,
                    feature_fields, task_fields, smiles_field,
                    split_field, id_field, threshold):
  """Featurizes raw input data."""
  if len(fields) != len(field_types):
    raise ValueError("number of fields does not equal number of field types")
  if id_field is None:
    id_field = smiles_field

  df = extract_data(self.input_file, input_type, fields, field_types, task_fields,
                    smiles_field, threshold)
  print("Standardizing User DataFrame")
  df = standardize_df(df, feature_fields, task_fields, smiles_field,
                      split_field, id_field)
  print("Generating circular fingerprints")
  add_vs_utils_features(df, "fingerprints")
  print("Generating rdkit descriptors")
  add_vs_utils_features(df, "descriptors")

  print("Writing DataFrame")
  df_filename = os.path.join(
      feature_dir, "%s.joblib" %(os.path.splitext(os.path.basename(input_file))[0]))
  save_to_disk(df, df_filename)
  print("Finished saving.")

def featurize_inputs(feature_dir, input_files, input_type, fields, field_types,
                     feature_fields, task_fields, smiles_field,
                     split_field, id_field, threshold):

  featurize_input_partial = partial(featurize_input, feature_dir=feature_dir,
                                    input_type=input_type, fields=fields,
                                    field_types=field_types,
                                    feature_fields=feature_fields,
                                    task_fields=task_fields,
                                    smiles_field=smiles_field,
                                    split_field=split_field, id_field=id_field,
                                    threshold=threshold)

  pool = mp.Pool(int(mp.cpu_count()/2))
  pool.map(featurize_input_partial, input_files)
  pool.terminate()
