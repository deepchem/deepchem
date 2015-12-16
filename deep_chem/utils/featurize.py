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
from vs_utils.utils import ScaffoldGenerator
from vs_utils.features.fingerprints import CircularFingerprint
from vs_utils.features.basic import SimpleDescriptors
from deep_chem.utils.save import save_sharded_dataset
from deep_chem.utils.save import load_sharded_dataset


def parse_float_input(val):
  """Safely parses a float input."""
  # TODO(rbharath): Correctly parse float ranges.
  try:
    if val is None:
      return val
    else:
      fval = float(val)
      return fval
  except ValueError:
    if ">" in val or "<" in val or "-" in val:
      return np.nan

#TODO(enf/rbharath): make agnostic to input type.
def get_rows(input_file, input_type):
  """Returns an iterator over all rows in input_file"""
  # TODO(rbharath): This function loads into memory, which can be painful. The
  # right option here might be to create a class which internally handles data
  # loading.
  if input_type == "csv":
    with open(input_file, "rb") as inp_file_obj:
      reader = csv.reader(inp_file_obj)
      return [row for row in reader]
  elif input_type == "pandas":
    dataframe = load_sharded_dataset(input_file)
    return dataframe.iterrows()
  elif input_type == "sdf":
    if ".gz" in input_file:
      with gzip.open(input_file) as inp_file_obj:
        supp = Chem.ForwardSDMolSupplier(inp_file_obj)
        mols = [mol for mol in supp if mol is not None]
      return mols
    else:
      with open(input_file) as inp_file_obj:
        supp = Chem.ForwardSDMolSupplier(inp_file_obj)
        mols = [mol for mol in supp if mol is not None]
      return mols

def get_colnames(row, input_type):
  """Get names of all columns."""
  if input_type == "csv":
    return row

def get_row_data(row, input_type, fields, smiles_field, colnames=None):
  """Extract information from row data."""
  row_data = {}
  if input_type == "csv":
    for ind, colname in enumerate(colnames):
      if colname in fields:
        row_data[colname] = row[ind]
  elif input_type == "pandas":
    # pandas rows are tuples (row_num, row_data)
    row = row[1]
    for field in fields:
      row_data[field] = row[field]
  elif input_type == "sdf":
    mol = row
    for field in fields:
      row_data[smiles_field] = Chem.MolToSmiles(mol)
      if not mol.HasProp(field):
        row_data[field] = None
      else:
        row_data[field] = mol.GetProp(field)
  return row_data

def process_field(data, field_type):
  """Parse data in a field."""
  if field_type == "string":
    return data
  elif field_type == "float":
    return parse_float_input(data)
  elif field_type == "list-string":
    if isinstance(data, list):
      return data
    else:
      return data.split(",")
  elif field_type == "list-float":
    return np.array(data.split(","))
  elif field_type == "ndarray":
    return data

def generate_scaffold(smiles_elt, include_chirality=False, smiles_field="smiles"):
  """Compute the Bemis-Murcko scaffold for a SMILES string."""
  smiles_string = smiles_elt[smiles_field]
  mol = Chem.MolFromSmiles(smiles_string)
  engine = ScaffoldGenerator(include_chirality=include_chirality)
  scaffold = engine.get_scaffold(mol)
  return scaffold

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

#TODO(enf/rbharath): This is broken for sdf files.
def extract_data(input_file, input_type, fields, field_types,
                 task_fields, smiles_field, threshold,
                 log_every_n=1000):
  """Extracts data from input as Pandas data frame"""
  rows = []
  colnames = []
  for row_index, raw_row in enumerate(get_rows(input_file, input_type)):
    if row_index % log_every_n == 0:
      print(row_index)
    # Skip empty rows
    if raw_row is None:
      continue
    # TODO(rbharath): The script expects that all columns in csv files
    # have column names attached. Check that this holds true somewhere
    # Get column names if csv and continue
    if input_type == "csv" and row_index == 0:
      colnames = get_colnames(raw_row, input_type)
      continue
    row, row_data = {}, get_row_data(raw_row, input_type, fields, smiles_field, colnames)
    for (field, field_type) in zip(fields, field_types):
      if field in task_fields and threshold is not None:
        raw_val = process_field(row_data[field], field_type)
        row[field] = 1 if raw_val > threshold else 0
      else:
        row[field] = process_field(row_data[field], field_type)
    #row["smiles"] = smiles.get_smiles(mol)
    rows.append(row)
  dataframe = pd.DataFrame(rows)
  return dataframe

def featurize_input(input_file, feature_dir, input_type, fields, field_types,
                    feature_fields, task_fields, smiles_field,
                    split_field, id_field, threshold):
  """Featurizes raw input data."""
  if len(fields) != len(field_types):
    raise ValueError("number of fields does not equal number of field types")
  if id_field is None:
    id_field = smiles_field

  df = extract_data(input_file, input_type, fields, field_types, task_fields,
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
  save_sharded_dataset(df, df_filename)
  print("Finished saving.")

  return

def featurize_inputs(feature_dir, input_files, input_type, fields, field_types,
                     feature_fields, task_fields, smiles_field,
                     split_field, id_field, threshold):

  featurize_input_partial = partial(featurize_input, feature_dir=feature_dir, input_type=input_type, 
                                    fields=fields, field_types=field_types, feature_fields=feature_fields,
                                    task_fields=task_fields, smiles_field=smiles_field,
                                    split_field=split_field, id_field=id_field, threshold=threshold)

  #for input_file in input_files:
  #  featurize_input_partial(input_file)
  pool = mp.Pool(mp.cpu_count())
  pool.map(featurize_input_partial, input_files)
  pool.terminate()
