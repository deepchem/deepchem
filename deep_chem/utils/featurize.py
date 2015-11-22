"""
Process an input dataset into a format suitable for machine learning.
"""
import os
import cPickle as pickle
import gzip
import functools
import itertools
import pandas as pd
import openpyxl as px
import numpy as np
import argparse
import csv
from rdkit import Chem
import subprocess
from vs_utils.utils import SmilesGenerator, ScaffoldGenerator
from vs_utils.features.fingerprints import CircularFingerprint
from vs_utils.features.basic import SimpleDescriptors

def generate_directories(name, out, feature_fields):
  """Generate directory structure for featurized dataset."""
  dataset_dir = os.path.join(out, name)
  if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)
  fingerprint_dir = os.path.join(dataset_dir, "fingerprints")
  if not os.path.exists(fingerprint_dir):
    os.makedirs(fingerprint_dir)
  descriptor_dir = os.path.join(dataset_dir, "descriptors")
  if not os.path.exists(descriptor_dir):
    os.makedirs(descriptor_dir)
  target_dir = os.path.join(dataset_dir, "targets")
  if not os.path.exists(target_dir):
    os.makedirs(target_dir)
  if feature_fields is not None:
    feature_field_dir = os.path.join(dataset_dir, "features")
    if not os.path.exists(feature_field_dir):
      os.makedirs(feature_field_dir)

  # Return names of files to be generated
  # TODO(rbharath): Explicitly passing around out_*_pkl is an encapsulation
  # failure. Remove this.
  out_y_pkl = os.path.join(target_dir, "%s.pkl.gz" % name)
  out_x_pkl = (os.path.join(feature_field_dir, "%s-features.pkl.gz" %name)
      if feature_fields is not None else None)
  return out_x_pkl, out_y_pkl

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

def generate_vs_utils_features(df, name, out, smiles_field, id_field, featuretype):
  """Generates circular fingerprints for dataset."""
  dataset_dir = os.path.join(out, name)
  feature_dir = os.path.join(dataset_dir, featuretype)
  features = os.path.join(feature_dir,
      "%s-%s.pkl.gz" % (name, featuretype))

  feature_df = pd.DataFrame([]) 
  feature_df["smiles"] = df[[smiles_field]]
  feature_df["scaffolds"] = df[[smiles_field]].apply(
    functools.partial(generate_scaffold, smiles_field=smiles_field),
    axis=1)
  feature_df["mol_id"] = df[[id_field]]

  mols = []
  for row in df.iterrows():
    # pandas rows are tuples (row_num, row_data)
    smiles = row[1][smiles_field]
    mols.append(Chem.MolFromSmiles(smiles))
  if featuretype == "fingerprints":
    featurizer = CircularFingerprint(size=1024)
  elif featuretype == "descriptors":
    featurizer = SimpleDescriptors()
  else:
    raise ValueError("Unsupported featuretype requested.")
  feature_df["features"] = pd.DataFrame(
      [{"features": feature} for feature in featurizer.featurize(mols)])

  with gzip.open(features, "wb") as f:
    pickle.dump(feature_df, f, pickle.HIGHEST_PROTOCOL)

def get_rows(input_file, input_type, delimiter):
  """Returns an iterator over all rows in input_file"""
  # TODO(rbharath): This function loads into memory, which can be painful. The
  # right option here might be to create a class which internally handles data
  # loading.
  if input_type == "xlsx":
    W = px.load_workbook(input_file, use_iterators=True)
    sheet_names = W.get_sheet_names()
    p = W.get_sheet_by_name(name=sheet_names[0])    # Take first sheet as the active sheet
    return p.iter_rows()
  elif input_type == "csv":
    with open(input_file, "rb") as f:
      reader = csv.reader(f, delimiter=delimiter)
      return [row for row in reader]
  elif input_type == "pandas":
    with gzip.open(input_file) as f:
      df = pickle.load(f)
    return df.iterrows()
  elif input_type == "sdf":
    if ".gz" in input_file:
      with gzip.open(input_file) as f:
        supp = Chem.ForwardSDMolSupplier(f)
        mols = [mol for mol in supp if mol is not None]
      return mols
    else:
      with open(input_file) as f:
        supp  = Chem.ForwardSDMolSupplier(f)
        mols = [mol for mol in supp if mol is not None]
      return mols

def get_colnames(row, input_type):
  """Get names of all columns."""
  if input_type == "xlsx":
    return [cell.internal_value for cell in row]
  elif input_type == "csv":
    return row

def get_row_data(row, input_type, fields, smiles_field, colnames=None):
  """Extract information from row data."""
  row_data = {}
  if input_type == "xlsx":
    for ind, colname in enumerate(colnames):
      if colname in fields:
        row_data[colname] = row[ind].internal_value
  elif input_type == "csv":
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
    if type(data) == list:
      return data
    else:
      return data.split(",")
  elif field_type == "list-float":
    return np.array(data.split(","))
  elif field_type == "ndarray":
    return data 

def generate_targets(df, mols, prediction_field, split_field,
    smiles_field, id_field, out_pkl):
  """Process input data file, generate labels, i.e. y"""
  #TODO(enf, rbharath): Modify package unique identifier to take user-specified 
    #unique identifier instead of assuming smiles string
  labels_df = pd.DataFrame([])
  labels_df["mol_id"] = df[[id_field]]
  labels_df["smiles"] = df[[smiles_field]]
  labels_df["prediction"] = df[[prediction_field]]
  if split_field is not None:
    labels_df["split"] = df[[split_field]]

  # Write pkl.gz file
  with gzip.open(out_pkl, "wb") as f:
    pickle.dump(labels_df, f, pickle.HIGHEST_PROTOCOL)

def generate_scaffold(smiles_elt, include_chirality=False, smiles_field="smiles"):
  smiles_string = smiles_elt[smiles_field]
  mol = Chem.MolFromSmiles(smiles_string)
  engine = ScaffoldGenerator(include_chirality=include_chirality)
  scaffold = engine.get_scaffold(mol)
  return(scaffold)

def generate_features(df, feature_fields, smiles_field, id_field, out_pkl):
  if feature_fields is None:
    print("No feature field specified by user.")
    return

  features_df = pd.DataFrame([]) 
  features_df["smiles"] = df[[smiles_field]]
  features_df["scaffolds"] = df[[smiles_field]].apply(
    functools.partial(generate_scaffold, smiles_field=smiles_field),
    axis=1)
  features_df["mol_id"] = df[[id_field]]

  features_data = []
  for row in df.iterrows():
    # pandas rows are tuples (row_num, row_data)
    row, feature_list = row[1], []
    for feature in feature_fields:
      feature_list.append(row[feature])
    features_data.append({"row": np.array(feature_list)})
  features_df["features"] = pd.DataFrame(features_data)

  with gzip.open(out_pkl, "wb") as f:
    pickle.dump(features_df, f, pickle.HIGHEST_PROTOCOL)

def extract_data(input_file, input_type, fields, field_types, 
      prediction_field, smiles_field, threshold, delimiter):
  """Extracts data from input as Pandas data frame"""
  rows, mols, smiles = [], [], SmilesGenerator()
  colnames = [] 
  for row_index, raw_row in enumerate(get_rows(input_file, input_type, delimiter)):
    print row_index
    # Skip empty rows
    if raw_row is None:
      continue
    # TODO(rbharath): The script expects that all columns in xlsx/csv files
    # have column names attached. Check that this holds true somewhere
    # Get column names if xlsx/csv and continue
    if (input_type == "xlsx" or input_type == "csv") and row_index == 0:  
      colnames = get_colnames(raw_row, input_type)
      continue
    row, row_data = {}, get_row_data(raw_row, input_type, fields, smiles_field, colnames)
    for ind, (field, field_type) in enumerate(zip(fields, field_types)):
      if field == prediction_field and threshold is not None:
        raw_val = process_field(row_data[field], field_type)
        row[field] = 1 if raw_val > threshold else 0 
      else:
        row[field] = process_field(row_data[field], field_type)
    mol = Chem.MolFromSmiles(row_data[smiles_field])
    row["smiles"] = smiles.get_smiles(mol)
    mols.append(mol)
    rows.append(row)
  df = pd.DataFrame(rows)
  return(df, mols)
