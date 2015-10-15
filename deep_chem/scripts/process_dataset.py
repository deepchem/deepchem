"""
Process an input dataset into a format suitable for machine learning.
"""
import os
import cPickle as pickle
import gzip
import pandas as pd
import openpyxl as px
import numpy as np
import argparse
import csv
from rdkit import Chem
import subprocess
from vs_utils.utils import SmilesGenerator

def parse_args(input_args=None):
  """Parse command-line arguments."""
  parser = argparse.ArgumentParser()
  parser.add_argument('--input-file', required=1,
                      help='Input file with data.')
  parser.add_argument("--columns", required=1, nargs="+",
                      help = "Names of columns.")
  parser.add_argument('--column-types', required=1, nargs="+",
                      choices=['string', 'float', 'list', 'float-array'],
                      help='Name of dataset to process.')
  parser.add_argument("--name", required=1,
                      help="Name of the dataset.")
  parser.add_argument("--out", required=1,
                      help="Folder to generate processed dataset in.")
  return parser.parse_args(input_args)

def generate_directories(name, out):
  """Generate processed dataset."""
  dataset_dir = os.path.join(out, name)
  if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)
  fingerprint_dir = os.path.join(dataset_dir, "fingerprints")
  if not os.path.exists(fingerprint_dir):
    os.makedirs(fingerprint_dir)
  target_dir = os.path.join(dataset_dir, "targets")
  if not os.path.exists(target_dir):
    os.makedirs(target_dir)
  shards_dir = os.path.join(dataset_dir, "shards")
  if not os.path.exists(shards_dir):
    os.makedirs(shards_dir)

  # Return names of files to be generated
  out_pkl = os.path.join(target_dir, "%s.pkl.gz" % name)
  out_sdf = os.path.join(shards_dir, "%s-0.sdf.gz" % name)
  return out_pkl, out_sdf

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

def generate_fingerprints(name, out):
  dataset_dir = os.path.join(out, name)
  fingerprint_dir = os.path.join(dataset_dir, "fingerprints")
  shards_dir = os.path.join(dataset_dir, "shards")
  sdf = os.path.join(shards_dir, "%s-0.sdf.gz" % name)
  fingerprints = os.path.join(fingerprint_dir,
      "%s-fingerprints.pkl.gz" % name)
  subprocess.call(["python", "-m", "vs_utils.scripts.featurize",
                   "--scaffolds", "--smiles",
                   sdf, fingerprints,
                   "circular", "--size", "1024"])

def globavir_specs():
  columns = ["compound_name", "isomeric_smiles", "tdo_ic50_nm", "tdo_Ki_nm",
    "tdo_percent_activity_10_um", "tdo_percent_activity_1_um", "ido_ic50_nm",
    "ido_Ki_nm", "ido_percent_activity_10_um", "ido_percent_activity_1_um"]
  column_types = ["string", "string", "float", "float", "float", "float",
      "float", "float", "float", "float"]

def gen_xlsx_rows(xlxs_file):
  W = px.load_workbook(xlsx_file, use_iterators=True)
  p = W.get_sheet_by_name(name="Sheet1")
  return p.iter_rows()

def get_xlsx_row_data(row):
  return [cell.internal_value for cell in row]

def gen_csv_rows(csv_file):
  # This is a memory leak...
  f = open(csv_file, "rb")
  return csv.reader(f, delimiter="\t")

def generate_targets(input_file, columns, column_types, out_pkl, out_sdf, type="csv"):
  """Process input data file."""
  rows, mols = [], []
  smiles = SmilesGenerator()
  if type == "xlsx":
    row_gen = gen_xlsx_rows(input_file)
  elif type == "csv":
    row_gen = gen_csv_rows(input_file)
  for row_index, raw_row in enumerate(row_gen):
    print row_index
    # Skip row labels.
    if row_index == 0:
      continue
    if type == "xlsx":
      row_data = get_xlsx_row_data(raw_row)
    elif type == "csv":
      row_data = raw_row 
      
    row = {}
    for ind, (column, column_type) in enumerate(zip(columns, column_types)):
      if column_type == "string":
        row[column] = row_data[ind]
      elif column_type == "float":
        row[column] = parse_float_input(row_data[ind])
      elif column_type == "list":
        row[column] = row_data[ind].split(",")
      elif column_type == "float-array":
        row[column] = np.array(row_data[ind].split(","))

    mol = Chem.MolFromSmiles(row["smiles"])
    row["smiles"] = smiles.get_smiles(mol)
    mols.append(mol)
    rows.append(row)
  df = pd.DataFrame(rows)
  # Write pkl.gz file
  with gzip.open(out_pkl, "wb") as f:
    pickle.dump(df, f, pickle.HIGHEST_PROTOCOL)
  # Write sdf.gz file
  with gzip.open(out_sdf, "wb") as gz:
    w = Chem.SDWriter(gz)
    for mol in mols:
      w.write(mol)
    w.close()

def main():
  args = parse_args()
  out_pkl, out_sdf = generate_directories(args.name, args.out)
  generate_targets(args.input_file, args.columns, args.column_types, out_pkl, out_sdf)
  generate_fingerprints(args.name, args.out)


if __name__ == "__main__":
  main()
