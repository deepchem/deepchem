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
from rdkit import Chem
import subprocess
from vs_utils.utils import SmilesGenerator

def parse_args(input_args=None):
  """Parse command-line arguments."""
  parser = argparse.ArgumentParser()
  parser.add_argument('--data', required=1,
                      help='Input file with data.')
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

def generate_targets(xlsx_file, columns, column_types, out_pkl, out_sdf):
  """Process input data file."""
  rows, mols = [], []
  W = px.load_workbook(xlsx_file, use_iterators=True)
  p = W.get_sheet_by_name(name="Sheet1")
  smiles = SmilesGenerator()
  for row_index, row in enumerate(p.iter_rows()):
    # Skip row labels.
    if row_index == 0:
      continue
    row_data = [cell.internal_value for cell in row]
    row = {}
    for ind, (column, column_type) in enumerate(zip(columns, column_types)):
      if column_type == "string":
        row[column] = row_data[ind]
      elif column_type == "float":
        row[column] = parse_float_input(row_data[ind])
      elif column_type == "float-array" and ind = len(columns) - 1:
        row[column] = np.array(row_data[ind:])

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
  generate_targets(args.data, out_pkl, out_sdf)
  generate_fingerprints(args.name, args.out)


if __name__ == "__main__":
  main()
