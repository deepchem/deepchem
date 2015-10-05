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
  parser.add_argument('--xlsx', required=1,
                      help='Excel file with Globavir data.')
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
  fingerprint_dir = os.path.join(dataset_dir, "circular-scaffold-smiles")
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
  fingerprint_dir = os.path.join(dataset_dir, "circular-scaffold-smiles")
  shards_dir = os.path.join(dataset_dir, "shards")
  sdf = os.path.join(shards_dir, "%s-0.sdf.gz" % name)
  fingerprints = os.path.join(fingerprint_dir,
      "%s-circular-scaffolds-smiles.pkl.gz" % name)
  subprocess.call(["python", "-m", "vs_utils.scripts.featurize",
                   "--scaffolds", "--smiles",
                   sdf, fingerprints,
                   "circular", "--size", "1024"])


def generate_targets(xlsx_file, out_pkl, out_sdf):
  """Process Globavir xlsx file."""
  rows, mols = [], []
  W = px.load_workbook(xlsx_file, use_iterators=True)
  p = W.get_sheet_by_name(name="Sheet1")
  smiles = SmilesGenerator()
  for row_index, row in enumerate(p.iter_rows()):
    # Skip row labels.
    if row_index == 0:
      continue
    row_data = [cell.internal_value for cell in row]
    # TODO(rbharath): Generalize this code to work for non-Globavir data. 
    row = {
      "compound_name": row_data[0],
      "isomeric_smiles": row_data[1],
      "tdo_ic50_nm": parse_float_input(row_data[5]),
      "tdo_Ki_nm": parse_float_input(row_data[6]),
      "tdo_percent_activity_10_um": parse_float_input(row_data[7]),
      "tdo_percent_activity_1_um": parse_float_input(row_data[8]),
      "ido_ic50_nm": parse_float_input(row_data[9]),
      "ido_Ki_nm": parse_float_input(row_data[10]),
      "ido_percent_activity_10_um": parse_float_input(row_data[11]),
      "ido_percent_activity_1_um": parse_float_input(row_data[12])
    }
    mol = Chem.MolFromSmiles(row["isomeric_smiles"])
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
  generate_targets(args.xlsx, out_pkl, out_sdf)
  generate_fingerprints(args.name, args.out)


if __name__ == "__main__":
  main()
