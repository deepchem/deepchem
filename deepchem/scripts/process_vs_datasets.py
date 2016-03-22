"""
Transform vs-datasets into standard-form CSV files.
"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import gzip
import argparse
import os
import csv
from rdkit import Chem
import cPickle as pickle

def parse_args(input_args=None):
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--data-dir", required=1,
      help="Directory with data to be loaded.")
  parser.add_argument(
      "--out", required=1,
      help="Location to write output csv file.")
  parser.add_argument(
      "--id-prefix", default="CID",
      help="Location to write output csv file.")
  return parser.parse_args(input_args)

def load_shards(shards_dir):
  all_mols = []
  shards = os.listdir(shards_dir)
  for shard in shards:
    if "sdf.gz" not in shard:
      continue
    print("Processing shard %s" % shard)
    shard = os.path.join(shards_dir, shard)
    with gzip.open(shard) as f:
      supp = Chem.ForwardSDMolSupplier(f)
      mols = [mol for mol in supp if mol is not None]
      all_mols += mols
  return all_mols

def mols_to_dict(mols, id_prefix):
  print("About to process molecules")
  mol_dict = {}
  for ind, mol in enumerate(mols):
    if ind % 1000 == 0:
      print("Processing mol %d" % ind)
    prop_names = mol.GetPropNames()
    CID_name = None
    for prop in prop_names:
      if "CID" in str(prop):
        CID_name = str(prop)
        break
    mol_id = mol.GetProp(CID_name)
    mol_dict[id_prefix + mol_id] = Chem.MolToSmiles(mol, isomericSmiles=True)
  return mol_dict

def get_target_names(targets_dir):
  targets = [target for target in os.listdir(targets_dir) if "pkl.gz" in target]
  # Remove the .pkl.gz
  return [os.path.splitext(os.path.splitext(target)[0])[0] for target in targets]

def load_targets(targets_dir):
  dfs = []
  targets = os.listdir(targets_dir)
  for target in targets:
    if "pkl.gz" not in target:
      continue
    print("Processing target %s" % target)
    target = os.path.join(targets_dir, target)
    with gzip.open(target) as f:
      df = pickle.load(f)
      dfs.append(df)
  return dfs

def targets_to_dict(targets_dir, dfs):
  data_dict = {}
  target_names = get_target_names(targets_dir) 
  for df_ind, df in enumerate(dfs):
    print("Handling dataframe %d" % df_ind)
    for index, row in df.iterrows():
      row = row.to_dict()
      mol_id = row[str("mol_id")]
      if index % 1000 == 0:
        print("Handling index %d in dataframe %d" % (index, df_ind))
      if mol_id in data_dict:
        data = data_dict[mol_id]
      else:
        data = {}
      target = row["target"]
      if row["outcome"] == "active":
        outcome = 1
      elif row["outcome"] == "inactive":
        outcome = 0
      else:
        raise ValueError("Invalid outcome on row %s" % str(row))
      data[target] = outcome
      data[str("mol_id")] = mol_id
      for target in target_names:
        if target not in data:
          # Encode missing data with an empty string.
          data[target] = ""
      data_dict[mol_id] = data 
  return data_dict

def merge_mol_data_dicts(mol_dict, data_dict):
  print("len(mol_dict) = %d" % len(mol_dict))
  print("len(data_dict) = %d" % len(data_dict))
  num_missing = 0
  merged_data = {}
  print("mol_dict.keys()[:100]")
  print(mol_dict.keys()[:100])
  print("data_dict.keys()[:100]")
  print(data_dict.keys()[:100])
  for ind, mol_id in enumerate(mol_dict.keys()):
    mol_smiles = mol_dict[mol_id]
    if mol_id not in data_dict:
      num_missing += 1
      continue
    data = data_dict[mol_id]
    data[str("smiles")] = mol_smiles
    merged_data[mol_id] = data
  print("Number of mismatched compounds: %d" % num_missing)
  return merged_data

def write_csv(targets_dir, merged_dict, out):
  targets = get_target_names(targets_dir)
  colnames = [str("mol_id"), str("smiles")] + targets
  with open(out, "wb") as csvfile:
    writer = csv.writer(csvfile, delimiter=str(","))
    writer.writerow(colnames)
    for index, mol_id in enumerate(merged_dict):
      if index % 1000 == 0:
        print("Writing row %d of csv" % index)
      row = []
      data = merged_dict[mol_id]
      for colname in colnames:
        row.append(data[colname])
      writer.writerow(row)

def generate_csv(data_dir, id_prefix, out):
  """Transforms a vs-dataset into a CSV file.

  Args:
    data_dir: Directory name. Should contain two subdirectories named
              "shards" and "targets". The "shards" directory should contain
              gzipped sdf files. The "targets" directory should contain
              one gzipped pkl file per molecular target.
    id_prefix: Desired prefix for compound names. Should be "TOX" or "CID".
    out: Name of csv file to write.
  """
  shards_dir = os.path.join(data_dir, "shards")
  targets_dir = os.path.join(data_dir, "targets")

  mols = load_shards(shards_dir)
  dfs = load_targets(targets_dir)

  mol_dict = mols_to_dict(mols, id_prefix)
  print("About to print mol_dict")
  print(mol_dict.items()[:10])
  print("About to print data_dict")
  data_dict = targets_to_dict(targets_dir, dfs)
  print(data_dict.items()[:10])

  merged_dict = merge_mol_data_dicts(mol_dict, data_dict)
  write_csv(targets_dir, merged_dict, out)

def main():
  args = parse_args()
  data_dir = args.data_dir
  id_prefix = args.id_prefix
  generate_csv(data_dir, id_prefix, out)

  
if __name__ == "__main__":
  main()
