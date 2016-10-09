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
import pickle
import warnings
from multiprocessing import Pool
import itertools
from functools import partial
import pandas as pd

def merge_dicts(*dict_args):
  """
  Given any number of dicts, shallow copy and merge into a new dict,
  precedence goes to key value pairs in latter dicts.
  """
  result = {}
  for dictionary in dict_args:
    result.update(dictionary)
  return result

def parse_args(input_args=None):
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--data-dir", required=1,
      help="Directory with data to be loaded.")
  parser.add_argument(
      "--out", required=1,
      help="Location to write output csv file.")
  parser.add_argument(
      "--num-cores", default=0, type=int,
      help="Number of cores to use for multiprocessing.")
  parser.add_argument(
      "--id-prefix", default="CID",
      help="Location to write output csv file.")
  parser.add_argument(
      "--overwrite", action="store_true",
      help="Overwrite partially processed files.")
  return parser.parse_args(input_args)

def load_shard(shard, shards_dir, id_prefix):
  if "sdf.gz" not in shard:
    return  
  print("Processing shard %s" % shard)
  shard = os.path.join(shards_dir, shard)
  with gzip.open(shard) as f:
    supp = Chem.ForwardSDMolSupplier(f)
    mols = [mol for mol in supp if mol is not None]
  mol_dict = mols_to_dict(mols, id_prefix)
  return mol_dict

def load_shards(shards_dir, id_prefix, worker_pool=None):
  all_mols = []
  shards = os.listdir(shards_dir)
  if worker_pool is None:
    for shard in shards:
      mol_dict = load_shard(shard, shards_dir=shards_dir, id_prefix=id_prefix)
      if mol_dict is not None: 
        all_mols.append(mol_dict)
  else:
    load_shard_partial = partial(
        load_shard, shards_dir=shards_dir, id_prefix=id_prefix)
    all_mols = worker_pool.map(load_shard_partial, shards)
  all_mols = [mol_dict for mol_dict in all_mols if mol_dict is not None]
  all_mols_dict = merge_dicts(*all_mols)
  return all_mols_dict

def mols_to_dict(mols, id_prefix, log_every_n=5000):
  """Turn list of rdkit mols to large dictionary."""
  print("About to process molecules")
  mol_dict = {}
  for ind, mol in enumerate(mols):
    if ind % log_every_n == 0:
      print("Processing mol %d" % ind)
    prop_names = mol.GetPropNames()
    CID_name = None
    for prop in prop_names:
      if "CID" in str(prop):
        CID_name = str(prop)
        break
    if CID_name is not None:
      mol_id = mol.GetProp(CID_name)
    else:
      # If mol_id is not set, then use isomeric smiles as unique identifier
      mol_id = Chem.MolToSmiles(mol, isomericSmiles=True)
    mol_dict[id_prefix + mol_id] = Chem.MolToSmiles(mol, isomericSmiles=True)
  return mol_dict

def get_target_names(targets_dir):
  """Read list of targets in collection from disk."""
  targets = [target for target in os.listdir(targets_dir)
             if "pkl.gz" in target]
  return [remove_extensions(target) for target in targets]

def process_target(target, targets_dir, overwrite):
  if "pkl.gz" not in target:
    return 
  print("Processing target %s" % target)
  target_file = os.path.join(targets_dir, target)
  with gzip.open(target_file) as f:
    df = pickle.load(f)
    csv_file = target_to_csv(
        targets_dir, df, os.path.basename(target), overwrite=overwrite)
  return csv_file

def process_targets(targets_dir, overwrite, worker_pool=None):
  csv_files = []
  targets = os.listdir(targets_dir)
  if worker_pool is None:
    for target in targets:
      csv_file = process_target(target, targets_dir, overwrite)
      if csv_file is not None:
        csv_files.append(csv_file)
  else:
    process_target_partial = partial(
        process_target, targets_dir=targets_dir, overwrite=overwrite)
    csv_files = worker_pool.map(process_target_partial, targets)
    csv_files = [csv_file for csv_file in csv_files if csv_file is not None]
  return csv_files

def remove_extensions(target_name):
  """Removes file extensions from given name"""
  target_name = os.path.basename(target_name)
  while "." in target_name:
    target_name = os.path.splitext(target_name)[0]
  return target_name

def target_to_csv(targets_dir, df, target_name, log_every_n=50000,
                  overwrite=False):
  """Converts the data in a target dataframe to a csv."""
  target = remove_extensions(target_name) 
  csv_file = os.path.join(targets_dir, target + ".csv")
  if not overwrite and os.path.isfile(csv_file):
    return csv_file
  target_names = get_target_names(targets_dir) 
  data_df = pd.DataFrame(columns=(["mol_id"] + target_names))
  data_df["mol_id"] = df["mol_id"]
  def get_outcome(row):
    if row["outcome"] == "active":
      return "1"
    elif row["outcome"] == "inactive":
      return "0"
    else:
      return "" 
  data_outcomes = df.apply(get_outcome, axis=1)
  for ind, outcome in enumerate(data_outcomes):
    data_df.set_value(ind, target, outcome)
    for other_target in target_names:
      if other_target != target:
        data_df.set_value(ind, other_target, "")
  
  #iterator = data_df.iterrows()
  data_df.fillna("")
  with open(csv_file, "wb") as f:
    data_df.to_csv(f)
  return csv_file

def join_dict_datapoints(old_record, new_record, target_names):
  """Merge two datapoints together."""
  assert old_record is not None
  # TODO(rbharath): BROKEN!
  assert new_record is not None
  assert old_record["mol_id"] == new_record["mol_id"]
  for target in target_names:
    if old_record[target] != "":
      continue 
    elif new_record[target] != "":
      old_record[target] = new_record[target]
  return old_record

def merge_mol_data_dicts(mol_dict, csv_files, target_names):
  """Merge data from target and molecule listings."""
  print("len(mol_dict) = %d" % len(mol_dict))
  num_missing = 0
  merged_data = {}
  fields = ["mol_id"] + target_names
  final_fields = ["mol_id", "smiles"] + target_names
  merged_df = pd.DataFrame(columns=final_fields)
  merge_pos, merge_map = 0, {}
  for ind, csv_file in enumerate(csv_files):
    print("Merging %d/%d targets" % (ind+1, len(csv_files)))
    data_df = pd.read_csv(csv_file, na_filter=False)
    data_df.fillna("")
    data_dicts = data_df.to_dict("records")
    for data_dict in data_dicts:
      # Trim unwanted indexing fields
      data_dict = {field: data_dict[field] for field in fields}
      mol_id = data_dict["mol_id"]
      if mol_id not in mol_dict:
        num_missing += 1
        continue
      mol_smiles = mol_dict[mol_id]
      data_dict["smiles"] = mol_smiles
      if mol_id not in merged_data:
        merged_data[mol_id] = data_dict
      else:
        merged_data[mol_id] = join_dict_datapoints(
            merged_data[mol_id], data_dict, target_names)
  print("Number of mismatched compounds: %d" % num_missing)
  return merged_data

def generate_csv(data_dir, id_prefix, out, overwrite, worker_pool=None):
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
  target_names = get_target_names(targets_dir)

  mol_dict = load_shards(shards_dir, id_prefix, worker_pool)

  csv_files = process_targets(targets_dir, overwrite, worker_pool)

  merged_dict = merge_mol_data_dicts(mol_dict, csv_files, target_names)
  merged_df = pd.DataFrame(merged_dict.values())
  merged_df.fillna("")

  with open(out, "wb") as f:
    print("Writing csv file to %s" % out)
    merged_df.to_csv(f, index=False)

def main():
  args = parse_args()
  data_dir = args.data_dir
  id_prefix = args.id_prefix
  num_cores = args.num_cores
  out = args.out
  overwrite = args.overwrite

  # Connect to running ipython server
  if num_cores > 0:
    p = Pool(processes=num_cores)
    generate_csv(data_dir, id_prefix, out, overwrite, worker_pool=p)
  else:
    generate_csv(data_dir, id_prefix, out, overwrite)

if __name__ == "__main__":
  main()
