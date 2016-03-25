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
import warnings
from multiprocessing import Pool
from functools import partial
import pandas as pd

def merge_dicts(*dict_args):
  '''
  Given any number of dicts, shallow copy and merge into a new dict,
  precedence goes to key value pairs in latter dicts.
  '''
  result = {}
  for dictionary in dict_args:
    if isinstance(dictionary, list):
      print("dictionary[:100]")
      print(dictionary[:100])
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
      "--num-cores", required=1, type=int,
      help="Number of cores to use for multiprocessing.")
  parser.add_argument(
      "--id-prefix", default="CID",
      help="Location to write output csv file.")
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
      mol_dict = load_shard(shard, id_prefix)
      if mol_dict is not None: 
        all_mols.append(mol_dict)
  else:
    load_shard_partial = partial(
        load_shard, shards_dir=shards_dir, id_prefix=id_prefix)
    all_mols = worker_pool.map(load_shard_partial, shards)
  all_mols_dict = merge_dicts(*all_mols)
  return all_mols_dict

def mols_to_dict(mols, id_prefix, log_every_n=2000):
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
    mol_id = mol.GetProp(CID_name)
    mol_dict[id_prefix + mol_id] = Chem.MolToSmiles(mol, isomericSmiles=True)
  return mol_dict

def get_target_names(targets_dir):
  targets = [target for target in os.listdir(targets_dir) if "pkl.gz" in target]
  # Remove the .pkl.gz
  return [os.path.splitext(os.path.splitext(target)[0])[0] for target in targets]

def load_target(target, targets_dir):
  if "pkl.gz" not in target:
    return 
  print("Processing target %s" % target)
  target = os.path.join(targets_dir, target)
  with gzip.open(target) as f:
    df = pickle.load(f)
    data_dicts = targets_to_dicts(targets_dir, df, os.path.basename(target))
  return data_dicts

def load_targets(targets_dir, worker_pool=None):
  all_data_dicts = []
  targets = os.listdir(targets_dir)
  if worker_pool is None:
    for target in targets:
      data_dicts = load_target(target, targets_dir)
      if data_dict is not None:
        all_data_dicts += data_dicts
  else:
    load_target_partial = partial(
        load_target, targets_dir=targets_dir)
    all_data_dicts = worker_pool.map(load_target_partial, targets)
  #all_data_dict = merge_dicts(*data_dicts)
  all_data_dicts = [data_dict for data_dicts in all_data_dicts for data_dict in data_dicts]
  return all_data_dicts

# TODO(rbharath): This step is now the bottleneck. Is there a good way to speed this u]p?
def targets_to_dicts(targets_dir, df, target_name, log_every_n=50000):
  target_names = get_target_names(targets_dir) 
  data_df = pd.DataFrame(columns=(["mol_id", "outcome"] + target_names))
  #print("df['mol_id']")
  #print(df['mol_id'])
  data_df["mol_id"] = df["mol_id"]
  #print("data_df['mol_id']")
  #print(data_df['mol_id'])
  data_targets = df["target"].tolist()
  def get_outcome(row):
    if row["outcome"] == "active":
      return 1
    elif row["outcome"] == "inactive":
      return 0
    elif row["outcome"] == "inconclusive":
      return "" 
  data_outcomes = df.apply(get_outcome, axis=1).tolist()
  for ind, (target, outcome) in enumerate(zip(data_targets, data_outcomes)):
    data_df.set_value(ind, target, outcome)
    for other_target in target_names:
      if other_target != target:
        data_df.set_value(ind, other_target, "")
  #for target in target_names:
  #  data_df[target] = df[[target]]
  data_df.fillna("")
  #print("data_df.iterrows().next()[1]")
  #print(data_df.iterrows().next()[1])
  return data_df.to_dict("records")
  #print("Handling dataframe for %s" % (target_name))
  #data_dict = {}
  #target_names = get_target_names(targets_dir) 
  #for index, row in df.iterrows():
  #  row = row.to_dict()
  #  mol_id = row[str("mol_id")]
  #  if index % log_every_n == 0:
  #    print("Handling index %d in dataframe for %s" % (index, target_name))
  #  if mol_id in data_dict:
  #    data = data_dict[mol_id]
  #  else:
  #    data = {}
  #  target = row["target"]
  #  if row["outcome"] == "active":
  #    outcome = 1
  #  elif row["outcome"] == "inactive":
  #    outcome = 0
  #  elif row["outcome"] == "inconclusive":
  #    continue
  #  else:
  #    #warnings.warn("Unknown outcome on row %s" % str(row))
  #    continue
  #  data[target] = outcome
  #  data[str("mol_id")] = mol_id
  #  for target in target_names:
  #    if target not in data:
  #      # Encode missing data with an empty string.
  #      data[target] = ""
  #  data_dict[mol_id] = data 
  #return data_dict

def merge_mol_data_dicts(mol_dict, data_dicts):
  print("len(mol_dict) = %d" % len(mol_dict))
  print("len(data_dicts) = %d" % len(data_dicts))
  num_missing = 0
  merged_data = {}
  #print("mol_dict.keys()[:100]")
  #print(mol_dict.keys()[:100])
  #print("data_dicts[:100]")
  #print(data_dicts[:100])
  for ind, data_dict in enumerate(data_dicts):
    mol_id = data_dict["mol_id"]
    if mol_id not in mol_dict:
      num_missing += 1
      continue
    mol_smiles = mol_dict[mol_id]
    data_dict["smiles"] = mol_smiles
    merged_data[mol_id] = data_dict
  #for ind, mol_id in enumerate(mol_dict.keys()):
  #  mol_smiles = mol_dict[mol_id]
  #  if mol_id not in data_dict:
  #    num_missing += 1
  #    continue
  #  data = data_dict[mol_id]
  #  data[str("smiles")] = mol_smiles
  #  merged_data[mol_id] = data
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

def generate_csv(data_dir, id_prefix, out, worker_pool=None):
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

  mol_dict = load_shards(shards_dir, id_prefix, worker_pool)
  #print("About to print mol_dict")
  #print(mol_dict.items()[:10])

  data_dicts = load_targets(targets_dir, worker_pool)
  #print("About to print data_dicts")
  #print(data_dicts[:10])

  merged_dict = merge_mol_data_dicts(mol_dict, data_dicts)
  write_csv(targets_dir, merged_dict, out)

def main():
  args = parse_args()
  data_dir = args.data_dir
  id_prefix = args.id_prefix
  num_cores = args.num_cores
  out = args.out

  # Connect to running ipython server
  p = Pool(processes=num_cores)
  generate_csv(data_dir, id_prefix, out, worker_pool=p)

if __name__ == "__main__":
  main()
