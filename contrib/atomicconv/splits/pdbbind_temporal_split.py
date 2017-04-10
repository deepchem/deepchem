import os
import numpy as np
import deepchem as dc
import pandas as pd
from rdkit import Chem


def load_pdbbind_labels(labels_file):
  """Loads pdbbind labels as dataframe

  Parameters
  ----------
  labels_file: str
    Location of PDBbind datafile.

  Returns
  -------
  contents_df: pd.DataFrame
    Dataframe containing contents of PDBbind datafile.

  """

  contents = []
  with open(labels_file) as f:
    for line in f:
      if line.startswith("#"):
        continue
      else:
        splitline = line.split()
        if len(splitline) == 8:
          contents.append(splitline)
        else:
          print("Incorrect data format")
          print(splitline)

  contents_df = pd.DataFrame(
      contents,
      columns=("PDB code", "resolution", "release year", "-logKd/Ki", "Kd/Ki",
               "ignore-this-field", "reference", "ligand name"))
  return contents_df


seed = 123
np.random.seed(seed)
base_dir = os.getcwd()
data_dir = os.path.join(base_dir, "refined_atomconv")
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")
pdbbind_dir = os.path.join(base_dir, "v2015")

print("Loading ids from %s" % data_dir)
d = dc.data.DiskDataset(data_dir)
ids = d.ids

pdbbind_data_file = os.path.join(pdbbind_dir, "INDEX_general_PL_data.2015")
contents_df = load_pdbbind_labels(pdbbind_data_file)
df_ids = contents_df["PDB code"].values.tolist()
df_years = contents_df["release year"].values


def shard_generator():
  for ind, pdb_code in enumerate(ids):

    i = df_ids.index(pdb_code)
    y = df_years[i]
    X = np.zeros((1, 5))
    w = np.ones((1, 1))
    yield X, y, w, pdb_code


print("Generating year dataset")
temp_d = dc.data.DiskDataset.create_dataset(shard_generator())

print("Performing Stratified split on year dataset")
s = dc.splits.SingletaskStratifiedSplitter()
train_ind, test_ind = s.train_test_indices(temp_d)

print("Using indices from Stratified splitter on pdbbind dataset")
splitter = dc.splits.IndiceSplitter(test_indices=test_ind)
train_dataset, test_dataset = splitter.train_test_split(d, train_dir, test_dir)
