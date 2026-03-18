import os
import numpy as np
import deepchem as dc
from rdkit import Chem

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


def compute_ligand_mol(pdb_subdir, pdb_code):
  ligand_file = os.path.join(pdb_subdir, "%s_ligand.pdb" % pdb_code)
  mol = Chem.MolFromPDBFile(str(ligand_file))
  return mol


def create_scaffold_indices(pdbbind_dir, base_dir):

  frac_train = 0.8
  frac_valid = 0.0

  scaffolds = {}
  y = np.array([0 for val in ids])
  w = np.ones_like(y)
  for ind, pdb_code in enumerate(ids):
    print("Processing %s" % str(pdb_code))
    pdb_subdir = os.path.join(pdbbind_dir, pdb_code)
    mol = compute_ligand_mol(pdb_subdir, pdb_code)
    if mol is not None:
      engine = dc.utils.ScaffoldGenerator(include_chirality=False)
      scaffold = engine.get_scaffold(mol)
      if scaffold not in scaffolds:
        scaffolds[scaffold] = [ind]
      else:
        scaffolds[scaffold].append(ind)
    else:
      print(pdb_code)

  # Sort from largest to smallest scaffold sets
  scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
  scaffold_sets = [
      scaffold_set
      for (scaffold, scaffold_set) in sorted(
          scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
  ]
  train_cutoff = frac_train * len(y)
  valid_cutoff = (frac_train + frac_valid) * len(y)
  train_inds, valid_inds, test_inds = [], [], []
  for scaffold_set in scaffold_sets:
    if len(train_inds) + len(scaffold_set) > train_cutoff:
      if len(train_inds) + len(valid_inds) + len(scaffold_set) > valid_cutoff:
        test_inds += scaffold_set
      else:
        valid_inds += scaffold_set
    else:
      train_inds += scaffold_set
  return train_inds, valid_inds, test_inds


print("Generating scaffold indices")
train_inds, _, test_inds = create_scaffold_indices(pdbbind_dir, base_dir)

print("Using indices from scaffold splitter on pdbbind dataset")
splitter = dc.splits.IndiceSplitter(test_indices=test_inds)
train_dataset, test_dataset = splitter.train_test_split(d, train_dir, test_dir)
