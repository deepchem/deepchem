"""
PDBBind dataset loader.
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import multiprocessing
import os
import re
import time
from multiprocessing.pool import Pool

import deepchem as dc
import numpy as np
import pandas as pd
from deepchem.utils.rdkit_util import MoleculeLoadException
from deepchem.feat.atomic_coordinates import ComplexNeighborListFragmentAtomicCoordinates


def load_pdbbind_labels(labels_file):
  """Loads pdbbind labels as dataframe"""
  # Some complexes have labels but no PDB files. Filter these manually
  contents = []
  with open(labels_file) as f:
    for line in f:
      if line.startswith("#"):
        continue
      else:
        # Some of the ligand-names are of form (FMN ox). Use regex
        # to merge into form (FMN-ox)
        p = re.compile('\(([^\)\s]*) ([^\)\s]*)\)')
        line = p.sub('(\\1-\\2)', line)
        elts = line.split()
        contents.append(elts)
  contents_df = pd.DataFrame(
      contents,
      columns=("PDB code", "resolution", "release year", "-logKd/Ki", "Kd/Ki",
               "ignore-this-field", "reference", "ligand name"))
  return contents_df


def featurize_pdbbind(data_dir=None, feat="grid", subset="core"):
  """Featurizes pdbbind according to provided featurization"""
  tasks = ["-logKd/Ki"]
  current_dir = os.path.dirname(os.path.realpath(__file__))
  data_dir = os.path.join(current_dir, "%s_%s" % (subset, feat))
  if os.path.exists(data_dir):
    return dc.data.DiskDataset(data_dir), tasks
  pdbbind_dir = os.path.join(current_dir, "v2015")

  # Load PDBBind dataset
  if subset == "core":
    labels_file = os.path.join(pdbbind_dir, "INDEX_core_data.2013")
  elif subset == "refined":
    labels_file = os.path.join(pdbbind_dir, "INDEX_refined_data.2015")
  elif subset == "full":
    labels_file = os.path.join(pdbbind_dir, "INDEX_general_PL_data.2015")
  else:
    raise ValueError("Only core, refined, and full subsets supported.")
  print("About to load contents.")
  if not os.path.exists(labels_file):
    raise ValueError("Run get_pdbbind.sh to download dataset.")
  contents_df = load_pdbbind_labels(labels_file)
  ids = contents_df["PDB code"].values
  y = np.array([float(val) for val in contents_df["-logKd/Ki"].values])

  # Define featurizers
  if feat == "grid":
    featurizer = dc.feat.RdkitGridFeaturizer(
        voxel_width=16.0,
        feature_types="voxel_combined",
        # TODO(rbharath, enf, leswing): Figure out why pi_stack and cation_pi
        # reduce validation performance
        # voxel_feature_types=["ecfp", "splif", "hbond", "pi_stack", "cation_pi",
        # "salt_bridge"], ecfp_power=9, splif_power=9,
        voxel_feature_types=["ecfp", "splif", "hbond", "salt_bridge"],
        ecfp_power=9,
        splif_power=9,
        parallel=True,
        flatten=True)
  elif feat == "coord":
    neighbor_cutoff = 4
    max_num_neighbors = 10
    featurizer = dc.feat.NeighborListComplexAtomicCoordinates(
        max_num_neighbors, neighbor_cutoff)
  elif feat == 'atomic_conv':
    dataset = compute_atomic_conv_features(tasks, data_dir, pdbbind_dir, y, ids)
    return dataset, tasks
  else:
    raise ValueError("feat not defined.")

  # Featurize Dataset
  features = []
  y_inds = []
  time1 = time.time()
  p = Pool(multiprocessing.cpu_count())
  args = []
  for ind, pdb_code in enumerate(ids):
    args.append((ind, pdb_code, pdbbind_dir, featurizer))
  results = p.map(compute_single_pdbbind_feature, args)
  feature_len = None
  for result in results:
    if result is None:
      continue
    if feature_len is None:
      feature_len = len(result[1])
    if len(result[1]) != feature_len:
      continue
    y_inds.append(result[0])
    features.append(result[1])
  time2 = time.time()
  print("TIMING: PDBBind Featurization took %0.3f s" % (time2 - time1))
  y = y[y_inds]
  X = np.vstack(features)
  w = np.ones_like(y)

  dataset = dc.data.DiskDataset.from_numpy(X, y, w, ids, data_dir=data_dir)
  return dataset, tasks


def compute_atomic_conv_features(tasks, data_dir, pdbbind_dir, y, ids):
  frag1_num_atoms = 140
  frag2_num_atoms = 821
  complex_num_atoms = 908
  max_num_neighbors = 8
  neighbor_cutoff = 12.0
  featurizer = ComplexNeighborListFragmentAtomicCoordinates(
      frag1_num_atoms, frag2_num_atoms, complex_num_atoms, max_num_neighbors,
      neighbor_cutoff)

  w = np.ones_like(y)

  # Currently featurizes with shard_size=1
  # Dataset can be reshard: dataset = dataset.reshard(48) for example
  def shard_generator():
    for ind, pdb_code in enumerate(ids):
      print("Processing %s" % str(pdb_code))
      pdb_subdir = os.path.join(pdbbind_dir, pdb_code)
      protein_file = os.path.join(pdb_subdir, "%s_pocket.pdb" % pdb_code)
      ligand_file = os.path.join(pdb_subdir, "%s_ligand.sdf" % pdb_code)
      computed_feature = featurizer._featurize_complex(
          str(ligand_file), str(protein_file))
      if computed_feature[0] is None:
        print("Bad featurization")
        continue
      else:
        X_b = np.reshape(np.array(computed_feature), (1, 9))
        y_b = y[ind]
        w_b = w[ind]
        y_b = np.reshape(y_b, (1, -1))
        w_b = np.reshape(w_b, (1, -1))
        yield (X_b, y_b, w_b, [pdb_code])

  dataset = dc.data.DiskDataset.create_dataset(
      shard_generator(), data_dir=data_dir, tasks=tasks)

  return dataset


def compute_single_pdbbind_feature(x):
  try:
    ind, pdb_code, pdbbind_dir, featurizer = x[0], x[1], x[2], x[3]
    print("Processing complex %d, %s" % (ind, str(pdb_code)))
    pdb_subdir = os.path.join(pdbbind_dir, pdb_code)
    computed_feature = compute_pdbbind_features(featurizer, pdb_subdir,
                                                pdb_code)
    return ind, computed_feature
  except Exception as e:
    logging.warning("Unable to compute features for %s" % str(x))
    return None


def compute_pdbbind_features(grid_featurizer, pdb_subdir, pdb_code):
  """Compute features for a given complex"""
  protein_file = os.path.join(pdb_subdir, "%s_protein.pdb" % pdb_code)
  ligand_file = os.path.join(pdb_subdir, "%s_ligand.sdf" % pdb_code)
  features = grid_featurizer.featurize_complexes([ligand_file], [protein_file])
  features = np.squeeze(features)
  return features


def load_pdbbind_grid(split="index", featurizer="grid", subset="core"):
  """Load PDBBind datasets. Does not do train/test split"""
  dataset, tasks = featurize_pdbbind(feat=featurizer, subset=subset)
  dataset.w

  splitters = {
      'index': dc.splits.IndexSplitter(),
      'random': dc.splits.RandomSplitter(),
      'scaffold': dc.splits.ScaffoldSplitter(),
      'stratified': dc.splits.RandomStratifiedSplitter()
  }
  splitter = splitters[split]
  train, valid, test = splitter.train_valid_test_split(dataset)

  transformers = []
  for transformer in transformers:
    train = transformer.transform(train)
  for transformer in transformers:
    valid = transformer.transform(valid)
  for transformer in transformers:
    test = transformer.transform(test)

  return tasks, (train, valid, test), transformers


if __name__ == "__main__":
  load_pdbbind_grid()
