"""
PDBBind dataset loader.
"""

from __future__ import division
from __future__ import unicode_literals

import logging
import multiprocessing
import os
import re
import time

import deepchem
import numpy as np
import pandas as pd
import logging
import tarfile
from deepchem.feat import rdkit_grid_featurizer as rgf
from deepchem.feat.atomic_coordinates import ComplexNeighborListFragmentAtomicCoordinates

logger = logging.getLogger(__name__)


def featurize_pdbbind(data_dir=None, feat="grid", subset="core"):
  """Featurizes pdbbind according to provided featurization"""
  tasks = ["-logKd/Ki"]
  data_dir = deepchem.utils.get_data_dir()
  pdbbind_dir = os.path.join(data_dir, "pdbbind")
  dataset_dir = os.path.join(pdbbind_dir, "%s_%s" % (subset, feat))

  if not os.path.exists(dataset_dir):
    deepchem.utils.download_url(
        'http://deepchem.io.s3-website-us-west-1.amazonaws.com/featurized_datasets/core_grid.tar.gz'
    )
    deepchem.utils.download_url(
        'http://deepchem.io.s3-website-us-west-1.amazonaws.com/featurized_datasets/full_grid.tar.gz'
    )
    deepchem.utils.download_url(
        'http://deepchem.io.s3-website-us-west-1.amazonaws.com/featurized_datasets/refined_grid.tar.gz'
    )
    if not os.path.exists(pdbbind_dir):
      os.system('mkdir ' + pdbbind_dir)
    deepchem.utils.untargz_file(
        os.path.join(data_dir, 'core_grid.tar.gz'), pdbbind_dir)
    deepchem.utils.untargz_file(
        os.path.join(data_dir, 'full_grid.tar.gz'), pdbbind_dir)
    deepchem.utils.untargz_file(
        os.path.join(data_dir, 'refined_grid.tar.gz'), pdbbind_dir)

  return deepchem.data.DiskDataset(dataset_dir), tasks


def load_pdbbind_grid(split="random",
                      featurizer="grid",
                      subset="core",
                      reload=True):
  """Load PDBBind datasets. Does not do train/test split"""
  if featurizer == 'grid':
    dataset, tasks = featurize_pdbbind(feat=featurizer, subset=subset)

    splitters = {
        'index': deepchem.splits.IndexSplitter(),
        'random': deepchem.splits.RandomSplitter(),
        'time': deepchem.splits.TimeSplitterPDBbind(dataset.ids)
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
  else:
    data_dir = deepchem.utils.get_data_dir()
    if reload:
      save_dir = os.path.join(
          data_dir, "pdbbind_" + subset + "/" + featurizer + "/" + str(split))

    dataset_file = os.path.join(data_dir, subset + "_smiles_labels.csv")

    if not os.path.exists(dataset_file):
      deepchem.utils.download_url(
          'http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/' +
          subset + "_smiles_labels.csv")

    tasks = ["-logKd/Ki"]
    if reload:
      loaded, all_dataset, transformers = deepchem.utils.save.load_dataset_from_disk(
          save_dir)
      if loaded:
        return tasks, all_dataset, transformers

    if featurizer == 'ECFP':
      featurizer = deepchem.feat.CircularFingerprint(size=1024)
    elif featurizer == 'GraphConv':
      featurizer = deepchem.feat.ConvMolFeaturizer()
    elif featurizer == 'Weave':
      featurizer = deepchem.feat.WeaveFeaturizer()
    elif featurizer == 'Raw':
      featurizer = deepchem.feat.RawFeaturizer()

    loader = deepchem.data.CSVLoader(
        tasks=tasks, smiles_field="smiles", featurizer=featurizer)
    dataset = loader.featurize(dataset_file, shard_size=8192)
    transformers = [
        deepchem.trans.NormalizationTransformer(
            transform_y=True, dataset=dataset)
    ]

    for transformer in transformers:
      dataset = transformer.transform(dataset)
    df = pd.read_csv(dataset_file)

    if split == None:
      return tasks, (dataset, None, None), transformers

    splitters = {
        'index': deepchem.splits.IndexSplitter(),
        'random': deepchem.splits.RandomSplitter(),
        'scaffold': deepchem.splits.ScaffoldSplitter(),
        'time': deepchem.splits.TimeSplitterPDBbind(np.array(df['id']))
    }
    splitter = splitters[split]
    train, valid, test = splitter.train_valid_test_split(dataset)

    if reload:
      deepchem.utils.save.save_dataset_to_disk(save_dir, train, valid, test,
                                               transformers)

    return tasks, (train, valid, test), transformers


def load_pdbbind(featurizer="grid", split="random", subset="core", reload=True):
  """Load and featurize raw PDBBind dataset.
  
  Parameters
  ----------
  data_dir: String, optional
    Specifies the data directory to store the featurized dataset.
  split: Str
    Either "random" or "index"
  feat: Str
    Either "grid" or "atomic" for grid and atomic featurizations.
  subset: Str
    Only "core" or "refined" for now.
  """
  pdbbind_tasks = ["-logKd/Ki"]
  data_dir = deepchem.utils.get_data_dir()
  if reload:
    save_dir = os.path.join(data_dir,
                            "pdbbind/" + featurizer + "/" + str(split))
    loaded, all_dataset, transformers = deepchem.utils.save.load_dataset_from_disk(
        save_dir)
    if loaded:
      return pdbbind_tasks, all_dataset, transformers
  dataset_file = os.path.join(data_dir, "pdbbind_v2015.tar.gz")
  data_folder = os.path.join(data_dir, "v2015")
  if not os.path.exists(dataset_file):
    logger.warning("About to download PDBBind full dataset. Large file, 2GB")
    deepchem.utils.download_url(
        'http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/' +
        "pdbbind_v2015.tar.gz")
  if os.path.exists(data_folder):
    logger.info("Data directory for %s already exists" % subset)
  else:
    print("Untarring full dataset")
    deepchem.utils.untargz_file(dataset_file, dest_dir=data_dir)
  if subset == "core":
    index_file = os.path.join(data_folder, "INDEX_core_name.2013")
    labels_file = os.path.join(data_folder, "INDEX_core_data.2013")
  elif subset == "refined":
    index_file = os.path.join(data_folder, "INDEX_refined_name.2013")
    labels_file = os.path.join(data_folder, "INDEX_refined_data.2013")
  else:
    raise ValueError("Other subsets not supported")
  # Extract locations of data
  pdbs = []
  with open(index_file, "r") as g:
    lines = g.readlines()
    for line in lines:
      line = line.split(" ")
      pdb = line[0]
      if len(pdb) == 4:
        pdbs.append(pdb)
  protein_files = [
      os.path.join(data_folder, pdb, "%s_protein.pdb" % pdb) for pdb in pdbs
  ]
  ligand_files = [
      os.path.join(data_folder, pdb, "%s_ligand.sdf" % pdb) for pdb in pdbs
  ]
  # Extract labels
  labels = []
  with open(labels_file, "r") as f:
    lines = f.readlines()
    for line in lines:
      # Skip comment lines
      if line[0] == "#":
        continue
      # Lines have format
      # PDB code, resolution, release year, -logKd/Ki, Kd/Ki, reference, ligand name
      line = line.split()
      # The base-10 logarithm, -log kd/pk
      log_label = line[3]
      labels.append(log_label)
  labels = np.array(labels)
  # Featurize Data
  if featurizer == "grid":
    # TODO: This is not the correct setting. Set hyperparameters correctly
    ecfp_power = 5
    splif_power = 5
    featurizer = rgf.RdkitGridFeaturizer(
        voxel_width=16.0,
        feature_types=['ecfp', 'splif', 'hbond', 'salt_bridge'],
        ecfp_power=ecfp_power,
        splif_power=splif_power,
        flatten=True)
  elif featurizer == "atomic":
    # Pulled from PDB files. For larger datasets with more PDBs, would use
    # max num atoms instead of exact.
    frag1_num_atoms = 70  # for ligand atoms
    frag2_num_atoms = 24000  # for protein atoms
    complex_num_atoms = 24060  # in total
    max_num_neighbors = 4
    # Cutoff in angstroms
    neighbor_cutoff = 4
    featurizer = ComplexNeighborListFragmentAtomicCoordinates(
        frag1_num_atoms, frag2_num_atoms, complex_num_atoms, max_num_neighbors,
        neighbor_cutoff)

  else:
    raise ValueError("Featurizer not supported")
  print("Featurizing Complexes")
  features, failures = featurizer.featurize_complexes(ligand_files,
                                                      protein_files)
  # Delete labels for failing elements
  labels = np.delete(labels, failures)
  dataset = deepchem.data.DiskDataset.from_numpy(features, labels)
  # No transformations of data
  transformers = []
  if split == None:
    return tasks, (dataset, None, None), transformers

  # TODO(rbharath): This should be modified to contain a cluster split so
  # structures of the same protein aren't in both train/test
  splitters = {
      'index': deepchem.splits.IndexSplitter(),
      'random': deepchem.splits.RandomSplitter(),
  }
  splitter = splitters[split]
  train, valid, test = splitter.train_valid_test_split(dataset)
  all_dataset = (train, valid, test)
  if reload:
    deepchem.utils.save.save_dataset_to_disk(save_dir, train, valid, test,
                                             transformers)
  return pdbbind_tasks, all_dataset, transformers
