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
  """Load PDBBind datasets."""
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


def extract_pdbbind(featurizer, split="random", subset="core", reload=True):
  """Load and featurize raw PDBBind dataset."""
  # TODO(rbharath): This should contain a cluster split so structures of the
  # same protein aren't in both train/test
  tasks = ["-logKd/Ki"]
  data_dir = deepchem.utils.get_data_dir()
  dataset_file = os.path.join(data_dir, "pdbbind_v2015.tar.gz")
  if subset == "core":
    data_folder = os.path.join(data_dir, "pdbbind_core")
  elif subset == "refined":
    data_folder = os.path.join(data_dir, "pdbbind_refined")
  else:
    raise ValueError("Unsupported subset %s." % subset)
  if os.path.exists(data_folder):
    logger.info("Data directory for %s already exists" % subset)
  print("dataset_file")
  print(dataset_file)
  print("os.path.exists(dataset_file)")
  print(os.path.exists(dataset_file))
  if not os.path.exists(dataset_file):
    logger.warning("About to download PDBBind full dataset. Large file, 2GB")
    deepchem.utils.download_url(
        'http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/' +
        "pdbbind_v2015.tar.gz")
  tar = tarfile.open(dataset_file, "r:gz")
  pdbs = []
  if subset == "core":
    f = tar.extractfile("v2015/INDEX_core_name.2013")
    contentlines = f.readlines()
    for line in contentlines:
      line = line.decode("utf-8")
      line = line.split(" ")
      pdb = line[0]
      print("pdb")
      print(pdb)
      print("len(pdb)")
      print(len(pdb))
      # TODO(rbharath): Why 6 instead of 4?
      if len(pdb) == 4:
        pdbs.append(pdb)
  elif subset == "refined":
    f = tar.extractfile("v2015/INDEX_refined_name.2015")
    contentlines = f.readlines()
    for line in contentlines:
      line = str(line)
      line = line.split(" ")
      pdb = line[0]
      if len(pdb) == 4:
        pdbs.append(pdb)
  else:
    raise ValueError("Other subsets not supported.")
  print("pdbs")
  print(pdbs)
  # Make dir
  if not os.path.exists(data_folder):
    os.makedirs(data_folder)
  for ind, pdb in enumerate(pdbs):
    protein_filename = "v2015/" + pdb + "/" + pdb + "_protein.pdb"
    ligand_filename = "v2015/" + pdb + "/" + pdb + "_ligand.sdf"
    print("ind")
    print(ind)
    print("protein_filename, ligand_filename")
    print(protein_filename, ligand_filename)
    protein_f = tar.extractfile(protein_filename)
    protein_lines = protein_f.readlines()
    ligand_f = tar.extractfile(ligand_filename)
    ligand_lines = ligand_f.readlines()
    print("read lines")
    protein_out = os.path.join(data_folder, pdb + "_protein.pdb")
    with open(protein_out, "w") as f:
      print("type(protein_lines)")
      print(type(protein_lines))
      f.writelines(protein_lines)
    ligand_out = os.path.join(data_folder, pdb + "_ligand.sdf")
    with open(ligand_out, "w") as f:
      f.writelines(ligand_lines)
    return pdbs
#  for member in tar.getmembers():
#    print("member.name")
#    print(member.name)
#    if member.name == "v2015/INDEX_core_name.2013":
#      f = tar.extractfile(member)
#      contentlines = f.read()
#      print("content")
#      print(content)
#      break
