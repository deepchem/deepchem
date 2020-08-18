"""
PDBBind dataset loader.
"""
import logging
import multiprocessing
import os
import re
import time

import deepchem
import numpy as np
import pandas as pd
import tarfile
from deepchem.feat import rdkit_grid_featurizer as rgf
from deepchem.feat.atomic_coordinates import ComplexNeighborListFragmentAtomicCoordinates
from deepchem.feat.graph_features import AtomicConvFeaturizer

logger = logging.getLogger(__name__)
DEFAULT_DATA_DIR = deepchem.utils.get_data_dir()


def featurize_pdbbind(data_dir=None, feat="grid", subset="core"):
  """Featurizes pdbbind according to provided featurization"""
  tasks = ["-logKd/Ki"]
  data_dir = deepchem.utils.get_data_dir()
  pdbbind_dir = os.path.join(data_dir, "pdbbind")
  dataset_dir = os.path.join(pdbbind_dir, "%s_%s" % (subset, feat))

  if not os.path.exists(dataset_dir):
    deepchem.utils.download_url(
        "https://deepchemdata.s3-us-west-1.amazonaws.com/featurized_datasets/core_grid.tar.gz"
    )
    deepchem.utils.download_url(
        "https://deepchemdata.s3-us-west-1.amazonaws.com/featurized_datasets/full_grid.tar.gz"
    )
    deepchem.utils.download_url(
        "https://deepchemdata.s3-us-west-1.amazonaws.com/featurized_datasets/refined_grid.tar.gz"
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

    all_dataset = (train, valid, test)
    return tasks, all_dataset, transformers

  else:
    data_dir = deepchem.utils.get_data_dir()
    if reload:
      save_dir = os.path.join(
          data_dir, "pdbbind_" + subset + "/" + featurizer + "/" + str(split))

    dataset_file = os.path.join(data_dir, subset + "_smiles_labels.csv")

    if not os.path.exists(dataset_file):
      deepchem.utils.download_url(
          "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/" + subset +
          "_smiles_labels.csv")

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
    df = pd.read_csv(dataset_file)

    if split == None:
      transformers = [
          deepchem.trans.NormalizationTransformer(
              transform_y=True, dataset=dataset)
      ]

      logger.info("Split is None, about to transform data.")
      for transformer in transformers:
        dataset = transformer.transform(dataset)
      return tasks, (dataset, None, None), transformers

    splitters = {
        'index': deepchem.splits.IndexSplitter(),
        'random': deepchem.splits.RandomSplitter(),
        'scaffold': deepchem.splits.ScaffoldSplitter(),
        'time': deepchem.splits.TimeSplitterPDBbind(np.array(df['id']))
    }
    splitter = splitters[split]
    logger.info("About to split dataset with {} splitter.".format(split))
    train, valid, test = splitter.train_valid_test_split(dataset)

    transformers = [
        deepchem.trans.NormalizationTransformer(
            transform_y=True, dataset=train)
    ]

    logger.info("About to transform dataset.")
    for transformer in transformers:
      train = transformer.transform(train)
      valid = transformer.transform(valid)
      test = transformer.transform(test)

    if reload:
      deepchem.utils.save.save_dataset_to_disk(save_dir, train, valid, test,
                                               transformers)

    return tasks, (train, valid, test), transformers


def load_pdbbind(reload=True,
                 data_dir=None,
                 subset="core",
                 load_binding_pocket=False,
                 featurizer="grid",
                 split="random",
                 split_seed=None,
                 save_dir=None,
                 save_timestamp=False):
  """Load raw PDBBind dataset by featurization and split.

  Parameters
  ----------
  reload: Bool, optional
    Reload saved featurized and splitted dataset or not.
  data_dir: Str, optional
    Specifies the directory storing the raw dataset.
  load_binding_pocket: Bool, optional
    Load binding pocket or full protein.
  subset: Str
    Specifies which subset of PDBBind, only "core" or "refined" for now.
  featurizer: Str
    Either "grid" or "atomic" for grid and atomic featurizations.
  split: Str
    Either "random" or "index".
  split_seed: Int, optional
    Specifies the random seed for splitter.
  save_dir: Str, optional
    Specifies the directory to store the featurized and splitted dataset when
    reload is False. If reload is True, it will load saved dataset inside save_dir.
  save_timestamp: Bool, optional
    Save featurized and splitted dataset with timestamp or not. Set it as True
    when running similar or same jobs simultaneously on multiple compute nodes.
  """

  pdbbind_tasks = ["-logKd/Ki"]

  deepchem_dir = deepchem.utils.get_data_dir()

  if data_dir == None:
    data_dir = DEFAULT_DATA_DIR
  data_folder = os.path.join(data_dir, "pdbbind", "v2015")

  if save_dir == None:
    save_dir = os.path.join(DEFAULT_DATA_DIR, "from-pdbbind")
  if load_binding_pocket:
    save_folder = os.path.join(
        save_dir, "protein_pocket-%s-%s-%s" % (subset, featurizer, split))
  else:
    save_folder = os.path.join(
        save_dir, "full_protein-%s-%s-%s" % (subset, featurizer, split))

  if save_timestamp:
    save_folder = "%s-%s-%s" % (save_folder,
                                time.strftime("%Y%m%d", time.localtime()),
                                re.search("\.(.*)", str(time.time())).group(1))

  if reload:
    if not os.path.exists(save_folder):
      print(
          "Dataset does not exist at {}. Reconstructing...".format(save_folder))
    else:
      print(
          "\nLoading featurized and splitted dataset from:\n%s\n" % save_folder)
    loaded, all_dataset, transformers = deepchem.utils.save.load_dataset_from_disk(
        save_folder)
    if loaded:
      return pdbbind_tasks, all_dataset, transformers

  dataset_file = os.path.join(data_dir, "pdbbind_v2015.tar.gz")
  if not os.path.exists(dataset_file):
    logger.warning("About to download PDBBind full dataset. Large file, 2GB")
    deepchem.utils.download_url(
        "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/pdbbind_v2015.tar.gz",
        dest_dir=data_dir)
  if os.path.exists(data_folder):
    logger.info("PDBBind full dataset already exists.")
  else:
    print("Untarring full dataset...")
    deepchem.utils.untargz_file(
        dataset_file, dest_dir=os.path.join(data_dir, "pdbbind"))

  print("\nRaw dataset:\n%s" % data_folder)
  print("\nFeaturized and splitted dataset:\n%s" % save_folder)

  if subset == "core":
    index_labels_file = os.path.join(data_folder, "INDEX_core_data.2013")
  elif subset == "refined":
    index_labels_file = os.path.join(data_folder, "INDEX_refined_data.2015")
  else:
    raise ValueError("Other subsets not supported")

  # Extract locations of data
  with open(index_labels_file, "r") as g:
    pdbs = [line[:4] for line in g.readlines() if line[0] != "#"]
  if load_binding_pocket:
    protein_files = [
        os.path.join(data_folder, pdb, "%s_pocket.pdb" % pdb) for pdb in pdbs
    ]
  else:
    protein_files = [
        os.path.join(data_folder, pdb, "%s_protein.pdb" % pdb) for pdb in pdbs
    ]
  ligand_files = [
      os.path.join(data_folder, pdb, "%s_ligand.sdf" % pdb) for pdb in pdbs
  ]

  # Extract labels
  with open(index_labels_file, "r") as g:
    labels = np.array([
        # Lines have format
        # PDB code, resolution, release year, -logKd/Ki, Kd/Ki, reference, ligand name
        # The base-10 logarithm, -log kd/pk
        float(line.split()[3]) for line in g.readlines() if line[0] != "#"
    ])

  # Featurize Data
  if featurizer == "grid":
    featurizer = rgf.RdkitGridFeaturizer(
        voxel_width=2.0,
        feature_types=[
            'ecfp', 'splif', 'hbond', 'salt_bridge', 'pi_stack', 'cation_pi',
            'charge'
        ],
        flatten=True)
  elif featurizer == "atomic" or featurizer == "atomic_conv":
    # Pulled from PDB files. For larger datasets with more PDBs, would use
    # max num atoms instead of exact.
    frag1_num_atoms = 70  # for ligand atoms
    if load_binding_pocket:
      frag2_num_atoms = 1000
      complex_num_atoms = 1070
    else:
      frag2_num_atoms = 24000  # for protein atoms
      complex_num_atoms = 24070  # in total
    max_num_neighbors = 4
    # Cutoff in angstroms
    neighbor_cutoff = 4
    if featurizer == "atomic":
      featurizer = ComplexNeighborListFragmentAtomicCoordinates(
          frag1_num_atoms=frag1_num_atoms,
          frag2_num_atoms=frag2_num_atoms,
          complex_num_atoms=complex_num_atoms,
          max_num_neighbors=max_num_neighbors,
          neighbor_cutoff=neighbor_cutoff)
    if featurizer == "atomic_conv":
      featurizer = AtomicConvFeaturizer(
          labels=labels,
          frag1_num_atoms=frag1_num_atoms,
          frag2_num_atoms=frag2_num_atoms,
          complex_num_atoms=complex_num_atoms,
          neighbor_cutoff=neighbor_cutoff,
          max_num_neighbors=max_num_neighbors,
          batch_size=64)
  else:
    raise ValueError("Featurizer not supported")

  print("\nFeaturizing Complexes for \"%s\" ...\n" % data_folder)
  feat_t1 = time.time()
  features, failures = featurizer.featurize(ligand_files, protein_files)
  feat_t2 = time.time()
  print("\nFeaturization finished, took %0.3f s." % (feat_t2 - feat_t1))

  # Delete labels and ids for failing elements
  labels = np.delete(labels, failures)
  labels = labels.reshape((len(labels), 1))
  ids = np.delete(pdbs, failures)

  print("\nConstruct dataset excluding failing featurization elements...")
  dataset = deepchem.data.DiskDataset.from_numpy(features, y=labels, ids=ids)

  # No transformations of data
  transformers = []

  # Split dataset
  print("\nSplit dataset...\n")
  if split == None:
    return pdbbind_tasks, (dataset, None, None), transformers

  # TODO(rbharath): This should be modified to contain a cluster split so
  # structures of the same protein aren't in both train/test
  splitters = {
      'index': deepchem.splits.IndexSplitter(),
      'random': deepchem.splits.RandomSplitter(),
  }
  splitter = splitters[split]
  train, valid, test = splitter.train_valid_test_split(dataset, seed=split_seed)

  all_dataset = (train, valid, test)
  print("\nSaving dataset to \"%s\" ..." % save_folder)
  deepchem.utils.save.save_dataset_to_disk(save_folder, train, valid, test,
                                           transformers)
  return pdbbind_tasks, all_dataset, transformers


def load_pdbbind_from_dir(data_folder,
                          index_files,
                          featurizer="grid",
                          split="random",
                          ex_ids=[],
                          save_dir=None):
  """Load and featurize raw PDBBind dataset from a local directory with the option to avoid certain IDs.

    Parameters
    ----------
    data_dir: String,
      Specifies the data directory to store the featurized dataset.
    index_files: List
      List of data and labels index file paths relative to the path in data_dir
    split: Str
      Either "random" or "index"
    feat: Str
      Either "grid" or "atomic" for grid and atomic featurizations.
    subset: Str
      Only "core" or "refined" for now.
    ex_ids: List
      List of PDB IDs to avoid loading if present
    save_dir: String
      Path to store featurized datasets
    """
  pdbbind_tasks = ["-logKd/Ki"]

  index_file = os.path.join(data_folder, index_files[0])
  labels_file = os.path.join(data_folder, index_files[1])

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
      os.path.join(data_folder, pdb, "%s_protein.pdb" % pdb)
      for pdb in pdbs
      if pdb not in ex_ids
  ]
  ligand_files = [
      os.path.join(data_folder, pdb, "%s_ligand.sdf" % pdb)
      for pdb in pdbs
      if pdb not in ex_ids
  ]
  # Extract labels
  labels_tmp = {}
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
      labels_tmp[line[0]] = log_label

  labels = np.array([labels_tmp[pdb] for pdb in pdbs])
  print(labels)
  # Featurize Data
  if featurizer == "grid":
    featurizer = rgf.RdkitGridFeaturizer(
        voxel_width=2.0,
        feature_types=[
            'ecfp', 'splif', 'hbond', 'salt_bridge', 'pi_stack', 'cation_pi',
            'charge'
        ],
        flatten=True)
  elif featurizer == "atomic":
    # Pulled from PDB files. For larger datasets with more PDBs, would use
    # max num atoms instead of exact.
    frag1_num_atoms = 70  # for ligand atoms
    frag2_num_atoms = 24000  # for protein atoms
    complex_num_atoms = 24070  # in total
    max_num_neighbors = 4
    # Cutoff in angstroms
    neighbor_cutoff = 4
    featurizer = ComplexNeighborListFragmentAtomicCoordinates(
        frag1_num_atoms, frag2_num_atoms, complex_num_atoms, max_num_neighbors,
        neighbor_cutoff)

  else:
    raise ValueError("Featurizer not supported")
  print("Featurizing Complexes")
  features, failures = featurizer.featurize(ligand_files, protein_files)
  # Delete labels for failing elements
  labels = np.delete(labels, failures)
  dataset = deepchem.data.DiskDataset.from_numpy(features, labels)
  # No transformations of data
  transformers = []
  if split == None:
    return pdbbind_tasks, (dataset, None, None), transformers

  # TODO(rbharath): This should be modified to contain a cluster split so
  # structures of the same protein aren't in both train/test
  splitters = {
      'index': deepchem.splits.IndexSplitter(),
      'random': deepchem.splits.RandomSplitter(),
  }
  splitter = splitters[split]
  train, valid, test = splitter.train_valid_test_split(dataset)
  all_dataset = (train, valid, test)
  if save_dir:
    deepchem.utils.save.save_dataset_to_disk(save_dir, train, valid, test,
                                             transformers)
  return pdbbind_tasks, all_dataset, transformers
