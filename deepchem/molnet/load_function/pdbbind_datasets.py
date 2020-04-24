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
from deepchem.feat import AtomicConvFeaturizer

logger = logging.getLogger(__name__)
DEFAULT_DATA_DIR = deepchem.utils.get_data_dir()
DEFAULT_DIR = deepchem.utils.get_data_dir()


def _fetch_precomputed_grid(data_dir=None, feat="grid", subset="core"):
  """Fetches precomputed grid features. Private method"""
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
    dataset, tasks = _fetch_precomputed_grid(feat=featurizer, subset=subset)

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

def download_pdbbind(data_dir=None,
                     subset="refined",
                     version="v2015",
                     interactions="protein-ligand"):
  """Downloads PDBBind data to a local directory.

  This utility function will download the raw PDBBind dataset into a
  local directory. If you don't specify `data_dir`, it will default to
  `deepchem.utils.get_data_dir()`. You can download either the 2015 or
  2019 versions of PDBBind data with this function by specifying the
  subset in question.

  Parameters
  ----------
  data_dir: Str, optional
    Specifies the directory storing the raw dataset.
  subset: str, optional
    This is one of "refined", or "other". These are subsets of
    the protein-ligand interactions (if `interactions !=
    "protein-ligand"` this field is ignored). If `version=="v2015"`,
    then this field is ignored since everything is stored in one
    big gzip together. If `version=="v2019"`, then "refined" and
    "other" are separate datasets which must be downloaded separately.
  version: str, optional
    Either "v2015" or "v2019". Only "v2015" is supported for now.
  interactions: str, optional
    Must be one of "protein-ligand", "protein-protein",
    "protein-nucleic-acid", "nucleic-acid-ligand". If
    `version=='v2015'`, only "protein-ligand" is supported.
  """
  if version == "v2015" and interactions != "protein-ligand":
    raise ValueError("Only protein-ligand interactions supported for v2015")
  if data_dir == None:
    data_dir = deepchem.utils.get_data_dir()

  if version == "v2015":
    data_folder = os.path.join(data_dir, "pdbbind", "v2015")
    dataset_file = os.path.join(data_dir, "pdbbind_v2015.tar.gz")
    if not os.path.exists(dataset_file):
      logger.warning("About to download PDBBind full dataset. Large file, 2GB")
      deepchem.utils.download_url(
          'http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/' +
          "pdbbind_v2015.tar.gz",
          dest_dir=data_dir)
    if os.path.exists(data_folder):
      logger.info("PDBBind v2015 full dataset already exists.")
    else:
      logger.info("Untarring PDBBind v2015 full dataset...")
      deepchem.utils.untargz_file(
          dataset_file, dest_dir=os.path.join(data_dir, "pdbbind"))
  elif version == "v2019":
    data_folder = os.path.join(data_dir, "pdbbind", "v2019")
    # Download the index file since we'll always need it
    index_file = os.path.join(data_dir, "PDBbind_2019_plain_text_index.tar.gz")
    index_folder = os.path.join(data_folder, "plain-text-index")
    if not os.path.exists(index_file):
      logger.info("About to download PDBBind 2019 index file.")
      deepchem.utils.download_url(
        'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/pdbbindv2019/PDBbind_2019_plain_text_index.tar.gz')

    if not os.path.exists(index_folder):
      logger.info("Untarring 2019 index dataset...")
      deepchem.utils.untargz_file(
          index_file, dest_dir=data_folder)

    if interactions == "protein-protein":
      pp_file = os.path.join(data_dir, "pdbbind_v2019_PP.tar.gz")
      if not os.path.exists(pp_file):
        logger.warning("About to download PDBBind 2019 protein-protein interactions. Large file of 688 MB")
        deepchem.utils.download_url(
          'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/pdbbindv2019/pdbbind_v2019_PP.tar.gz')
      pp_folder = os.path.join(data_folder, "PP")
      if not os.path.exists(pp_folder):
        logger.info("Untarring 2019 protein-protein dataset...")
        deepchem.utils.untargz_file(
            pp_file, dest_dir=data_folder)
    elif interactions == "protein-nucleic-acid":
      pn_file = os.path.join(data_dir, "pdbbind_v2019_PN.tar.gz")
      if not os.path.exists(pn_file):
        logger.warning("About to download PDBBind 2019 protein-nucleic-acid interactions. Large file of 229 MB")
        deepchem.utils.download_url(
          'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/pdbbindv2019/pdbbind_v2019_PN.tar.gz')
      pn_folder = os.path.join(data_folder, "PN")
      if not os.path.exists(pn_folder):
        logger.info("Untarring 2019 protein-nucleic-acid dataset...")
        deepchem.utils.untargz_file(
            pn_file, dest_dir=data_folder)
    elif interactions == "nucleic-acid-ligand":
      nl_file = os.path.join(data_dir, "pdbbind_v2019_NL.tar.gz")
      if not os.path.exists(nl_file):
        logger.warning("About to download PDBBind 2019 nucleic-acid-ligand interactions. File of 17 MB")
        deepchem.utils.download_url(
          'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/pdbbindv2019/pdbbind_v2019_NL.tar.gz')
      nl_folder = os.path.join(data_folder, "NL")
      if not os.path.exists(nl_folder):
        logger.info("Untarring 2019 nucleic-acid-ligand dataset...")
        deepchem.utils.untargz_file(
            nl_file, dest_dir=data_folder)
    elif interactions == "protein-ligand" and subset=="refined":
      pl_refined_file = os.path.join(data_dir, "pdbbind_v2019_refined.tar.gz")
      if not os.path.exists(pl_refined_file):
        logger.warning("About to download PDBBind 2019 protein-ligand refined interactions. Large File of 622 MB")
        deepchem.utils.download_url(
          'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/pdbbindv2019/pdbbind_v2019_refined.tar.gz')
      pl_refined_folder = os.path.join(data_folder, "refined-set")
      if not os.path.exists(pl_refined_folder):
        logger.info("Untarring 2019 protein-ligand refined dataset...")
        deepchem.utils.untargz_file(
            pl_refined_file, dest_dir=data_folder)
    elif interactions == "protein-ligand" and subset=="other":
      pl_other_file = os.path.join(data_dir, "pdbbind_v2019_other_PL.tar.gz")
      if not os.path.exists(pl_other_file):
        logger.warning("About to download PDBBind 2019 protein-ligand other interactions. Large File of 1.6 GB")
        deepchem.utils.download_url(
          'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/pdbbindv2019/pdbbind_v2019_other_PL.tar.gz')
      pl_other_folder = os.path.join(data_folder, "v2019-other-PL")
      if not os.path.exists(pl_other_folder):
        logger.info("Untarring 2019 protein-ligand other dataset...")
        deepchem.utils.untargz_file(
            pl_other_file, dest_dir=data_folder)

def get_pdbbind_molecular_complex_files(data_dir=None,
                                        subset="refined",
                                        version="v2015",
                                        interactions="protein-ligand",
                                        load_binding_pocket=False):
  """Get a list of the structure files for each dataset.

  A macromolecular complex is typically specified by one or more
  molecular structure files (PDB/sdf/mol2). This function returns a
  list of the structure files associated with various PDBBind
  datasets. This is needed for downstream processing by
  ComplexFeaturizers which work from these structure files to produce
  vectorial or tensorial representations of the data.

  Parameters
  ----------
  data_dir: Str, optional
    Specifies the directory storing the raw dataset.
  subset: str, optional
    This is one of "core", "refined", "general". These are
    subsets of the protein-ligand interactions (if `interactions !=
    "protein-ligand"` this field is ignored). Note that v2019 doesn't
    have an included "core" set. 
  version: str, optional
    Either "v2015" or "v2019". Only "v2015" is supported for now.
  interactions: str, optional
    Must be one of "protein-ligand", "protein-protein",
    "protein-nucleic-acid", "nucleic-acid-ligand". If
    `version=='v2015'`, only "protein-ligand" is supported.
  load_binding_pocket: Bool, optional
    Load binding pocket or full protein. Only valid for v2015
    protein-ligand dataset.
                            
  Returns
  -------
  List. If a molecular complex has multiple files they are returned
  together as tuple.
  """
  if data_dir == None:
    data_dir = deepchem.utils.get_data_dir()

  if version == "v2015":
    data_folder = os.path.join(data_dir, "pdbbind", "v2015")
    if subset == "core":
      index_labels_file = os.path.join(data_folder, "INDEX_core_data.2013")
    elif subset == "refined":
      index_labels_file = os.path.join(data_folder, "INDEX_refined_data.2015")
    elif subset == "general":
      index_labels_file = os.path.join(data_folder, "INDEX_general_PL_data.2015")
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
    return list(zip(protein_files, ligand_files))
  elif version == "v2019":
    data_folder = os.path.join(data_dir, "pdbbind", "v2019")
    if interactions == "protein-protein":
      index_labels_file = os.path.join(data_folder, "plain-text-index", "index", "INDEX_general_PP.2019")

      with open(index_labels_file, "r") as g:
        pdbs = [line[:4] for line in g.readlines() if line[0] != "#"]
      pp_folder = os.path.join(data_folder, "PP")
      protein_files = [
          os.path.join(pp_folder, "%s.ent.pdb" % pdb) for pdb in pdbs
      ]
      return protein_files
    elif interactions == "protein-nucleic-acid":
      index_labels_file = os.path.join(data_folder, "plain-text-index", "index", "INDEX_general_PN.2019")

      with open(index_labels_file, "r") as g:
        pdbs = [line[:4] for line in g.readlines() if line[0] != "#"]
      pn_folder = os.path.join(data_folder, "PN")
      protein_files = [
          os.path.join(pn_folder, "%s.ent.pdb" % pdb) for pdb in pdbs
      ]
      return protein_files
    elif interactions == "nucleic-acid-ligand":
      index_labels_file = os.path.join(data_folder, "plain-text-index", "index", "INDEX_general_NL.2019")

      with open(index_labels_file, "r") as g:
        pdbs = [line[:4] for line in g.readlines() if line[0] != "#"]
      nl_folder = os.path.join(data_folder, "NL")
      complex_files = [
          os.path.join(nl_folder, "%s.ent.pdb" % pdb) for pdb in pdbs
      ]
      return complex_files
    elif interactions == "protein-ligand":
      if subset == "refined":
        index_labels_file = os.path.join(data_folder, "plain-text-index", "index", "INDEX_refined_data.2019")

        with open(index_labels_file, "r") as g:
          pdbs = [line[:4] for line in g.readlines() if line[0] != "#"]
        refined_folder = os.path.join(data_folder, "refined-set")
        protein_files = [
            os.path.join(refined_folder, pdb, "%s_protein.pdb" % pdb) for pdb in pdbs
        ]
        ligand_files = [
            os.path.join(data_folder, pdb, "%s_ligand.sdf" % pdb) for pdb in pdbs
        ]
        return list(zip(protein_files, ligand_files))
      elif subset == "general":
        index_labels_file = os.path.join(data_folder, "plain-text-index", "index", "INDEX_general_PL_data.2019")

        with open(index_labels_file, "r") as g:
          pdbs = [line[:4] for line in g.readlines() if line[0] != "#"]
        general_folder = os.path.join(data_folder, "v2019-other-PL")
        protein_files = [
            os.path.join(general_folder, pdb, "%s_protein.pdb" % pdb) for pdb in pdbs
        ]
        ligand_files = [
            os.path.join(data_folder, pdb, "%s_ligand.sdf" % pdb) for pdb in pdbs
        ]
        return list(zip(protein_files, ligand_files))
      else:
        raise ValueError("Other subsets not supported")
  else:
    raise ValueError("Only v2015 and v2019 versions are supported.")

def get_pdbbind_molecular_complex_labels(data_dir=None,
                                         subset="refined",
                                         version="v2015",
                                         interactions="protein-ligand",
                                         load_binding_pocket=False):
  """Get a list of the labels for each dataset.

  This function returns a list of the labels associated with various
  PDBBind datasets. Labels will be returned in the same order as the
  raw files from `get_pdbbind_molecular_complex_files`.

  Protein-ligand datasets have units -log Kd/Ki, while the
  protein-protein, protein-nucleic-acid, nucleic-acid-ligand datasets
  have units as raw Kd/Ki/IC50 values. These later files are
  occasionally thresholded, with readings such as 'Ki<1fM' converted
  to '1fM'. If you'd like to do a more refined conversion, you'll need
  to refer to the source label file.

  Parameters
  ----------
  data_dir: Str, optional
    Specifies the directory storing the raw dataset.
  subset: str, optional
    This is one of "core", "refined", "general". These are
    subsets of the protein-ligand interactions (if `interactions !=
    "protein-ligand"` this field is ignored). Note that v2019 doesn't
    have an included "core" set. 
  version: str, optional
    Either "v2015" or "v2019". Only "v2015" is supported for now.
  interactions: str, optional
    Must be one of "protein-ligand", "protein-protein",
    "protein-nucleic-acid", "nucleic-acid-ligand". If
    `version=='v2015'`, only "protein-ligand" is supported.
  load_binding_pocket: Bool, optional
    Load binding pocket or full protein. Only valid for v2015
    protein-ligand dataset.
                            
  Returns
  -------
  List. If a molecular complex has multiple files they are returned
  together as tuple.
  """
  if data_dir == None:
    data_dir = deepchem.utils.get_data_dir()

  if version == "v2015":
    data_folder = os.path.join(data_dir, "pdbbind", "v2015")
    if subset == "core":
      index_labels_file = os.path.join(data_folder, "INDEX_core_data.2013")
    elif subset == "refined":
      index_labels_file = os.path.join(data_folder, "INDEX_refined_data.2015")
    elif subset == "general":
      index_labels_file = os.path.join(data_folder, "INDEX_general_PL_data.2015")
    else:
      raise ValueError("Other subsets not supported")
    # Extract labels
    with open(index_labels_file, "r") as g:
      labels = np.array([
          # Lines have format
          # PDB code, resolution, release year, -logKd/Ki, Kd/Ki, reference, ligand name
          # The base-10 logarithm, -log kd/pk
          float(line.split()[3]) for line in g.readlines() if line[0] != "#"
      ])
    return labels
  elif version == "v2019":
    data_folder = os.path.join(data_dir, "pdbbind", "v2019")
    # process interactions that have binding data
    if interactions in ["protein-protein", "protein-nucleic-acid", "nucleic-acid-ligand"]:
      if interactions == "protein-protein":
        index_labels_file = os.path.join(data_folder, "plain-text-index", "index", "INDEX_general_PP.2019")
      elif interactions == "protein-nucleic-acid":
        index_labels_file = os.path.join(data_folder, "plain-text-index", "index", "INDEX_general_PN.2019")
      elif interactions == "nucleic-acid-ligand":
        index_labels_file = os.path.join(data_folder, "plain-text-index", "index", "INDEX_general_NL.2019")

      # Lines have format
      # PDB code, resolution, release year, binding data, reference, ligand name
      # Extract labels
      with open(index_labels_file, "r") as g:
        raw_labels = [
            line.split()[3] for line in g.readlines() if line[0] != "#"
        ]
      clean_labels = []
      # Here are few types of raw labels we can see in the source
      # data: Kd=31.8uM, IC50=0.6nM, Kd=22.8pM, Kd~1nM, Kd>500uM,
      # Ki<0.002nM, Kd=1fM, Kd=3mM.

      # We have to do two steps of processing. The first is to remove
      # the separator. Possible values are ["=", "<", "~", ">"].
      # For the inequalities, we threshold so "<1fM" becomes "1fM".
      # This may cause some distortion in learning.
      separators = ["=", "<", "~", ">"]

      # The second step we have to do is separate the units
      unit_conversions = {"mM": 1e-3, "uM":1e-6, "nM":1e-9, "pM":1e-12, "fM":1e-15}
      number = None
      for raw in raw_labels:
        for separator in separators:
          if separator in raw:
            pieces = raw.split(separator)
            # This is something like "0.6nM"
            number = pieces[-1]
            break
        if number is None:
          raise ValueError("Don't know how to parse %s" % raw)
        # Shave off units
        mantissa = float(number[:-2] )
        units = unit_conversions[number[-2:]]
        clean_labels.append(mantissa * units)
      # Make sure we didn't miss any labels in processing
      assert len(clean_labels) == len(raw_labels)
      return clean_labels
    elif interactions == "protein-ligand":
      if subset == "refined":
        index_labels_file = os.path.join(data_folder, "plain-text-index", "index", "INDEX_refined_data.2019")
      elif subset == "general":
        index_labels_file = os.path.join(data_folder, "plain-text-index", "index", "INDEX_general_PL_data.2019")
      else:
        raise ValueError("Other subsets not supported")

      # Extract labels
      with open(index_labels_file, "r") as g:
        labels = np.array([
            # Lines have format
            # PDB code, resolution, release year, -logKd/Ki, Kd/Ki, reference, ligand name
            # The base-10 logarithm, -log kd/pk
            float(line.split()[3]) for line in g.readlines() if line[0] != "#"
        ])
      return labels
  else:
    raise ValueError("Only v2015 and v2019 versions are supported.")



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
    Specifies the directory to store the featurized and splitted
    dataset when reload is False. If reload is True, it will load
    saved dataset inside save_dir.
  save_timestamp: Bool, optional
    Save featurized and splitted dataset with timestamp or not. Set it
    as True when running similar or same jobs simultaneously on
    multiple compute nodes.
  """

  pdbbind_tasks = ["-logKd/Ki"]

  if data_dir == None:
    data_dir = DEFAULT_DIR
  data_folder = os.path.join(data_dir, "pdbbind", "v2015")

  if save_dir == None:
    save_dir = os.path.join(DEFAULT_DIR, "from-pdbbind")
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
      logger.info(
          "Dataset does not exist at {}. Reconstructing...".format(save_folder))
    else:
      logger.info(
          "\nLoading featurized and splitted dataset from:\n%s\n" % save_folder)
    loaded, all_dataset, transformers = deepchem.utils.save.load_dataset_from_disk(
        save_folder)
    if loaded:
      return pdbbind_tasks, all_dataset, transformers

  dataset_file = os.path.join(data_dir, "pdbbind_v2015.tar.gz")
  if not os.path.exists(dataset_file):
    logger.warning("About to download PDBBind full dataset. Large file, 2GB")
    deepchem.utils.download_url(
        'http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/' +
        "pdbbind_v2015.tar.gz",
        dest_dir=data_dir)
  if os.path.exists(data_folder):
    logger.info("PDBBind full dataset already exists.")
  else:
    logger.info("Untarring full dataset...")
    deepchem.utils.untargz_file(
        dataset_file, dest_dir=os.path.join(data_dir, "pdbbind"))

  logger.info("\nRaw dataset:\n%s" % data_folder)
  logger.info("\nFeaturized and splitted dataset:\n%s" % save_folder)

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
  elif featurizer == "atomic":
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
      featurizer = dc.feat.AtomicConvFeaturizer(
          frag1_num_atoms=frag1_num_atoms,
          frag2_num_atoms=frag2_num_atoms,
          complex_num_atoms=complex_num_atoms,
          max_num_neighbors=max_num_neighbors,
          neighbor_cutoff=neighbor_cutoff)
  else:
    raise ValueError("Featurizer not supported")

  logger.info("\nFeaturizing Complexes for \"%s\" ...\n" % data_folder)
  feat_t1 = time.time()
  features, failures = featurizer.featurize(ligand_files, protein_files)
  feat_t2 = time.time()
  logger.info("\nFeaturization finished, took %0.3f s." % (feat_t2 - feat_t1))

  # Delete labels and ids for failing elements
  labels = np.delete(labels, failures)
  labels = labels.reshape((len(labels), 1))
  ids = np.delete(pdbs, failures)

  logger.info("\nConstruct dataset excluding failing featurization elements...")
  dataset = deepchem.data.DiskDataset.from_numpy(features, y=labels, ids=ids)

  # No transformations of data
  transformers = []

  # Split dataset
  logger.info("\nSplit dataset...\n")
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
  logger.info("\nSaving dataset to \"%s\" ..." % save_folder)
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
    featurizer = dc.feat.AtomicConvFeaturizer(
        frag1_num_atoms, frag2_num_atoms, complex_num_atoms, max_num_neighbors,
        neighbor_cutoff)

  else:
    raise ValueError("Featurizer not supported")
  logger.info("Featurizing Complexes")
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
