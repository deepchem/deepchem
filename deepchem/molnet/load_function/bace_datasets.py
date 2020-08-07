"""
bace dataset loader.
"""
import os
import logging
import deepchem
from deepchem.molnet.load_function.bace_features import bace_user_specified_features

logger = logging.getLogger(__name__)

DEFAULT_DIR = deepchem.utils.get_data_dir()
BACE_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv"


def load_bace_regression(featurizer='ECFP',
                         split='random',
                         reload=True,
                         move_mean=True,
                         data_dir=None,
                         save_dir=None,
                         **kwargs):
  """ Load BACE dataset, regression labels

  The BACE dataset provides quantitative IC50 and qualitative (binary label)
  binding results for a set of inhibitors of human beta-secretase 1 (BACE-1).

  All data are experimental values reported in scientific literature over the
  past decade, some with detailed crystal structures available. A collection
  of 1522 compounds is provided, along with the regression labels of IC50.

  Scaffold splitting is recommended for this dataset.

  The raw data csv file contains columns below:

  - "mol" - SMILES representation of the molecular structure
  - "pIC50" - Negative log of the IC50 binding affinity
  - "class" - Binary labels for inhibitor

  References
  ----------
  .. [1] Subramanian, Govindan, et al. "Computational modeling of β-secretase 1
     (BACE-1) inhibitors using ligand based approaches." Journal of chemical
     information and modeling 56.10 (2016): 1936-1949.
  """
  # Featurize bace dataset
  logger.info("About to featurize bace dataset.")
  if data_dir is None:
    data_dir = DEFAULT_DIR
  if save_dir is None:
    save_dir = DEFAULT_DIR

  bace_tasks = ["pIC50"]

  if reload:
    save_folder = os.path.join(save_dir, "bace_r-featurized")
    if not move_mean:
      save_folder = os.path.join(save_folder, str(featurizer) + "_mean_unmoved")
    else:
      save_folder = os.path.join(save_folder, str(featurizer))

    if featurizer == "smiles2img":
      img_spec = kwargs.get("img_spec", "std")
      save_folder = os.path.join(save_folder, img_spec)
    save_folder = os.path.join(save_folder, str(split))

    loaded, all_dataset, transformers = deepchem.utils.save.load_dataset_from_disk(
        save_folder)
    if loaded:
      return bace_tasks, all_dataset, transformers

  dataset_file = os.path.join(data_dir, "bace.csv")
  if not os.path.exists(dataset_file):
    deepchem.utils.download_url(url=BACE_URL, dest_dir=data_dir)

  if featurizer == 'ECFP':
    featurizer = deepchem.feat.CircularFingerprint(size=1024)
  elif featurizer == 'GraphConv':
    featurizer = deepchem.feat.ConvMolFeaturizer()
  elif featurizer == 'Weave':
    featurizer = deepchem.feat.WeaveFeaturizer()
  elif featurizer == 'Raw':
    featurizer = deepchem.feat.RawFeaturizer()
  elif featurizer == 'UserDefined':
    featurizer = deepchem.feat.UserDefinedFeaturizer(
        bace_user_specified_features)
  elif featurizer == "smiles2img":
    img_spec = kwargs.get("img_spec", "std")
    img_size = kwargs.get("img_size", 80)
    featurizer = deepchem.feat.SmilesToImage(
        img_size=img_size, img_spec=img_spec)

  loader = deepchem.data.CSVLoader(
      tasks=bace_tasks, smiles_field="mol", featurizer=featurizer)

  dataset = loader.featurize(dataset_file, shard_size=8192)
  if split is None:
    # Initialize transformers
    transformers = [
        deepchem.trans.NormalizationTransformer(
            transform_y=True, dataset=dataset, move_mean=move_mean)
    ]

    logger.info("Split is None, about to transform data")
    for transformer in transformers:
      dataset = transformer.transform(dataset)

    return bace_tasks, (dataset, None, None), transformers

  splitters = {
      'index': deepchem.splits.IndexSplitter(),
      'random': deepchem.splits.RandomSplitter(),
      'scaffold': deepchem.splits.ScaffoldSplitter(),
      'stratified': deepchem.splits.SingletaskStratifiedSplitter()
  }
  splitter = splitters[split]
  logger.info("About to split data using {} splitter".format(split))
  frac_train = kwargs.get("frac_train", 0.8)
  frac_valid = kwargs.get('frac_valid', 0.1)
  frac_test = kwargs.get('frac_test', 0.1)

  train, valid, test = splitter.train_valid_test_split(
      dataset,
      frac_train=frac_train,
      frac_valid=frac_valid,
      frac_test=frac_test)

  transformers = [
      deepchem.trans.NormalizationTransformer(
          transform_y=True, dataset=train, move_mean=move_mean)
  ]

  logger.info("About to transform data.")
  for transformer in transformers:
    train = transformer.transform(train)
    valid = transformer.transform(valid)
    test = transformer.transform(test)

  if reload:
    deepchem.utils.save.save_dataset_to_disk(save_folder, train, valid, test,
                                             transformers)
  return bace_tasks, (train, valid, test), transformers


def load_bace_classification(featurizer='ECFP',
                             split='random',
                             reload=True,
                             data_dir=None,
                             save_dir=None,
                             **kwargs):
  """ Load BACE dataset, classification labels

  BACE dataset with classification labels ("class").
  """
  # Featurize bace dataset
  logger.info("About to featurize bace dataset.")
  if data_dir is None:
    data_dir = DEFAULT_DIR
  if save_dir is None:
    save_dir = DEFAULT_DIR

  bace_tasks = ["Class"]

  if reload:
    save_folder = os.path.join(save_dir, "bace_c-featurized", str(featurizer))
    if featurizer == "smiles2img":
      img_spec = kwargs.get("img_spec", "std")
      save_folder = os.path.join(save_folder, img_spec)
    save_folder = os.path.join(save_folder, str(split))

    loaded, all_dataset, transformers = deepchem.utils.save.load_dataset_from_disk(
        save_folder)
    if loaded:
      return bace_tasks, all_dataset, transformers

  dataset_file = os.path.join(data_dir, "bace.csv")
  if not os.path.exists(dataset_file):
    deepchem.utils.download_url(url=BACE_URL, dest_dir=data_dir)

  if featurizer == 'ECFP':
    featurizer = deepchem.feat.CircularFingerprint(size=1024)
  elif featurizer == 'GraphConv':
    featurizer = deepchem.feat.ConvMolFeaturizer()
  elif featurizer == 'Weave':
    featurizer = deepchem.feat.WeaveFeaturizer()
  elif featurizer == 'Raw':
    featurizer = deepchem.feat.RawFeaturizer()
  elif featurizer == 'UserDefined':
    featurizer = deepchem.feat.UserDefinedFeaturizer(
        bace_user_specified_features)
  elif featurizer == "smiles2img":
    img_spec = kwargs.get("img_spec", "std")
    img_size = kwargs.get("img_size", 80)
    featurizer = deepchem.feat.SmilesToImage(
        img_size=img_size, img_spec=img_spec)

  loader = deepchem.data.CSVLoader(
      tasks=bace_tasks, smiles_field="mol", featurizer=featurizer)

  dataset = loader.featurize(dataset_file, shard_size=8192)

  if split is None:
    # Initialize transformers
    transformers = [deepchem.trans.BalancingTransformer(dataset=dataset)]

    logger.info("Split is None, about to transform data")
    for transformer in transformers:
      dataset = transformer.transform(dataset)

    return bace_tasks, (dataset, None, None), transformers

  splitters = {
      'index': deepchem.splits.IndexSplitter(),
      'random': deepchem.splits.RandomSplitter(),
      'scaffold': deepchem.splits.ScaffoldSplitter(),
      'stratified': deepchem.splits.RandomStratifiedSplitter()
  }

  splitter = splitters[split]
  logger.info("About to split data using {} splitter".format(split))
  frac_train = kwargs.get("frac_train", 0.8)
  frac_valid = kwargs.get('frac_valid', 0.1)
  frac_test = kwargs.get('frac_test', 0.1)

  train, valid, test = splitter.train_valid_test_split(
      dataset,
      frac_train=frac_train,
      frac_valid=frac_valid,
      frac_test=frac_test)

  transformers = [deepchem.trans.BalancingTransformer(dataset=train)]

  logger.info("About to transform data.")
  for transformer in transformers:
    train = transformer.transform(train)
    valid = transformer.transform(valid)
    test = transformer.transform(test)

  if reload:
    deepchem.utils.save.save_dataset_to_disk(save_folder, train, valid, test,
                                             transformers)
  return bace_tasks, (train, valid, test), transformers
