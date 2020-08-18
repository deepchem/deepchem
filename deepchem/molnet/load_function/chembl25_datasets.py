"""
ChEMBL dataset loader, for training ChemNet
"""
import os
import numpy as np
import logging
import gzip
import shutil
import deepchem as dc
import pickle

from deepchem.feat import SmilesToSeq, SmilesToImage
from deepchem.feat.smiles_featurizers import create_char_to_idx

CHEMBL_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/chembl_25.csv.gz"
DEFAULT_DIR = dc.utils.get_data_dir()

logger = logging.getLogger(__name__)

chembl25_tasks = [
    "MolWt", "HeavyAtomMolWt", "MolLogP", "MolMR", "TPSA", "LabuteASA",
    "HeavyAtomCount", "NHOHCount", "NOCount", "NumHAcceptors", "NumHDonors",
    "NumHeteroatoms", "NumRotatableBonds", "NumRadicalElectrons",
    "NumValenceElectrons", "NumAromaticRings", "NumSaturatedRings",
    "NumAliphaticRings", "NumAromaticCarbocycles", "NumSaturatedCarbocycles",
    "NumAliphaticCarbocycles", "NumAromaticHeterocycles",
    "NumSaturatedHeterocycles", "NumAliphaticHeterocycles", "PEOE_VSA1",
    "PEOE_VSA2", "PEOE_VSA3", "PEOE_VSA4", "PEOE_VSA5", "PEOE_VSA6",
    "PEOE_VSA7", "PEOE_VSA8", "PEOE_VSA9", "PEOE_VSA10", "PEOE_VSA11",
    "PEOE_VSA12", "PEOE_VSA13", "PEOE_VSA14", "SMR_VSA1", "SMR_VSA2",
    "SMR_VSA3", "SMR_VSA4", "SMR_VSA5", "SMR_VSA6", "SMR_VSA7", "SMR_VSA8",
    "SMR_VSA9", "SMR_VSA10", "SlogP_VSA1", "SlogP_VSA2", "SlogP_VSA3",
    "SlogP_VSA4", "SlogP_VSA5", "SlogP_VSA6", "SlogP_VSA7", "SlogP_VSA8",
    "SlogP_VSA9", "SlogP_VSA10", "SlogP_VSA11", "SlogP_VSA12", "EState_VSA1",
    "EState_VSA2", "EState_VSA3", "EState_VSA4", "EState_VSA5", "EState_VSA6",
    "EState_VSA7", "EState_VSA8", "EState_VSA9", "EState_VSA10", "EState_VSA11",
    "VSA_EState1", "VSA_EState2", "VSA_EState3", "VSA_EState4", "VSA_EState5",
    "VSA_EState6", "VSA_EState7", "VSA_EState8", "VSA_EState9", "VSA_EState10",
    "BalabanJ", "BertzCT", "Ipc", "Kappa1", "Kappa2", "Kappa3", "HallKierAlpha",
    "Chi0", "Chi1", "Chi0n", "Chi1n", "Chi2n", "Chi3n", "Chi4n", "Chi0v",
    "Chi1v", "Chi2v", "Chi3v", "Chi4v"
]


def load_chembl25(featurizer="smiles2seq",
                  split="random",
                  data_dir=None,
                  save_dir=None,
                  split_seed=None,
                  reload=True,
                  transformer_type='minmax',
                  **kwargs):
  """Loads the ChEMBL25 dataset, featurizes it, and does a split.
  Parameters
  ----------
  featurizer: str, default smiles2seq
    Featurizer to use
  split: str, default None
    Splitter to use
  data_dir: str, default None
    Directory to download data to, or load dataset from. (TODO: If None, make tmp)
  save_dir: str, default None
    Directory to save the featurized dataset to. (TODO: If None, make tmp)
  split_seed: int, default None
    Seed to be used for splitting the dataset
  reload: bool, default True
    Whether to reload saved dataset
  transformer_type: str, default minmax:
    Transformer to use
  """
  if data_dir is None:
    data_dir = DEFAULT_DIR
  if save_dir is None:
    save_dir = DEFAULT_DIR

  save_folder = os.path.join(save_dir, "chembl_25-featurized", str(featurizer))
  if featurizer == "smiles2img":
    img_spec = kwargs.get("img_spec", "std")
    save_folder = os.path.join(save_folder, img_spec)

  if reload:
    if not os.path.exists(save_folder):
      logger.warning(
          "{} does not exist. Reconstructing dataset.".format(save_folder))
    else:
      logger.info("{} exists. Restoring dataset.".format(save_folder))
      loaded, dataset, transformers = dc.utils.save.load_dataset_from_disk(
          save_folder)
      if loaded:
        return chembl25_tasks, dataset, transformers

  dataset_file = os.path.join(data_dir, "chembl_25.csv.gz")

  if not os.path.exists(dataset_file):
    logger.warning("File {} not found. Downloading dataset. (~555 MB)".format(
        dataset_file))
    dc.utils.download_url(url=CHEMBL_URL, dest_dir=data_dir)

  if featurizer == 'ECFP':
    featurizer = deepchem.feat.CircularFingerprint(size=1024)
  elif featurizer == 'GraphConv':
    featurizer = deepchem.feat.ConvMolFeaturizer()
  elif featurizer == 'Weave':
    featurizer = deepchem.feat.WeaveFeaturizer()
  elif featurizer == 'Raw':
    featurizer = deepchem.feat.RawFeaturizer()
  elif featurizer == "smiles2seq":
    max_len = kwargs.get('max_len', 250)
    pad_len = kwargs.get('pad_len', 10)
    char_to_idx = create_char_to_idx(
        dataset_file, max_len=max_len, smiles_field="smiles")
    featurizer = SmilesToSeq(
        char_to_idx=char_to_idx, max_len=max_len, pad_len=pad_len)
  elif featurizer == "smiles2img":
    img_size = kwargs.get("img_size", 80)
    img_spec = kwargs.get("img_spec", "engd")
    res = kwargs.get("res", 0.5)
    featurizer = SmilesToImage(img_size=img_size, img_spec=img_spec, res=res)

  else:
    raise ValueError(
        "Featurizer of type {} is not supported".format(featurizer))

  loader = dc.data.CSVLoader(
      tasks=chembl25_tasks, smiles_field='smiles', featurizer=featurizer)
  dataset = loader.featurize(
      input_files=[dataset_file], shard_size=10000, data_dir=save_folder)

  if split is None:
    if transformer_type == "minmax":
      transformers = [
          dc.trans.MinMaxTransformer(
              transform_X=False, transform_y=True, dataset=dataset)
      ]
    else:
      transformers = [
          dc.trans.NormalizationTransformer(
              transform_X=False, transform_y=True, dataset=dataset)
      ]

    logger.info("Split is None, about to transform dataset.")
    for transformer in transformers:
      dataset = transformer.transform(dataset)
    return chembl25_tasks, (dataset, None, None), transformers

  splitters = {
      'index': dc.splits.IndexSplitter(),
      'random': dc.splits.RandomSplitter(),
      'scaffold': dc.splits.ScaffoldSplitter(),
  }

  logger.info("About to split data with {} splitter.".format(split))
  splitter = splitters[split]

  frac_train = kwargs.get('frac_train', 4 / 6)
  frac_valid = kwargs.get('frac_valid', 1 / 6)
  frac_test = kwargs.get('frac_test', 1 / 6)

  train, valid, test = splitter.train_valid_test_split(
      dataset,
      seed=split_seed,
      frac_train=frac_train,
      frac_test=frac_test,
      frac_valid=frac_valid)
  if transformer_type == "minmax":
    transformers = [
        dc.trans.MinMaxTransformer(
            transform_X=False, transform_y=True, dataset=train)
    ]
  else:
    transformers = [
        dc.trans.NormalizationTransformer(
            transform_X=False, transform_y=True, dataset=train)
    ]

  for transformer in transformers:
    train = transformer.transform(train)
    valid = transformer.transform(valid)
    test = transformer.transform(test)

  if reload:
    dc.utils.save.save_dataset_to_disk(save_folder, train, valid, test,
                                       transformers)

  return chembl25_tasks, (train, valid, test), transformers
