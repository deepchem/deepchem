"""
Loading Kamlet-Taft (KT) parameter dataset 

  KT parameters are scales to measure and quantify the Lewis acidity and basicity of molecules. The parameters are obtained through Nuclear Magnetic Resonance (NMR) spectra.

  Random splitting is recommended for this dataset.

  The raw data csv file contains columns below:

  - "csmiles" - canonical SMILES representation of the molecular structure
  - "cid" - PubChem CID of molecules
  - "alpha" - KT paramter that quantifies the acidity of a molecule
  - "beta" - KT paramter that quantifies the basicity of a molecule

  Please refer to https://arxiv.org/pdf/2008.08078 (2020) for which the dataset was hand curated.

  References
  ----------
  [1] Marcus, Yizhak. "The properties of organic liquids that are relevant to their use as solvating solvents." Chemical Society Reviews 22, no. 6 (1993): 409-416.
"""
import os
import logging
import deepchem
from deepchem.molnet.load_function.bace_features import bace_user_specified_features

logger = logging.getLogger(__name__)

DEFAULT_DIR = deepchem.utils.get_data_dir()
KT_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/KTparameterDataset.csv"


def load_kt_dataset(featurizer='ECFP',
                         split='random',
                         reload=True,
                         move_mean=True,
                         data_dir=None,
                         save_dir=None,
                         **kwargs):

  # Featurize kt dataset
  logger.info("About to featurize kt dataset.")
  if data_dir is None:
    data_dir = DEFAULT_DIR
  if save_dir is None:
    save_dir = DEFAULT_DIR

  kt_tasks = ["alpha","beta"]

  if reload:
    save_folder = os.path.join(save_dir, "kt-featurized")
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
      return kt_tasks, all_dataset, transformers

  dataset_file = os.path.join(data_dir, "KTparameterDataset.csv")
  if not os.path.exists(dataset_file):
    deepchem.utils.download_url(url=KT_URL, dest_dir=data_dir)

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
      tasks=kt_tasks, smiles_field="csmiles", featurizer=featurizer)

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

    return kt_tasks, (dataset, None, None), transformers

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
  return kt_tasks, (train, valid, test), transformers